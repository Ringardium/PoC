"""Batched YOLO detector for the Phase 4 multi-stream pipeline.

A single shared :class:`ultralytics.YOLO` instance receives frames from
many streams in one ``predict([f1, f2, ...])`` call. This pays the CUDA
launch / kernel setup cost once per batch instead of once per stream and
lets cuDNN pick batched kernels — the throughput uplift on A100 is
roughly 2-3× at batch=4 over per-stream sequential inference.

Tracker state stays per-stream (see ``standalone_tracker.StandaloneTracker``).

Only detection moves to the shared model; everything downstream
(per-stream tracker, ReID, behavior detection, drawing, metadata push)
is unchanged.

Concurrency model
-----------------
``BatchedDetector`` lives in the same asyncio loop as the stream tasks.
Each stream awaits :meth:`detect`, which:

1. Enqueues the frame + a future
2. Wakes the drain task
3. Returns when the drain task fulfils the future

The drain task batches up to ``max_batch`` queued requests, runs the
heavy ``model.predict`` inside an executor (so the loop stays
responsive), and dispatches per-request detections back to each future.

A short ``batch_wait`` window (default 5 ms) gives slightly out-of-phase
streams a chance to join the same batch — without it, a single fast
stream would always run alone.

Per-stream config differences are handled by:
  * predicting with the **union of all classes** in the batch
  * predicting with the **minimum conf** (most permissive), then
    filtering per-stream above that threshold
  * sharing one ``iou`` value (median of batch) — accept tiny accuracy
    drift for the throughput gain. Per-stream NMS would defeat batching.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StreamDetections:
    """Per-stream detection bundle returned by :meth:`BatchedDetector.detect`."""
    xyxy: np.ndarray            # (N, 4) corner-xyxy in source pixels
    conf: np.ndarray            # (N,) confidences
    cls: np.ndarray             # (N,) class ids (int32)
    inference_ms: float         # how long this batch took (shared across batch peers)
    batch_size: int             # how many streams shared the batch this frame was in


@dataclass
class _PendingRequest:
    stream_id: str
    frame: np.ndarray
    classes: tuple             # tuple(int) — per-stream wanted classes
    conf: float                # per-stream conf threshold
    future: "asyncio.Future[StreamDetections]"


class BatchedDetector:
    """Shared YOLO model + asyncio batching coordinator.

    Args:
        model_path: path passed to ``ultralytics.YOLO(...)``.
        device: ``"cuda:0"`` / ``"cpu"`` / etc.
        half: FP16 inference (CUDA only).
        max_batch: cap per ``predict()`` call. 4-8 typical on A100.
        batch_wait: seconds to wait for more requests to join a batch
            after the first arrives. 5 ms is a good default — small
            enough to be invisible, large enough to coalesce streams
            running at similar fps.
        iou: NMS IoU threshold shared by the batch.
        verbose: ultralytics' own log spam — usually False.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        half: bool = True,
        max_batch: int = 4,
        batch_wait: float = 0.005,
        iou: float = 0.5,
        verbose: bool = False,
    ):
        # Defer ultralytics import — keeps PoC importable on CPU-only dev boxes
        # that haven't installed the heavy deps.
        from ultralytics import YOLO

        self._model = YOLO(model_path, task="detect")
        self._device = device
        self._half = half
        self._max_batch = max(1, int(max_batch))
        self._batch_wait = max(0.0, float(batch_wait))
        self._iou = float(iou)
        self._verbose = bool(verbose)

        self._pending: List[_PendingRequest] = []
        self._lock = asyncio.Lock()  # guards _pending
        # Acquired by the drain task only — serializes inference (the GPU
        # already serializes; this prevents two batches running at once
        # from different drain triggers).
        self._infer_lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._running = True
        self._drain_task: Optional[asyncio.Task] = None
        # ThreadPoolExecutor borrowed from the loop's default; we just
        # call run_in_executor.
        self._stats_lock = threading.Lock()
        self._stats = {"batches": 0, "frames": 0, "infer_ms_sum": 0.0}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        if self._drain_task is None or self._drain_task.done():
            self._drain_task = asyncio.create_task(self._drain_loop(), name="BatchedDetector.drain")
            logger.info(
                f"BatchedDetector started — device={self._device}, half={self._half}, "
                f"max_batch={self._max_batch}, batch_wait={self._batch_wait*1000:.1f}ms"
            )

    async def stop(self):
        self._running = False
        self._wake.set()
        if self._drain_task is not None:
            try:
                await asyncio.wait_for(self._drain_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._drain_task.cancel()
        # Cancel any pending requests
        async with self._lock:
            for r in self._pending:
                if not r.future.done():
                    r.future.set_exception(RuntimeError("BatchedDetector stopped"))
            self._pending.clear()

    def get_stats(self) -> dict:
        with self._stats_lock:
            s = dict(self._stats)
        s["avg_batch"] = (s["frames"] / s["batches"]) if s["batches"] else 0.0
        s["avg_infer_ms"] = (s["infer_ms_sum"] / s["batches"]) if s["batches"] else 0.0
        return s

    # ------------------------------------------------------------------
    # Public — called from per-stream loops
    # ------------------------------------------------------------------

    async def detect(
        self,
        stream_id: str,
        frame: np.ndarray,
        classes: Sequence[int],
        conf: float,
    ) -> StreamDetections:
        """Submit one frame; await detections for that frame's classes."""
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        req = _PendingRequest(
            stream_id=stream_id,
            frame=frame,
            classes=tuple(int(c) for c in (classes or [])),
            conf=float(conf),
            future=future,
        )
        async with self._lock:
            self._pending.append(req)
            self._wake.set()
        return await future

    # ------------------------------------------------------------------
    # Internal — drain loop
    # ------------------------------------------------------------------

    async def _drain_loop(self):
        loop = asyncio.get_event_loop()
        while self._running:
            await self._wake.wait()
            self._wake.clear()
            if not self._running:
                break
            # Tiny window to coalesce more requests into the same batch.
            if self._batch_wait > 0:
                await asyncio.sleep(self._batch_wait)

            async with self._lock:
                if not self._pending:
                    continue
                batch = self._pending[: self._max_batch]
                self._pending = self._pending[self._max_batch :]
                # If anything remains, wake immediately for the next batch.
                if self._pending:
                    self._wake.set()

            await self._infer_lock.acquire()
            try:
                await loop.run_in_executor(None, self._run_batch, batch)
            except Exception as e:
                logger.error(f"BatchedDetector batch failed: {e}")
                for r in batch:
                    if not r.future.done():
                        r.future.set_exception(e)
            finally:
                self._infer_lock.release()

    # ------------------------------------------------------------------
    # Internal — single batched predict (runs in executor thread)
    # ------------------------------------------------------------------

    def _run_batch(self, batch: List[_PendingRequest]):
        if not batch:
            return

        # Predict with the union of classes & minimum conf so each stream
        # can post-filter to its own settings.
        cls_union = sorted({c for r in batch for c in r.classes}) or None
        conf_min = min((r.conf for r in batch), default=0.25)

        frames = [r.frame for r in batch]

        t0 = time.perf_counter()
        try:
            results = self._model.predict(
                frames,
                classes=cls_union,
                conf=conf_min,
                iou=self._iou,
                device=self._device,
                half=self._half,
                verbose=self._verbose,
            )
        except Exception as e:
            for r in batch:
                if not r.future.done():
                    r.future.set_exception(e)
            return
        infer_ms = (time.perf_counter() - t0) * 1000.0

        with self._stats_lock:
            self._stats["batches"] += 1
            self._stats["frames"] += len(batch)
            self._stats["infer_ms_sum"] += infer_ms

        # Dispatch per-request, applying per-stream conf filter
        # (predict() already restricted classes to cls_union).
        for req, res in zip(batch, results):
            try:
                if res.boxes is None or res.boxes.xyxy is None or len(res.boxes) == 0:
                    out = StreamDetections(
                        xyxy=np.zeros((0, 4), dtype=np.float32),
                        conf=np.zeros((0,), dtype=np.float32),
                        cls=np.zeros((0,), dtype=np.int32),
                        inference_ms=infer_ms,
                        batch_size=len(batch),
                    )
                else:
                    xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32, copy=False)
                    cf = res.boxes.conf.cpu().numpy().astype(np.float32, copy=False)
                    cl = res.boxes.cls.cpu().numpy().astype(np.int32, copy=False)
                    keep = cf >= req.conf
                    if req.classes:
                        wanted = np.fromiter(req.classes, dtype=np.int32)
                        keep = keep & np.isin(cl, wanted)
                    out = StreamDetections(
                        xyxy=xyxy[keep],
                        conf=cf[keep],
                        cls=cl[keep],
                        inference_ms=infer_ms,
                        batch_size=len(batch),
                    )
                if not req.future.done():
                    req.future.set_result(out)
            except Exception as e:
                if not req.future.done():
                    req.future.set_exception(e)
