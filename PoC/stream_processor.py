"""Multi-stream processor for real-time pet tracking with web streaming support."""

import asyncio
import logging
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from config import SystemConfig, StreamConfig, GPUConfig, ProcessingConfig
from monitor import StatsAggregator

logger = logging.getLogger(__name__)

# Add parent directory for tracking/detection imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class StreamState:
    """Per-stream mutable state for tracking and behavior detection."""

    def __init__(self, config: StreamConfig):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_cnt: int = 0
        self.source_fps: int = 30
        self.width: int = 0
        self.height: int = 0
        # Behavior detection state
        self.inert_coor: Dict = {}
        self.sleep_coor: Dict = {}
        self.sleep_bbox: Dict = {}
        self.eat_coor: Dict = {}
        self.eat_near_count: Dict = {}
        self.bathroom_coor: Dict = {}
        self.bathroom_bbox: Dict = {}
        self.close_count: Optional[torch.Tensor] = None
        self.far_count: Optional[torch.Tensor] = None
        # DeepSORT tracker
        self.tracker = None
        # Output writer
        self.writer: Optional[cv2.VideoWriter] = None
        self.active: bool = False

    def init_behavior_state(self, max_number: int = 500):
        self.close_count = torch.zeros((max_number, max_number))
        self.far_count = torch.zeros((max_number, max_number))


class MultiStreamProcessor:
    """Process multiple video/RTSP streams with optional YOLO tracking and behavior detection."""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.running = False
        self.streams: Dict[str, StreamConfig] = {}
        self.web_frames: Dict[str, bytes] = {}
        self.web_frames_lock = threading.Lock()

        self._states: Dict[str, StreamState] = {}
        self._stats = StatsAggregator()
        self._executor = ThreadPoolExecutor(max_workers=max(len(config.streams), 4))
        self._stream_tasks: Dict[str, asyncio.Task] = {}
        self._stats_task: Optional[asyncio.Task] = None
        self._model = None
        self._model_lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Register initial streams
        for sc in config.streams:
            self._register_stream(sc)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _needs_model(self) -> bool:
        """Check if any stream requires YOLO inference."""
        for sc in self.streams.values():
            if sc.yolo_classes:
                return True
        return False

    def _load_model(self):
        """Load YOLO model if needed."""
        if self._model is not None:
            return
        if not self._needs_model():
            logger.info("No streams require YOLO — skipping model load")
            return

        from ultralytics import YOLO

        model_path = self.config.model_path
        logger.info(f"Loading YOLO model: {model_path}")
        self._model = YOLO(model_path)

        # GPU optimizations
        gpu = self.config.gpu
        if torch.cuda.is_available():
            if gpu.enable_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            if gpu.enable_cudnn_benchmark:
                torch.backends.cudnn.benchmark = True

    # ------------------------------------------------------------------
    # Stream lifecycle
    # ------------------------------------------------------------------

    def _register_stream(self, sc: StreamConfig):
        self.streams[sc.stream_id] = sc
        self._states[sc.stream_id] = StreamState(sc)
        self._stats.register_stream(sc.stream_id)

    def _unregister_stream(self, stream_id: str):
        state = self._states.get(stream_id)
        if state:
            state.active = False
            if state.cap and state.cap.isOpened():
                state.cap.release()
            if state.writer:
                state.writer.release()
        self.streams.pop(stream_id, None)
        self._states.pop(stream_id, None)
        with self.web_frames_lock:
            self.web_frames.pop(stream_id, None)

    def add_stream_dynamic(self, sc: StreamConfig) -> bool:
        """Add a stream at runtime. Returns False if max streams reached."""
        if len(self.streams) >= self.config.processing.max_streams:
            return False
        self._register_stream(sc)
        # If already running, start the stream task
        if self.running and self._loop:
            task = self._loop.create_task(self._stream_loop(sc.stream_id))
            self._stream_tasks[sc.stream_id] = task
        return True

    def remove_stream_dynamic(self, stream_id: str):
        """Remove a stream at runtime."""
        task = self._stream_tasks.pop(stream_id, None)
        if task:
            task.cancel()
        self._unregister_stream(stream_id)

    # ------------------------------------------------------------------
    # Main lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Start processing all streams (async entry point)."""
        self.running = True
        self._loop = asyncio.get_running_loop()

        self._load_model()

        # Start per-stream loops
        for sid in list(self.streams):
            task = asyncio.create_task(self._stream_loop(sid))
            self._stream_tasks[sid] = task

        # Start stats printer
        self._stats_task = asyncio.create_task(self._stats_loop())

        # Wait until stopped
        try:
            while self.running:
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
        finally:
            await self._cleanup()

    def stop(self):
        """Signal all processing to stop."""
        self.running = False

    async def _cleanup(self):
        for task in self._stream_tasks.values():
            task.cancel()
        if self._stats_task:
            self._stats_task.cancel()
        for state in self._states.values():
            if state.cap and state.cap.isOpened():
                state.cap.release()
            if state.writer:
                state.writer.release()
        self._executor.shutdown(wait=False)
        self._stats.cleanup()

    # ------------------------------------------------------------------
    # Per-stream processing loop
    # ------------------------------------------------------------------

    async def _stream_loop(self, stream_id: str):
        """Capture → (optional) track → (optional) detect → encode for web."""
        sc = self.streams.get(stream_id)
        state = self._states.get(stream_id)
        if not sc or not state:
            return

        # Open capture
        state.cap = await self._open_capture(sc)
        if state.cap is None:
            logger.error(f"[{stream_id}] Failed to open: {sc.input_source}")
            return

        state.source_fps = int(state.cap.get(cv2.CAP_PROP_FPS)) or 30
        state.width = int(state.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        state.height = int(state.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        state.active = True

        needs_yolo = bool(sc.yolo_classes) and self._model is not None
        needs_behavior = sc.task_fight or sc.task_escape or sc.task_inert or sc.task_sleep or sc.task_eat or sc.task_bathroom

        if needs_behavior:
            state.init_behavior_state()

        # Init DeepSORT if needed
        if needs_yolo and sc.method == "deepsort":
            from deep_sort import nn_matching
            from deep_sort.tracker import Tracker as DSTracker
            metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.5, None)
            state.tracker = DSTracker(metric)

        # Init output writer
        if sc.output_path:
            Path(sc.output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            state.writer = cv2.VideoWriter(
                sc.output_path, fourcc, sc.target_fps,
                (state.width, state.height),
            )

        frame_interval = 1.0 / sc.target_fps
        logger.info(f"[{stream_id}] Started — {state.width}x{state.height} @ {sc.target_fps}fps (source {state.source_fps}fps)")

        loop = asyncio.get_running_loop()

        try:
            while self.running and state.active:
                t0 = time.perf_counter()

                # Read frame in thread pool (blocking I/O)
                ret, frame = await loop.run_in_executor(self._executor, self._read_frame, state)
                if not ret or frame is None:
                    # For video files, stop; for RTSP, retry
                    if self._is_rtsp(sc.input_source):
                        logger.warning(f"[{stream_id}] Read failed, reconnecting...")
                        await asyncio.sleep(1)
                        state.cap = await self._open_capture(sc)
                        continue
                    else:
                        logger.info(f"[{stream_id}] End of video")
                        break

                state.frame_cnt += 1

                # --- YOLO tracking (optional) ---
                if needs_yolo:
                    frame, boxes, track_ids = await loop.run_in_executor(
                        self._executor, self._run_tracking, sc, state, frame
                    )
                    # --- Behavior detection (optional) ---
                    if needs_behavior and boxes and len(boxes) > 0:
                        frame = self._run_behavior_detection(sc, state, frame, boxes, track_ids)

                    # Update stats
                    stats = self._stats.get_stream_stats(stream_id)
                    if stats:
                        stats.tracked_objects = len(track_ids) if track_ids else 0

                # --- Encode JPEG for web ---
                jpeg = await loop.run_in_executor(self._executor, self._encode_jpeg, frame)
                with self.web_frames_lock:
                    self.web_frames[stream_id] = jpeg

                # --- Write output ---
                if state.writer:
                    state.writer.write(frame)

                # --- Update stats ---
                stats = self._stats.get_stream_stats(stream_id)
                if stats:
                    stats.frames_processed += 1
                    stats.update_fps()
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    stats.add_latency(elapsed_ms)

                # --- Frame rate control ---
                elapsed = time.perf_counter() - t0
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[{stream_id}] Error: {e}", exc_info=True)
        finally:
            state.active = False
            if state.cap and state.cap.isOpened():
                state.cap.release()
            if state.writer:
                state.writer.release()
                state.writer = None
            logger.info(f"[{stream_id}] Stopped")

    # ------------------------------------------------------------------
    # Helpers — run in thread pool
    # ------------------------------------------------------------------

    @staticmethod
    def _read_frame(state: StreamState):
        if state.cap is None or not state.cap.isOpened():
            return False, None
        return state.cap.read()

    @staticmethod
    def _encode_jpeg(frame: np.ndarray, quality: int = 70) -> bytes:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else b""

    @staticmethod
    def _is_rtsp(source: str) -> bool:
        return source.lower().startswith("rtsp://")

    async def _open_capture(self, sc: StreamConfig) -> Optional[cv2.VideoCapture]:
        loop = asyncio.get_running_loop()
        cap = await loop.run_in_executor(self._executor, self._do_open_capture, sc.input_source)
        return cap

    @staticmethod
    def _do_open_capture(source: str) -> Optional[cv2.VideoCapture]:
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)
        if cap.isOpened():
            return cap
        return None

    # ------------------------------------------------------------------
    # YOLO tracking
    # ------------------------------------------------------------------

    def _run_tracking(self, sc: StreamConfig, state: StreamState, frame: np.ndarray):
        """Run YOLO tracking on a single frame. Returns (frame, boxes, track_ids)."""
        from tracking import track_with_bytetrack, track_with_botsort, track_with_deepsort

        with self._model_lock:
            if sc.method == "bytetrack":
                boxes, track_ids, frame = track_with_bytetrack(self._model, frame)
            elif sc.method == "botsort":
                boxes, track_ids, frame = track_with_botsort(self._model, frame)
            elif sc.method == "deepsort":
                boxes, track_ids, frame = track_with_deepsort(self._model, state.tracker, frame)
            else:
                boxes, track_ids = [], []

        return frame, boxes, track_ids

    # ------------------------------------------------------------------
    # Behavior detection
    # ------------------------------------------------------------------

    def _run_behavior_detection(
        self, sc: StreamConfig, state: StreamState,
        frame: np.ndarray, boxes: list, track_ids: list
    ) -> np.ndarray:
        from detect_fight import detect_fight
        from detect_inert import detect_inert
        from detect_sleep import detect_sleep
        from detect_eat import detect_eat
        from detect_bathroom import detect_bathroom

        # Prepare coordinates
        for tid in track_ids:
            if tid not in state.inert_coor:
                state.inert_coor[tid] = deque([], maxlen=sc.inert_frames)
            if tid not in state.sleep_coor:
                state.sleep_coor[tid] = deque([], maxlen=sc.sleep_frames)
                state.sleep_bbox[tid] = deque([], maxlen=sc.sleep_frames)
            if tid not in state.eat_coor:
                state.eat_coor[tid] = deque([], maxlen=sc.eat_direction_frames * 3)
            if tid not in state.bathroom_coor:
                state.bathroom_coor[tid] = deque([], maxlen=sc.bathroom_trigger_frames * 2)
                state.bathroom_bbox[tid] = deque([], maxlen=sc.bathroom_trigger_frames * 2)

        id_to_idx = {tid: i for i, tid in enumerate(track_ids)}
        x_centers, y_centers = [], []
        last_w, last_h = 0, 0

        for tid, box in zip(track_ids, boxes):
            if hasattr(box, 'tolist'):
                xc, yc, w, h = box.tolist()
            else:
                xc, yc, w, h = box
            x_centers.append(xc)
            y_centers.append(yc)
            last_w, last_h = int(w), int(h)
            state.inert_coor[tid].append([xc, yc])
            state.sleep_coor[tid].append([xc, yc])
            state.sleep_bbox[tid].append([w, h])
            state.eat_coor[tid].append([xc, yc])
            state.bathroom_coor[tid].append([xc, yc])
            state.bathroom_bbox[tid].append([w, h])

        x_arr = np.array(x_centers)
        y_arr = np.array(y_centers)
        status_parts = []

        # Fight detection
        if sc.task_fight and state.close_count is not None:
            fight_indices = detect_fight(
                x_arr, y_arr, track_ids, state.close_count, state.far_count,
                sc.threshold, sc.reset_frames, sc.flag_frames,
                last_w, last_h,
            )
            for ids in fight_indices:
                for i in ids:
                    idx = id_to_idx.get(i.item())
                    if idx is not None:
                        bx = boxes[idx]
                        xc, yc, w, h = (bx.tolist() if hasattr(bx, 'tolist') else bx)
                        pt1 = (int(xc - w / 2), int(yc - h / 2))
                        pt2 = (int(xc + w / 2), int(yc + h / 2))
                        cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
            if fight_indices:
                status_parts.append("Fight")
                stats = self._stats.get_stream_stats(sc.stream_id)
                if stats:
                    stats.detections["fight"] += len(fight_indices)

        # Escape detection
        if sc.task_escape and sc.escape_polygon:
            from detect_escape import detect_escape
            w, h = frame.shape[1], frame.shape[0]
            frame, escaped_ids = detect_escape(
                boxes, track_ids, frame, state.frame_cnt, sc.escape_polygon, w, h
            )
            if escaped_ids:
                status_parts.append("Escape")
                stats = self._stats.get_stream_stats(sc.stream_id)
                if stats:
                    stats.detections["escape"] += len(escaped_ids)

        # Inert detection
        if sc.task_inert:
            inert_ids = detect_inert(state.inert_coor, sc.inert_threshold, sc.inert_frames)
            for tid in inert_ids:
                idx = id_to_idx.get(tid)
                if idx is not None:
                    bx = boxes[idx]
                    xc, yc, w, h = (bx.tolist() if hasattr(bx, 'tolist') else bx)
                    pt1 = (int(xc - w / 2), int(yc - h / 2))
                    pt2 = (int(xc + w / 2), int(yc + h / 2))
                    cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
            if inert_ids:
                status_parts.append("Inert")
                stats = self._stats.get_stream_stats(sc.stream_id)
                if stats:
                    stats.detections["inert"] += len(inert_ids)

        # Sleep detection
        if sc.task_sleep:
            sleep_ids = detect_sleep(
                state.sleep_coor, state.sleep_bbox,
                sc.sleep_threshold, sc.sleep_frames,
                sc.sleep_aspect_ratio, sc.sleep_area_stability,
            )
            for tid in sleep_ids:
                idx = id_to_idx.get(tid)
                if idx is not None:
                    bx = boxes[idx]
                    xc, yc, w, h = (bx.tolist() if hasattr(bx, 'tolist') else bx)
                    pt1 = (int(xc - w / 2), int(yc - h / 2))
                    pt2 = (int(xc + w / 2), int(yc + h / 2))
                    cv2.rectangle(frame, pt1, pt2, (128, 0, 128), 2)
            if sleep_ids:
                status_parts.append("Sleep")
                stats = self._stats.get_stream_stats(sc.stream_id)
                if stats:
                    stats.detections["sleep"] = stats.detections.get("sleep", 0) + len(sleep_ids)

        # Eat detection
        if sc.task_eat and self._model is not None:
            with self._model_lock:
                bowl_results = self._model.predict(
                    frame, conf=sc.bowl_conf, iou=0.5, classes=[3], verbose=False
                )
            bowl_boxes = []
            if len(bowl_results[0].boxes) > 0:
                bowl_boxes = bowl_results[0].boxes.xyxy.cpu().numpy()

            eat_ids = detect_eat(
                state.eat_coor, state.eat_near_count,
                boxes, track_ids, bowl_boxes,
                sc.eat_iou_threshold, sc.eat_dwell_frames, sc.eat_direction_frames,
            )
            for tid in eat_ids:
                idx = id_to_idx.get(tid)
                if idx is not None:
                    bx = boxes[idx]
                    xc, yc, w, h = (bx.tolist() if hasattr(bx, 'tolist') else bx)
                    pt1 = (int(xc - w / 2), int(yc - h / 2))
                    pt2 = (int(xc + w / 2), int(yc + h / 2))
                    cv2.rectangle(frame, pt1, pt2, (255, 105, 180), 2)
            if eat_ids:
                status_parts.append("Eating")
                stats = self._stats.get_stream_stats(sc.stream_id)
                if stats:
                    stats.detections["eat"] = stats.detections.get("eat", 0) + len(eat_ids)

        # Bathroom detection
        if sc.task_bathroom:
            bathroom_ids = detect_bathroom(
                state.bathroom_coor, state.bathroom_bbox,
                boxes, track_ids, frame,
                sc.bathroom_cls_model,
                sc.bathroom_trigger_frames, sc.bathroom_height_drop,
                cls_confidence=sc.bathroom_cls_conf,
            )
            for tid in bathroom_ids:
                idx = id_to_idx.get(tid)
                if idx is not None:
                    bx = boxes[idx]
                    xc, yc, w, h = (bx.tolist() if hasattr(bx, 'tolist') else bx)
                    pt1 = (int(xc - w / 2), int(yc - h / 2))
                    pt2 = (int(xc + w / 2), int(yc + h / 2))
                    cv2.rectangle(frame, pt1, pt2, (0, 191, 255), 2)
            if bathroom_ids:
                status_parts.append("Bathroom")
                stats = self._stats.get_stream_stats(sc.stream_id)
                if stats:
                    stats.detections["bathroom"] = stats.detections.get("bathroom", 0) + len(bathroom_ids)

        if status_parts:
            cv2.putText(
                frame, f"State: {' '.join(status_parts)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
            )

        return frame

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return stats in the format expected by web_server.py and main.py."""
        return self._stats.get_summary()

    async def _stats_loop(self):
        """Periodically print stats to console."""
        try:
            while self.running:
                await asyncio.sleep(self.config.stats_interval)
                if self.running:
                    self._stats.print_status()
        except asyncio.CancelledError:
            pass
