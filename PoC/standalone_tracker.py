"""Standalone ByteTrack / BoT-SORT wrapper for batched-detection pipeline (Phase 4).

Ultralytics' ``model.track(persist=True)`` keeps tracker state inside
``model.predictor`` — sharing one model instance across streams corrupts
that state. To allow detection batching across streams while keeping
per-stream tracker isolation, we instantiate the underlying ByteTrack /
BoT-SORT trackers ourselves and feed them detection results.

Each :class:`StandaloneTracker` holds the state for a single stream.
Detection happens elsewhere (a shared YOLO model called in batch); the
tracker only sees per-frame raw boxes/scores/classes and emits track IDs.

Usage:
    tracker = StandaloneTracker(method="bytetrack", target_fps=15)
    # On every YOLO frame:
    tracks = tracker.update(boxes_xyxy, scores, classes, frame=frame)
    # tracks: ndarray (N, 7) [x1, y1, x2, y2, track_id, score, cls]
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import yaml

# Ultralytics ships standalone trackers we can reuse.
from ultralytics.trackers import BOTSORT, BYTETracker  # type: ignore


# bytetrack.yaml / botsort.yaml live at repo root (one level up from PoC/).
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent


def _load_tracker_args(method: str) -> SimpleNamespace:
    """Load default ByteTrack / BoT-SORT yaml as an attribute namespace."""
    yaml_name = "bytetrack.yaml" if method == "bytetrack" else "botsort.yaml"
    yaml_path = _DEFAULT_CONFIG_DIR / yaml_name
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Trackers expect attribute access (cfg.track_high_thresh, ...).
    return SimpleNamespace(**cfg)


class _DetWrapper:
    """Mimics the subset of ``ultralytics.engine.results.Boxes`` that
    BYTETracker.update consumes.

    The tracker reads ``conf``, ``xywh``, ``xyxy``, ``cls``; supports
    ``len()`` and ``__getitem__`` for boolean-mask slicing.
    """

    __slots__ = ("xyxy", "xywh", "conf", "cls")

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = xyxy.astype(np.float32, copy=False)
        self.conf = conf.astype(np.float32, copy=False)
        self.cls = cls.astype(np.float32, copy=False)
        # xywh = center-xy + wh
        if len(xyxy):
            cx = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2.0
            cy = (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2.0
            w = self.xyxy[:, 2] - self.xyxy[:, 0]
            h = self.xyxy[:, 3] - self.xyxy[:, 1]
            self.xywh = np.stack([cx, cy, w, h], axis=1).astype(np.float32)
        else:
            self.xywh = np.zeros((0, 4), dtype=np.float32)

    def __len__(self) -> int:
        return self.xyxy.shape[0]

    def __getitem__(self, idx) -> "_DetWrapper":
        return _DetWrapper(self.xyxy[idx], self.conf[idx], self.cls[idx])


class StandaloneTracker:
    """Per-stream tracker fed by an external (batched) detector.

    Holds the ByteTrack / BoT-SORT state for one video stream. Replaces
    the in-model tracker that ``model.track(persist=True)`` would normally
    own, so we can share a single YOLO instance across many streams.
    """

    def __init__(
        self,
        method: str = "bytetrack",
        target_fps: int = 30,
        custom_args: Optional[dict] = None,
    ):
        if method not in ("bytetrack", "botsort"):
            raise ValueError(f"Unsupported tracker method: {method!r}")
        self.method = method
        args = _load_tracker_args(method)
        if custom_args:
            for k, v in custom_args.items():
                setattr(args, k, v)
        cls_ = BYTETracker if method == "bytetrack" else BOTSORT
        self._tracker = cls_(args, frame_rate=int(target_fps))

    def reset(self):
        """Drop all tracks; call after long disconnect or scene change."""
        self._tracker.reset()

    def update(
        self,
        xyxy: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
        frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Feed one frame's detections; return tracked objects.

        Args:
            xyxy: (N, 4) corner-xyxy boxes in source pixels.
            scores: (N,) confidences.
            classes: (N,) class ids (int).
            frame: optional original BGR frame (BoT-SORT GMC needs it).

        Returns:
            ndarray (M, 7): ``[x1, y1, x2, y2, track_id, score, cls]``
            for each surviving track. Empty (0, 7) when no tracks.
        """
        if xyxy is None or len(xyxy) == 0:
            det = _DetWrapper(
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        else:
            det = _DetWrapper(np.asarray(xyxy), np.asarray(scores), np.asarray(classes))

        # BYTETracker.update returns (M, 8)?: the upstream signature returns
        # tracks where the trailing column is the index back into the input
        # detections. We strip that and keep [x1,y1,x2,y2, tid, score, cls].
        tracks = self._tracker.update(det, img=frame)
        if tracks is None or len(tracks) == 0:
            return np.zeros((0, 7), dtype=np.float32)
        return tracks[:, :7].astype(np.float32, copy=False)
