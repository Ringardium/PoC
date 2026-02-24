"""Multi-stream processor for real-time pet tracking with web streaming support."""

import asyncio
import logging
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Suppress FFmpeg HEVC decoder warnings (POC ref not found, PPS out of range)
# These are non-fatal warnings from B-frame decoding in HEVC RTSP streams.
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "error"
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from dotenv import load_dotenv

from config import SystemConfig, StreamConfig, GPUConfig, ProcessingConfig
from event_sender import BehaviorStateTracker, EventSender, BEHAVIOR_TYPE_MAP
from event_clip_uploader import EventClipRecorder
from hls_uploader import HLSUploader
from monitor import StatsAggregator

logger = logging.getLogger(__name__)


class SyncClock:
    """Cross-stream timestamp synchroniser using time-slot bucketing.

    All streams snap their frame timestamps to the nearest slot boundary
    so that events from different streams sharing the same real-world moment
    receive identical timestamps.
    """

    def __init__(self, slot_interval: float = None, target_fps: int = 30):
        self._slot_interval = slot_interval or (1.0 / target_fps)
        self._mono_origin = time.monotonic()
        self._unix_origin = time.time()

    def stamp(self) -> Tuple[float, float]:
        """Return (slot_monotonic, slot_unix) for the current moment."""
        now_mono = time.monotonic()
        elapsed = now_mono - self._mono_origin
        slot_index = round(elapsed / self._slot_interval)
        slot_mono = self._mono_origin + slot_index * self._slot_interval
        slot_unix = self._unix_origin + (slot_mono - self._mono_origin)
        return slot_mono, slot_unix


# Add parent directory for tracking/detection imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import apply_blur, apply_mosaic, apply_black_box


class StreamState:
    """Per-stream mutable state for tracking and behavior detection."""

    def __init__(self, config: StreamConfig):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_cnt: int = 0
        self.frame_timestamp: float = 0.0  # SyncClock-aligned unix timestamp
        self.source_fps: int = 30
        self.width: int = 0
        self.height: int = 0
        # Per-stream YOLO model instance (avoids tracker state contamination)
        self.model = None
        # Behavior detection state
        self.inert_coor: Dict = {}
        self.sleep_coor: Dict = {}
        self.sleep_bbox: Dict = {}
        self.eat_coor: Dict = {}
        self.eat_near_count: Dict = {}
        self.bathroom_coor: Dict = {}
        self.bathroom_bbox: Dict = {}
        self.active_coor: Dict = {}
        self.close_count: Optional[torch.Tensor] = None
        self.far_count: Optional[torch.Tensor] = None
        # Previous detection state (for send-once-on-start)
        self.prev_fight_ids: set = set()
        self.prev_escape_ids: set = set()
        self.prev_inert_ids: set = set()
        self.prev_eat_ids: set = set()
        self.prev_bathroom_ids: set = set()
        # DeepSORT tracker
        self.tracker = None
        # ReID tracker
        self.reid_tracker = None
        self.global_id_map: Dict = {}
        self.global_id_names: Dict[int, str] = {}  # global_id -> pet name
        # ID stabilization buffer: track_id -> {candidate_id, count}
        self.id_stable_buffer: Dict[int, Dict] = {}
        self.id_stable_threshold: int = 5  # N consecutive frames to confirm ID change
        # Output writer
        self.writer: Optional[cv2.VideoWriter] = None
        self.active: bool = False
        # Tracker reset flag: set True to use persist=False on next frame
        self.reset_tracker: bool = False
        self.no_detection_count: int = 0

    def init_behavior_state(self, max_number: int = 500):
        self.close_count = torch.zeros((max_number, max_number))
        self.far_count = torch.zeros((max_number, max_number))


class MultiStreamProcessor:
    """Process multiple video/RTSP streams with optional YOLO tracking and behavior detection."""

    def __init__(self, config: SystemConfig, web_enabled: bool = True):
        self.config = config
        self.web_enabled = web_enabled
        self.running = False
        self.streams: Dict[str, StreamConfig] = {}

        # Web frame buffer (only when web_enabled)
        if web_enabled:
            self.web_frames: Dict[str, bytes] = {}
            self.web_frames_lock = threading.Lock()
        else:
            self.web_frames = None
            self.web_frames_lock = None

        self._states: Dict[str, StreamState] = {}
        self._stats = StatsAggregator()
        self._executor = ThreadPoolExecutor(max_workers=max(len(config.streams) * 3, 8))
        # Limit concurrent GPU inference to prevent CUDA OOM (2 = overlap read+infer)
        self._gpu_sem = threading.Semaphore(2)
        self._stream_tasks: Dict[str, asyncio.Task] = {}
        self._stats_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Cross-stream timestamp synchronisation
        max_fps = max((sc.target_fps for sc in config.streams), default=30)
        self._sync_clock = SyncClock(target_fps=max_fps)

        # Behavior duration tracking (per-stream)
        self._sleep_trackers: Dict[str, BehaviorStateTracker] = {}
        self._active_trackers: Dict[str, BehaviorStateTracker] = {}
        self._event_sender: Optional[EventSender] = None
        if config.event_api_url:
            self._event_sender = EventSender(config.event_api_url)

        # HLS → S3 uploaders (per-stream, created after capture opens)
        load_dotenv()
        self._hls_uploaders: Dict[str, HLSUploader] = {}

        # Event clip recorders (per-stream, created after capture opens)
        self._clip_recorders: Dict[str, EventClipRecorder] = {}

        # Privacy filter YOLO model (shared across streams)
        self._privacy_model = None

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

    def _load_models(self):
        """Load per-stream YOLO model instances.

        Each stream gets its own YOLO instance so that ``model.track(persist=True)``
        keeps tracker state (Kalman filters, track IDs) isolated between streams.
        """
        if not self._needs_model():
            return

        from ultralytics import YOLO

        model_path = self.config.model_path

        # GPU optimizations (apply once)
        gpu = self.config.gpu
        if torch.cuda.is_available():
            if gpu.enable_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            if gpu.enable_cudnn_benchmark:
                torch.backends.cudnn.benchmark = True

        # Create one YOLO instance per stream that needs detection
        for sid, sc in self.streams.items():
            state = self._states[sid]
            if sc.yolo_classes and state.model is None:
                state.model = YOLO(model_path)

        # Load privacy filter model if any stream needs it (shared — no persist state)
        if self._privacy_model is None:
            for sc in self.streams.values():
                if sc.privacy:
                    self._privacy_model = YOLO(sc.privacy_model)
                    break

    # ------------------------------------------------------------------
    # Stream lifecycle
    # ------------------------------------------------------------------

    def _register_stream(self, sc: StreamConfig):
        self.streams[sc.stream_id] = sc
        self._states[sc.stream_id] = StreamState(sc)
        self._stats.register_stream(sc.stream_id)
        if sc.task_sleep:
            self._sleep_trackers[sc.stream_id] = BehaviorStateTracker("sleep", min_duration_min=1.0)
        if sc.task_active:
            self._active_trackers[sc.stream_id] = BehaviorStateTracker("active", min_duration_min=1.0)

    def _unregister_stream(self, stream_id: str):
        state = self._states.get(stream_id)
        if state:
            state.active = False
            if state.cap and state.cap.isOpened():
                state.cap.release()
            if state.writer:
                state.writer.release()
            state.model = None  # release VRAM
        self.streams.pop(stream_id, None)
        self._states.pop(stream_id, None)
        if self.web_enabled:
            with self.web_frames_lock:
                self.web_frames.pop(stream_id, None)

    def add_stream_dynamic(self, sc: StreamConfig) -> bool:
        """Add a stream at runtime. Returns False if max streams reached."""
        if len(self.streams) >= self.config.processing.max_streams:
            return False
        self._register_stream(sc)
        # Create per-stream model if needed
        state = self._states[sc.stream_id]
        if sc.yolo_classes and state.model is None:
            from ultralytics import YOLO
            state.model = YOLO(self.config.model_path)
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

        self._load_models()

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
            state.model = None  # release VRAM
        self._executor.shutdown(wait=False)
        self._stats.cleanup()
        if self._event_sender:
            self._event_sender.stop()
        for hls in self._hls_uploaders.values():
            hls.stop()
        self._hls_uploaders.clear()
        for clip_rec in self._clip_recorders.values():
            clip_rec.stop()
        self._clip_recorders.clear()

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

        # Auto-add bowl class (3) if eat detection is enabled
        if sc.task_eat and 3 not in sc.yolo_classes:
            sc.yolo_classes = list(sc.yolo_classes) + [3]

        needs_yolo = bool(sc.yolo_classes) and state.model is not None
        needs_behavior = sc.task_fight or sc.task_escape or sc.task_inert or sc.task_sleep or sc.task_eat or sc.task_bathroom or sc.task_active

        # Init ReID tracker
        if sc.use_reid:
            from reid import ReIDTracker
            state.reid_tracker = ReIDTracker(
                reid_method=sc.reid_method,
                similarity_threshold=sc.reid_threshold,
                correction_enabled=True,
                global_id_enabled=sc.reid_global_id,
            )
            # Load pet names from profiles
            try:
                from tools.pet_profiles import PetProfileStore
                store = PetProfileStore("references")
                state.global_id_names = store.get_name_map()
            except Exception:
                pass

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

        # Init HLS → S3 uploader
        if self.config.hls_s3_bucket:
            # Use RTMP stream key as S3 path (e.g. facility/ddnapet_gmail/every1)
            hls_stream_key = stream_id
            src = sc.input_source.lower()
            if src.startswith("rtmp://"):
                from urllib.parse import urlparse
                parsed = urlparse(sc.input_source)
                path = parsed.path.lstrip("/")  # "live/facility/ddnapet_gmail/every1"
                # Remove RTMP app name (first segment, e.g. "live")
                parts = path.split("/", 1)
                if len(parts) > 1:
                    hls_stream_key = parts[1]  # "facility/ddnapet_gmail/every1"
                elif path:
                    hls_stream_key = path

            self._hls_uploaders[stream_id] = HLSUploader(
                stream_id=hls_stream_key,
                fps=sc.target_fps,
                width=state.width,
                height=state.height,
                s3_bucket=self.config.hls_s3_bucket,
                s3_prefix=self.config.hls_s3_prefix,
            )

        # Init event clip recorder (requires S3 bucket)
        if self.config.hls_s3_bucket and needs_behavior:
            try:
                clip_stream_key = hls_stream_key
                self._clip_recorders[stream_id] = EventClipRecorder(
                    stream_id=clip_stream_key,
                    fps=sc.target_fps,
                    width=state.width,
                    height=state.height,
                    s3_bucket=self.config.hls_s3_bucket,
                    s3_prefix=self.config.clip_s3_prefix,
                    pre_seconds=self.config.clip_pre_seconds,
                    post_seconds=self.config.clip_post_seconds,
                )
            except Exception as e:
                logger.warning(f"[{stream_id}] EventClipRecorder init failed: {e}")

        frame_interval = 1.0 / sc.target_fps
        logger.info(f"[{stream_id}] Ready — {state.width}x{state.height} @ {sc.target_fps}fps")

        loop = asyncio.get_running_loop()
        reconnect_attempts = 0
        max_reconnect_backoff = 30  # max seconds between reconnection attempts
        consecutive_errors = 0
        max_consecutive_errors = 50  # restart capture after this many processing errors

        try:
            while self.running and state.active:
                t0 = time.perf_counter()

                # Read frame in thread pool (blocking I/O)
                ret, frame = await loop.run_in_executor(self._executor, self._read_frame, state)
                if not ret or frame is None:
                    # For live streams (RTSP/RTMP), reconnect with exponential backoff
                    if self._is_live_stream(sc.input_source):
                        reconnect_attempts += 1
                        backoff = min(2 ** min(reconnect_attempts, 5), max_reconnect_backoff)
                        logger.warning(f"[{stream_id}] Read failed, reconnecting (attempt {reconnect_attempts}, backoff {backoff}s)...")
                        # Release old capture to prevent resource leak
                        if state.cap is not None:
                            try:
                                state.cap.release()
                            except Exception:
                                pass
                            state.cap = None
                        await asyncio.sleep(backoff)
                        state.cap = await self._open_capture(sc)
                        if state.cap is not None:
                            state.reset_tracker = True  # reset ByteTrack state on next frame
                            logger.info(f"[{stream_id}] Reconnected successfully")
                        continue
                    else:
                        logger.info(f"[{stream_id}] End of video")
                        break

                # Reset reconnect counter on successful read
                reconnect_attempts = 0

                state.frame_cnt += 1
                _, state.frame_timestamp = self._sync_clock.stamp()

                # --- Processing block: wrapped so transient errors don't kill the stream ---
                try:
                    # --- Privacy filter (person detection → blur/mosaic/black) ---
                    if sc.privacy and self._privacy_model is not None:
                        frame = await loop.run_in_executor(
                            self._executor, self._apply_privacy, sc, frame
                        )

                    # --- YOLO tracking (optional) ---
                    if needs_yolo:
                        frame, boxes, track_ids, bowl_boxes = await loop.run_in_executor(
                            self._executor, self._run_tracking, sc, state, frame
                        )

                        # --- ReID correction (optional) ---
                        if state.reid_tracker is not None and len(boxes) > 0:
                            reid_result = state.reid_tracker.process(frame, boxes, track_ids)
                            raw_corrected = reid_result['corrected_ids']

                            # ID stabilization: only accept correction after N consistent frames
                            stable_ids = []
                            for orig_tid, corr_tid in zip(track_ids, raw_corrected):
                                if orig_tid == corr_tid:
                                    # No correction — clear buffer and keep original
                                    state.id_stable_buffer.pop(orig_tid, None)
                                    stable_ids.append(orig_tid)
                                else:
                                    buf = state.id_stable_buffer.get(orig_tid)
                                    if buf and buf['candidate'] == corr_tid:
                                        buf['count'] += 1
                                    else:
                                        state.id_stable_buffer[orig_tid] = {'candidate': corr_tid, 'count': 1}
                                        buf = state.id_stable_buffer[orig_tid]

                                    if buf['count'] >= state.id_stable_threshold:
                                        # Confirmed — accept correction
                                        stable_ids.append(corr_tid)
                                        state.id_stable_buffer.pop(orig_tid, None)
                                    else:
                                        # Not yet confirmed — keep original
                                        stable_ids.append(orig_tid)

                            track_ids = stable_ids
                            if reid_result.get('global_ids'):
                                for tid, gid in zip(track_ids, reid_result['global_ids']):
                                    state.global_id_map[tid] = gid

                        # --- Behavior detection (optional) ---
                        if needs_behavior and len(boxes) > 0:
                            frame = self._run_behavior_detection(sc, state, frame, boxes, track_ids, bowl_boxes)

                        # Update stats
                        stats = self._stats.get_stream_stats(stream_id)
                        if stats:
                            stats.tracked_objects = len(track_ids) if track_ids else 0

                    consecutive_errors = 0  # reset on success

                except Exception as proc_err:
                    consecutive_errors += 1
                    if consecutive_errors <= 3 or consecutive_errors % 50 == 0:
                        logger.warning(f"[{stream_id}] Processing error ({consecutive_errors}): {proc_err}", exc_info=(consecutive_errors == 1))
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"[{stream_id}] Too many consecutive errors ({consecutive_errors}), restarting capture...")
                        consecutive_errors = 0
                        if state.cap is not None:
                            try:
                                state.cap.release()
                            except Exception:
                                pass
                            state.cap = None
                        await asyncio.sleep(2)
                        state.cap = await self._open_capture(sc)
                        continue

                # --- Encode JPEG for web ---
                if self.web_enabled:
                    jpeg = await loop.run_in_executor(self._executor, self._encode_jpeg, frame)
                    with self.web_frames_lock:
                        self.web_frames[stream_id] = jpeg

                # --- Write output ---
                if state.writer:
                    state.writer.write(frame)

                # --- HLS → S3 ---
                hls = self._hls_uploaders.get(stream_id)
                if hls:
                    hls.write_frame(frame)

                # --- Event clip buffer ---
                clip_rec = self._clip_recorders.get(stream_id)
                if clip_rec:
                    clip_rec.push_frame(frame)

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
            logger.error(f"[{stream_id}] Fatal error: {e}", exc_info=True)
        finally:
            state.active = False
            if state.cap and state.cap.isOpened():
                state.cap.release()
            if state.writer:
                state.writer.release()
                state.writer = None
            hls = self._hls_uploaders.pop(stream_id, None)
            if hls:
                hls.stop()
            clip_rec = self._clip_recorders.pop(stream_id, None)
            if clip_rec:
                clip_rec.stop()
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
    def _is_live_stream(source: str) -> bool:
        """Check if the source is a live stream (RTSP or RTMP) that should auto-reconnect."""
        s = source.lower()
        return s.startswith("rtsp://") or s.startswith("rtmp://")

    @staticmethod
    def _is_rtsp(source: str) -> bool:
        return source.lower().startswith("rtsp://")

    async def _open_capture(self, sc: StreamConfig) -> Optional[cv2.VideoCapture]:
        loop = asyncio.get_running_loop()
        cap = await loop.run_in_executor(self._executor, self._do_open_capture, sc.input_source)
        return cap

    @staticmethod
    def _do_open_capture(source: str) -> Optional[cv2.VideoCapture]:
        t0 = time.perf_counter()
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
        elif source.lower().startswith("rtsp://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|analyzeduration;5000000|probesize;5000000"
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            # Flush initial frames to skip past incomplete GOP
            if cap.isOpened():
                for _ in range(30):
                    cap.grab()
        elif source.lower().startswith("rtmp://"):
            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay|analyzeduration;500000|probesize;500000"
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(source)
        elapsed = time.perf_counter() - t0
        if cap.isOpened():
            return cap
        logger.warning(f"[capture] Failed after {elapsed:.1f}s: {source}")
        return None

    def _apply_privacy(self, sc: StreamConfig, frame: np.ndarray) -> np.ndarray:
        """Detect persons (class 0) and apply privacy filter."""
        results = self._privacy_model(frame, classes=[0], verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                pad = 10
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2 = min(frame.shape[1], x2 + pad)
                y2 = min(frame.shape[0], y2 + pad)
                if sc.privacy_method == "mosaic":
                    frame = apply_mosaic(frame, x1, y1, x2, y2)
                elif sc.privacy_method == "black":
                    frame = apply_black_box(frame, x1, y1, x2, y2)
                else:
                    frame = apply_blur(frame, x1, y1, x2, y2)
        return frame

    # ------------------------------------------------------------------
    # YOLO tracking
    # ------------------------------------------------------------------

    def _run_tracking(self, sc: StreamConfig, state: StreamState, frame: np.ndarray):
        """Run YOLO tracking on a single frame. Returns (frame, boxes, track_ids, bowl_boxes).

        Each stream uses its own ``state.model`` instance so tracker state
        (Kalman filters, track IDs) is fully isolated between streams.
        Concurrent GPU inference is limited via ``_gpu_sem`` to prevent CUDA OOM.
        """
        device = f"cuda:{self.config.gpu.device_id}" if torch.cuda.is_available() else "cpu"
        half = self.config.gpu.half_precision and torch.cuda.is_available()

        if sc.method in ("bytetrack", "botsort"):
            tracker_yaml = "bytetrack.yaml" if sc.method == "bytetrack" else "botsort.yaml"
            # Use persist=False to reset tracker after reconnection or prolonged no-detection
            use_persist = sc.yolo_persist and not state.reset_tracker
            with self._gpu_sem:
                results = state.model.track(
                    frame, persist=use_persist, conf=sc.yolo_conf, iou=sc.yolo_iou,
                    classes=sc.yolo_classes or [1], tracker=tracker_yaml,
                    device=device, half=half, verbose=False,
                )
            if state.reset_tracker:
                state.reset_tracker = False
            ids = results[0].boxes.id
            cls = results[0].boxes.cls.cpu() if results[0].boxes.cls is not None else None
            if ids is None:
                state.no_detection_count += 1
                # Auto-reset tracker if no detections for 90+ consecutive frames
                if state.no_detection_count >= 90:
                    state.reset_tracker = True
                    state.no_detection_count = 0
                boxes, track_ids, bowl_boxes = [], [], []
            else:
                state.no_detection_count = 0
                all_xywh = results[0].boxes.xywh.cpu()
                all_xyxy = results[0].boxes.xyxy.cpu().numpy()
                all_ids = results[0].boxes.id.int().cpu().tolist()
                # Separate pets (class 1) and bowls (class 3)
                all_conf = results[0].boxes.conf.cpu()
                pet_entries = []  # (conf, xywh, id)
                bowl_boxes = []
                for i, c in enumerate(cls):
                    if int(c) == 1:
                        pet_entries.append((float(all_conf[i]), all_xywh[i], all_ids[i]))
                    elif int(c) == 3:
                        bowl_boxes.append(all_xyxy[i])

                # Deduplicate: keep highest confidence per track ID
                seen_ids = {}
                for conf, xywh, tid in pet_entries:
                    if tid not in seen_ids or conf > seen_ids[tid][0]:
                        seen_ids[tid] = (conf, xywh)
                boxes = [v[1] for v in seen_ids.values()]
                track_ids = list(seen_ids.keys())

                if len(boxes) > 0:
                    boxes = torch.stack(boxes)
                if len(bowl_boxes) > 0:
                    bowl_boxes = np.array(bowl_boxes)
                # Don't use results[0].plot() — we draw our own bbox + labels
        elif sc.method == "deepsort":
            from tracking import track_with_deepsort  # noqa: root-level module
            boxes, track_ids, frame = track_with_deepsort(state.model, state.tracker, frame)
            bowl_boxes = []
        else:
            boxes, track_ids, bowl_boxes = [], [], []

        return frame, boxes, track_ids, bowl_boxes

    # ------------------------------------------------------------------
    # Behavior detection
    # ------------------------------------------------------------------

    def _run_behavior_detection(
        self, sc: StreamConfig, state: StreamState,
        frame: np.ndarray, boxes: list, track_ids: list,
        bowl_boxes: list = None
    ) -> np.ndarray:
        from detection import detect_fight, detect_inert, detect_sleep, detect_eat, detect_bathroom, detect_active

        # Clip recorder for this stream
        clip_rec = self._clip_recorders.get(sc.stream_id)

        def _send_event(event: dict):
            """Send event via API and trigger clip recording. Skips if dogId is None."""
            if event.get("dogId") is None:
                return
            event["detectedAt"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            if clip_rec:
                s3_key = clip_rec.trigger(event)
                if s3_key and self.config.cdn_base_url:
                    event["clipUrl"] = f"{self.config.cdn_base_url.rstrip('/')}/{s3_key}"
            if self._event_sender:
                self._event_sender.send(event)

        # Map track_ids → behavior_ids (use global_id when available)
        behavior_ids = []
        tid_to_bid = {}
        for tid in track_ids:
            gid = state.global_id_map.get(tid)
            bid = gid if gid is not None else tid
            behavior_ids.append(bid)
            tid_to_bid[tid] = bid

        def _resolve_dog_id(bid):
            """Return global_id for API events, or None if not assigned."""
            if bid in state.global_id_map.values():
                return bid
            return None

        # Prepare coordinates (keyed by behavior_id)
        for bid in behavior_ids:
            if bid not in state.inert_coor:
                state.inert_coor[bid] = deque([], maxlen=sc.inert_frames)
            if bid not in state.sleep_coor:
                state.sleep_coor[bid] = deque([], maxlen=sc.sleep_frames)
                state.sleep_bbox[bid] = deque([], maxlen=sc.sleep_frames)
            if bid not in state.eat_coor:
                state.eat_coor[bid] = deque([], maxlen=sc.eat_direction_frames * 3)
            if bid not in state.bathroom_coor:
                state.bathroom_coor[bid] = deque([], maxlen=sc.bathroom_trigger_frames * 2)
                state.bathroom_bbox[bid] = deque([], maxlen=sc.bathroom_trigger_frames * 2)
            if bid not in state.active_coor:
                state.active_coor[bid] = deque([], maxlen=sc.active_frames)

        x_centers, y_centers = [], []
        last_w, last_h = 0, 0

        for tid, bid, box in zip(track_ids, behavior_ids, boxes):
            if hasattr(box, 'tolist'):
                xc, yc, w, h = box.tolist()
            else:
                xc, yc, w, h = box
            x_centers.append(xc)
            y_centers.append(yc)
            last_w, last_h = int(w), int(h)
            state.inert_coor[bid].append([xc, yc])
            state.sleep_coor[bid].append([xc, yc])
            state.sleep_bbox[bid].append([w, h])
            state.eat_coor[bid].append([xc, yc])
            state.bathroom_coor[bid].append([xc, yc])
            state.bathroom_bbox[bid].append([w, h])
            state.active_coor[bid].append([xc, yc])

        x_arr = np.array(x_centers)
        y_arr = np.array(y_centers)

        # Per-dog behavior label (highest priority wins for color)
        # Priority: fight > escape > sleeping > bathroom > feeding > playing > inactive > none
        dog_behavior: Dict[int, str] = {}  # tid -> behavior name

        # Color map (BGR)
        COLOR_MAP = {
            "fight":    (0, 0, 255),       # red
            "escape":   (0, 255, 255),     # yellow
            "sleeping": (128, 0, 128),     # purple
            "bathroom": (235, 206, 135),   # sky blue
            "feeding":  (180, 105, 255),   # pink
            "playing":  (0, 165, 255),     # orange
            "inactive": (180, 0, 0),       # dark blue
            "normal":   (0, 200, 0),       # green
            "none":     (144, 238, 144),   # light green
        }

        # Compute per-dog speeds for fight detection (use track_ids for fight matrix)
        speeds = {}
        for tid in track_ids:
            bid = tid_to_bid[tid]
            if bid in state.inert_coor and len(state.inert_coor[bid]) >= 2:
                coor = np.array(state.inert_coor[bid])
                recent = coor[-min(10, len(coor)):]
                dists = np.sqrt(np.sum((recent[1:] - recent[:-1]) ** 2, axis=1))
                speeds[tid] = float(np.mean(dists))
            else:
                speeds[tid] = 0.0

        # --- Detect all behaviors (no drawing yet) ---

        # Fight detection
        fight_set = set()
        if sc.task_fight and state.close_count is not None:
            fight_indices = detect_fight(
                x_arr, y_arr, track_ids, state.close_count, state.far_count,
                sc.threshold, sc.reset_frames, sc.flag_frames,
                last_w, last_h,
                speeds=speeds,
                fight_speed_threshold=sc.fight_speed_threshold,
            )
            for ids in fight_indices:
                for i in ids:
                    fight_set.add(i.item())
            if len(fight_set) > 0:
                for tid in (fight_set - state.prev_fight_ids):
                    bid = tid_to_bid.get(tid, tid)
                    _send_event({"dogId": _resolve_dog_id(bid), "behaviorType": BEHAVIOR_TYPE_MAP["fight"]})
                stats = self._stats.get_stream_stats(sc.stream_id)
                if stats:
                    stats.detections["fight"] += len(fight_indices)
            state.prev_fight_ids = fight_set
            for tid in fight_set:
                bid = tid_to_bid.get(tid, tid)
                dog_behavior[bid] = "fight"

        # Escape detection (uses track_ids for bbox position check)
        escape_set = set()
        if sc.task_escape and sc.escape_polygon:
            from detection import detect_escape
            w, h = frame.shape[1], frame.shape[0]
            frame, escaped_ids = detect_escape(
                boxes, track_ids, frame, state.frame_cnt, sc.escape_polygon, w, h
            )
            # Convert escaped track_ids to behavior_ids
            escaped_bids = set(tid_to_bid.get(tid, tid) for tid in escaped_ids)
            escape_set = escaped_bids
            if len(escape_set) > 0:
                for bid in (escape_set - state.prev_escape_ids):
                    _send_event({"dogId": _resolve_dog_id(bid), "behaviorType": BEHAVIOR_TYPE_MAP["escape"]})
                stats = self._stats.get_stream_stats(sc.stream_id)
                if stats:
                    stats.detections["escape"] += len(escaped_ids)
            state.prev_escape_ids = escape_set
            for bid in escape_set:
                if bid not in dog_behavior:
                    dog_behavior[bid] = "escape"

        # Sleep detection (state dicts already keyed by behavior_id)
        sleep_set = set()
        if sc.task_sleep:
            sleep_ids = detect_sleep(
                state.sleep_coor, state.sleep_bbox,
                sc.sleep_threshold, sc.sleep_frames,
                sc.sleep_aspect_ratio, sc.sleep_area_stability,
            )
            sleep_set = set(sleep_ids)
            if len(sleep_set) > 0:
                stats = self._stats.get_stream_stats(sc.stream_id)
                if stats:
                    stats.detections["sleep"] = stats.detections.get("sleep", 0) + len(sleep_ids)
            sleep_tracker = self._sleep_trackers.get(sc.stream_id)
            if sleep_tracker:
                ended = sleep_tracker.update(sleep_ids, state.frame_cnt, sc.target_fps,
                                            current_time=state.frame_timestamp)
                for ev in ended:
                    ev["dogId"] = _resolve_dog_id(ev["dogId"])
                    _send_event(ev)
            for bid in sleep_set:
                if bid not in dog_behavior:
                    dog_behavior[bid] = "sleeping"

        # Bathroom detection
        bathroom_set = set()
        if sc.task_bathroom:
            bathroom_ids = detect_bathroom(
                state.bathroom_coor, state.bathroom_bbox,
                boxes, behavior_ids, frame,
                sc.bathroom_cls_model,
                sc.bathroom_trigger_frames, sc.bathroom_height_drop,
                cls_confidence=sc.bathroom_cls_conf,
            )
            bathroom_set = set(bathroom_ids)
            if len(bathroom_set) > 0:
                for bid in (bathroom_set - state.prev_bathroom_ids):
                    _send_event({"dogId": _resolve_dog_id(bid), "behaviorType": BEHAVIOR_TYPE_MAP["bathroom"]})
                stats = self._stats.get_stream_stats(sc.stream_id)
                if stats:
                    stats.detections["bathroom"] = stats.detections.get("bathroom", 0) + len(bathroom_ids)
            state.prev_bathroom_ids = bathroom_set
            for bid in bathroom_set:
                if bid not in dog_behavior:
                    dog_behavior[bid] = "bathroom"

        # Eat detection
        eat_set = set()
        if sc.task_eat:
            eat_ids = detect_eat(
                state.eat_coor, state.eat_near_count,
                boxes, behavior_ids,
                bowl_boxes if (bowl_boxes is not None and len(bowl_boxes) > 0) else [],
                sc.eat_iou_threshold, sc.eat_dwell_frames, sc.eat_direction_frames,
            )
            eat_set = set(eat_ids)
            if len(eat_set) > 0:
                for bid in (eat_set - state.prev_eat_ids):
                    _send_event({"dogId": _resolve_dog_id(bid), "behaviorType": BEHAVIOR_TYPE_MAP["eat"]})
                stats = self._stats.get_stream_stats(sc.stream_id)
                if stats:
                    stats.detections["eat"] = stats.detections.get("eat", 0) + len(eat_ids)
            state.prev_eat_ids = eat_set
            for bid in eat_set:
                if bid not in dog_behavior:
                    dog_behavior[bid] = "feeding"

        # Active detection
        active_set = set()
        if sc.task_active:
            active_ids = detect_active(
                state.active_coor, sc.active_threshold, sc.active_frames,
                speed_threshold=sc.active_speed_threshold,
            )
            active_set = set(active_ids)
            if len(active_set) > 0:
                stats = self._stats.get_stream_stats(sc.stream_id)
                if stats:
                    stats.detections["active"] = stats.detections.get("active", 0) + len(active_ids)
            active_tracker = self._active_trackers.get(sc.stream_id)
            if active_tracker:
                ended = active_tracker.update(active_ids, state.frame_cnt, sc.target_fps,
                                            current_time=state.frame_timestamp)
                for ev in ended:
                    ev["dogId"] = _resolve_dog_id(ev["dogId"])
                    _send_event(ev)
            for bid in active_set:
                if bid not in dog_behavior:
                    dog_behavior[bid] = "playing"

        # Inert detection
        inert_set = set()
        if sc.task_inert:
            inert_ids = detect_inert(state.inert_coor, sc.inert_threshold, sc.inert_frames)
            inert_set = set(inert_ids)
            if len(inert_set) > 0:
                for bid in (inert_set - state.prev_inert_ids):
                    _send_event({"dogId": _resolve_dog_id(bid), "behaviorType": BEHAVIOR_TYPE_MAP["inert"]})
                stats = self._stats.get_stream_stats(sc.stream_id)
                if stats:
                    stats.detections["inert"] += len(inert_ids)
            state.prev_inert_ids = inert_set
            for bid in inert_set:
                if bid not in dog_behavior:
                    dog_behavior[bid] = "inactive"

        # --- Draw bbox + label per dog ---
        registered_only = sc.label_registered_only
        for tid, box in zip(track_ids, boxes):
            bid = tid_to_bid.get(tid, tid)
            behavior = dog_behavior.get(bid, "normal")

            if hasattr(box, 'tolist'):
                xc, yc, w, h = box.tolist()
            else:
                xc, yc, w, h = box
            pt1 = (int(xc - w / 2), int(yc - h / 2))
            pt2 = (int(xc + w / 2), int(yc + h / 2))

            color = COLOR_MAP.get(behavior, COLOR_MAP["none"])
            gid = state.global_id_map.get(tid)
            pet_name = state.global_id_names.get(gid) if gid else None

            # Determine label text
            if pet_name:
                label = f"{pet_name} {behavior}"
            elif registered_only:
                # Unregistered pet — bbox only, no label
                cv2.rectangle(frame, pt1, pt2, COLOR_MAP["none"], 1)
                continue
            elif gid is not None:
                label = f"G:{gid} {behavior}"
            else:
                label = f"#{tid} {behavior}"

            cv2.rectangle(frame, pt1, pt2, color, 2)
            if behavior != "normal":
                # Label background
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (pt1[0], pt1[1] - th - 10), (pt1[0] + tw, pt1[1]), color, -1)
                cv2.putText(frame, label, (pt1[0], pt1[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

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
