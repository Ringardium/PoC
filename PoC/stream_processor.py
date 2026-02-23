"""Multi-stream processor for real-time pet tracking with web streaming support."""

import asyncio
import logging
import os
import sys
import threading
import time
from collections import deque
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
from hls_uploader import HLSUploader
from monitor import StatsAggregator

logger = logging.getLogger(__name__)

# Add parent directory for tracking/detection imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import apply_blur, apply_mosaic, apply_black_box


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

        # Behavior duration tracking (per-stream)
        self._sleep_trackers: Dict[str, BehaviorStateTracker] = {}
        self._active_trackers: Dict[str, BehaviorStateTracker] = {}
        self._event_sender: Optional[EventSender] = None
        if config.event_api_url:
            self._event_sender = EventSender(config.event_api_url)

        # HLS → S3 uploaders (per-stream, created after capture opens)
        load_dotenv()
        self._hls_uploaders: Dict[str, HLSUploader] = {}

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

        # Load privacy filter model if any stream needs it
        if self._privacy_model is None:
            for sc in self.streams.values():
                if sc.privacy:
                    from ultralytics import YOLO as _YOLO
                    self._privacy_model = _YOLO(sc.privacy_model)
                    logger.info(f"Privacy model loaded: {sc.privacy_model}")
                    break

        # GPU optimizations
        gpu = self.config.gpu
        if torch.cuda.is_available():
            if gpu.enable_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            if gpu.enable_cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
            logger.info(f"GPU ready: cuda:{gpu.device_id} (half={gpu.half_precision})")

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
        if self._event_sender:
            self._event_sender.stop()
        for hls in self._hls_uploaders.values():
            hls.stop()
        self._hls_uploaders.clear()

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
        logger.info(f"[{stream_id}] Capture info: {state.width}x{state.height} @ {state.source_fps}fps")

        # Auto-add bowl class (3) if eat detection is enabled
        if sc.task_eat and 3 not in sc.yolo_classes:
            sc.yolo_classes = list(sc.yolo_classes) + [3]
            logger.info(f"[{stream_id}] Auto-added class 3 (bowl) for eat detection")

        needs_yolo = bool(sc.yolo_classes) and self._model is not None
        needs_behavior = sc.task_fight or sc.task_escape or sc.task_inert or sc.task_sleep or sc.task_eat or sc.task_bathroom or sc.task_active
        logger.info(f"[{stream_id}] needs_yolo={needs_yolo}, needs_behavior={needs_behavior}")

        logger.info(f"[{stream_id}] Checking ReID... use_reid={sc.use_reid}, reid_global_id={sc.reid_global_id}")
        # Init ReID tracker
        if sc.use_reid or sc.reid_global_id:
            from reid import ReIDTracker
            state.reid_tracker = ReIDTracker(
                reid_method=sc.reid_method,
                similarity_threshold=sc.reid_threshold,
                correction_enabled=True,
                global_id_enabled=sc.reid_global_id,
            )
            mode = "full pipeline" if sc.reid_global_id else "ID correction"
            logger.info(f"[{stream_id}] ReID enabled — {mode}, method={sc.reid_method}")
            # Load pet names from profiles
            try:
                from tools.pet_profiles import PetProfileStore
                store = PetProfileStore("references")
                state.global_id_names = store.get_name_map()
                if state.global_id_names:
                    logger.info(f"[{stream_id}] Pet names loaded: {state.global_id_names}")
            except Exception:
                pass

        if needs_behavior:
            state.init_behavior_state()
        logger.info(f"[{stream_id}] Behavior init done")

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

        logger.info(f"[{stream_id}] DeepSORT init done, checking HLS...")

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

                # --- Encode JPEG for web ---
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
            hls = self._hls_uploaders.pop(stream_id, None)
            if hls:
                hls.stop()
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
        logger.info(f"Opening capture: {source}")
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
            logger.info(f"Capture opened in {elapsed:.1f}s: {source}")
            return cap
        logger.warning(f"Capture failed after {elapsed:.1f}s: {source}")
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
        """Run YOLO tracking on a single frame. Returns (frame, boxes, track_ids)."""
        device = f"cuda:{self.config.gpu.device_id}" if torch.cuda.is_available() else "cpu"
        half = self.config.gpu.half_precision and torch.cuda.is_available()

        with self._model_lock:
            if sc.method in ("bytetrack", "botsort"):
                tracker_yaml = "bytetrack.yaml" if sc.method == "bytetrack" else "botsort.yaml"
                results = self._model.track(
                    frame, persist=sc.yolo_persist, conf=sc.yolo_conf, iou=sc.yolo_iou,
                    classes=sc.yolo_classes or [1], tracker=tracker_yaml,
                    device=device, half=half, verbose=sc.yolo_verbose,
                )
                ids = results[0].boxes.id
                cls = results[0].boxes.cls.cpu() if results[0].boxes.cls is not None else None
                if ids is None:
                    boxes, track_ids, bowl_boxes = [], [], []
                else:
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
                boxes, track_ids, frame = track_with_deepsort(self._model, state.tracker, frame)
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

        # Map track_ids → behavior_ids (use global_id when available)
        behavior_ids = []
        tid_to_bid = {}
        for tid in track_ids:
            gid = state.global_id_map.get(tid)
            bid = gid if gid is not None else tid
            behavior_ids.append(bid)
            tid_to_bid[tid] = bid

        # Resolve display name / event ID: pet_name > global_id > track_id
        def _resolve_dog_id(bid):
            """Return the best identifier for API events."""
            name = state.global_id_names.get(bid)
            if name:
                return name
            return bid

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
                if self._event_sender:
                    for tid in (fight_set - state.prev_fight_ids):
                        bid = tid_to_bid.get(tid, tid)
                        self._event_sender.send({"dogId": _resolve_dog_id(bid), "behaviorType": BEHAVIOR_TYPE_MAP["fight"]})
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
                if self._event_sender:
                    for bid in (escape_set - state.prev_escape_ids):
                        self._event_sender.send({"dogId": _resolve_dog_id(bid), "behaviorType": BEHAVIOR_TYPE_MAP["escape"]})
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
                ended = sleep_tracker.update(sleep_ids, state.frame_cnt, sc.target_fps)
                if len(ended) > 0 and self._event_sender:
                    for ev in ended:
                        ev["dogId"] = _resolve_dog_id(ev["dogId"])
                        self._event_sender.send(ev)
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
                if self._event_sender:
                    for bid in (bathroom_set - state.prev_bathroom_ids):
                        self._event_sender.send({"dogId": _resolve_dog_id(bid), "behaviorType": BEHAVIOR_TYPE_MAP["bathroom"]})
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
                boxes, behavior_ids, bowl_boxes or [],
                sc.eat_iou_threshold, sc.eat_dwell_frames, sc.eat_direction_frames,
            )
            eat_set = set(eat_ids)
            if len(eat_set) > 0:
                if self._event_sender:
                    for bid in (eat_set - state.prev_eat_ids):
                        self._event_sender.send({"dogId": _resolve_dog_id(bid), "behaviorType": BEHAVIOR_TYPE_MAP["eat"]})
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
                ended = active_tracker.update(active_ids, state.frame_cnt, sc.target_fps)
                if len(ended) > 0 and self._event_sender:
                    for ev in ended:
                        ev["dogId"] = _resolve_dog_id(ev["dogId"])
                        self._event_sender.send(ev)
            for bid in active_set:
                if bid not in dog_behavior:
                    dog_behavior[bid] = "playing"

        # Inert detection
        inert_set = set()
        if sc.task_inert:
            inert_ids = detect_inert(state.inert_coor, sc.inert_threshold, sc.inert_frames)
            inert_set = set(inert_ids)
            if len(inert_set) > 0:
                if self._event_sender:
                    for bid in (inert_set - state.prev_inert_ids):
                        self._event_sender.send({"dogId": _resolve_dog_id(bid), "behaviorType": BEHAVIOR_TYPE_MAP["inert"]})
                stats = self._stats.get_stream_stats(sc.stream_id)
                if stats:
                    stats.detections["inert"] += len(inert_ids)
            state.prev_inert_ids = inert_set
            for bid in inert_set:
                if bid not in dog_behavior:
                    dog_behavior[bid] = "inactive"

        # --- Draw bbox + label per dog (skip "none") ---
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
            if pet_name:
                label = f"{pet_name} {behavior}"
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
