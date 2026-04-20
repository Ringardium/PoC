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

# Lock for serialising os.environ mutation during capture open (multi-stream safety)
_capture_env_lock = threading.Lock()


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
from tools.adaptive_fps import AdaptiveFPSController


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
        self.id_stable_threshold: int = 8  # N consecutive frames to confirm ID change
        # Output writer
        self.writer: Optional[cv2.VideoWriter] = None
        self.active: bool = False
        # Tracker reset flag: set True to use persist=False on next frame
        self.reset_tracker: bool = False
        self.no_detection_count: int = 0
        # Frame validation state
        self.prev_frame_hash: int = 0           # CRC32 of previous frame thumbnail
        self.stuck_frame_count: int = 0         # Consecutive identical-hash frames
        self.validation_fail_count: int = 0     # Consecutive validation failures
        # Adaptive frame skip state
        self.skip_counter: int = 0              # Remaining frames to skip YOLO
        self.last_boxes: list = []              # Cached detection results
        self.last_track_ids: list = []
        self.last_bowl_boxes: list = []
        self.last_proc_time: float = 0.0        # Last YOLO processing time (seconds)
        self.last_velocities: Dict = {}         # track_id -> (dx, dy) per frame for bbox interpolation
        self.prev_centers: Dict = {}            # track_id -> (cx, cy) from last YOLO frame
        self._yolo_frame_gap: int = 1           # frames between consecutive YOLO runs
        self._frames_since_yolo: int = 0        # counter incremented every frame
        self.last_dog_behavior: Dict[int, str] = {}  # bid -> behavior name (cached for skip frames)
        # Bbox EMA smoothing: track_id -> [cx, cy, w, h]
        self.smoothed_boxes: Dict[int, list] = {}
        # Privacy: cached person bboxes for skip frames (list of [x1,y1,x2,y2])
        self.last_person_boxes: list = []
        # Class stabilization: {track_id: {'cls': int, 'candidate': int, 'count': int}}
        self.track_class_buffer: Dict[int, Dict] = {}
        # Adaptive FPS controller (컨텐츠 기반 YOLO skip, config.adaptive_fps_enabled=True 시 초기화)
        self.adaptive_fps_ctrl: Optional[AdaptiveFPSController] = None
        self.last_analysis_time: float = 0.0  # AdaptiveFPSController용 마지막 분석 시각

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

        # Privacy filter is now handled within _run_tracking (same inference pass).
        # No separate model needed — person (class 0) is added to detection classes.

    # ------------------------------------------------------------------
    # Stream lifecycle
    # ------------------------------------------------------------------

    def _register_stream(self, sc: StreamConfig):
        self.streams[sc.stream_id] = sc
        self._states[sc.stream_id] = StreamState(sc)
        self._stats.register_stream(sc.stream_id)
        if sc.task_sleep:
            self._sleep_trackers[sc.stream_id] = BehaviorStateTracker("sleep")
        if sc.task_active:
            self._active_trackers[sc.stream_id] = BehaviorStateTracker("active")

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
            from reid.tracker import ReIDTrackerConfig
            reid_config = ReIDTrackerConfig(
                use_appearance=True,
                use_motion=True,
                use_deep_model=(sc.reid_method in ['adaptive', 'mobilenet']),
                similarity_threshold=sc.reid_threshold,
                correction_enabled=True,
                global_id_enabled=sc.reid_global_id,
                freeze_registered=sc.reid_freeze_registered,
            )
            state.reid_tracker = ReIDTracker(
                reid_method=sc.reid_method,
                config=reid_config,
            )
            # Load pet profiles: pre-register galleries + name map
            try:
                ref_dir = Path(__file__).resolve().parent / "references"
                # Pre-register pet reference images using tracker's own feature extractor
                state.reid_tracker.register_pet_profiles(str(ref_dir))
                # Load name map for label display
                from tools.pet_profiles import PetProfileStore
                store = PetProfileStore(str(ref_dir))
                state.global_id_names = store.get_name_map()
                if state.global_id_names:
                    logger.info(f"Loaded pet names: {state.global_id_names}")
            except Exception as e:
                logger.warning(f"Failed to load pet profiles: {e}")

        if needs_behavior:
            state.init_behavior_state()

        # Init AdaptiveFPSController (컨텐츠 기반 YOLO skip)
        if sc.adaptive_fps_enabled:
            state.adaptive_fps_ctrl = AdaptiveFPSController(
                max_fps=sc.adaptive_fps_max,
                min_fps=sc.adaptive_fps_min,
                idle_fps=sc.adaptive_fps_idle,
                displacement_low=sc.adaptive_fps_displacement_low,
                displacement_high=sc.adaptive_fps_displacement_high,
            )
            state.last_analysis_time = 0.0
            logger.info(
                f"[{stream_id}] AdaptiveFPS enabled — "
                f"max={sc.adaptive_fps_max}, idle={sc.adaptive_fps_idle}, "
                f"min={sc.adaptive_fps_min} fps"
            )

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
        next_frame_time = time.perf_counter()  # cumulative frame-rate target
        reconnect_attempts = 0
        max_reconnect_backoff = 30  # max seconds between reconnection attempts
        consecutive_errors = 0
        max_consecutive_errors = 50  # restart capture after this many processing errors

        try:
            while self.running and state.active:
                t0 = time.perf_counter()

                # Read frame in thread pool with timeout (blocking I/O)
                try:
                    read_future = loop.run_in_executor(self._executor, self._read_frame, state)
                    ret, frame = await asyncio.wait_for(read_future, timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning(f"[{stream_id}] cap.read() timed out (5s)")
                    # Discard the stuck cap object (don't release — thread may still be blocked)
                    state.cap = None
                    ret, frame = False, None
                if not ret or frame is None:
                    # For live streams (RTSP/RTMP), reconnect with exponential backoff
                    if self._is_live_stream(sc.input_source):
                        reconnect_attempts += 1
                        backoff = min(2 ** min(reconnect_attempts, 5), max_reconnect_backoff)
                        logger.warning(
                            f"[{stream_id}] Read failed — source={sc.input_source}, "
                            f"attempt={reconnect_attempts}, backoff={backoff}s"
                        )
                        if state.cap is not None:
                            try:
                                state.cap.release()
                            except Exception:
                                pass
                            state.cap = None
                        # Reset validation state
                        state.prev_frame_hash = 0
                        state.stuck_frame_count = 0
                        state.validation_fail_count = 0
                        # Estimate dropped frames during backoff
                        stats = self._stats.get_stream_stats(stream_id)
                        if stats:
                            stats.frames_dropped += int(backoff * sc.target_fps)
                        await asyncio.sleep(backoff)
                        state.cap = await self._open_capture(sc)
                        if state.cap is not None:
                            state.reset_tracker = True
                            logger.info(f"[{stream_id}] Reconnected (attempt {reconnect_attempts})")
                        else:
                            logger.error(f"[{stream_id}] Reconnection failed (attempt {reconnect_attempts})")
                        continue
                    else:
                        logger.info(f"[{stream_id}] End of video")
                        break

                # Reset reconnect counter on successful read
                reconnect_attempts = 0

                # --- Frame validation ---
                is_valid, reason = self._validate_frame(frame, state)
                if not is_valid:
                    state.validation_fail_count += 1
                    stats = self._stats.get_stream_stats(stream_id)
                    if stats:
                        stats.frames_dropped += 1
                    # Proactive reconnection after sustained validation failures
                    if state.validation_fail_count >= 90 and self._is_live_stream(sc.input_source):
                        logger.warning(
                            f"[{stream_id}] {state.validation_fail_count} consecutive "
                            f"validation failures (last: {reason}), reconnecting..."
                        )
                        state.validation_fail_count = 0
                        if state.cap is not None:
                            try:
                                state.cap.release()
                            except Exception:
                                pass
                            state.cap = None
                        await asyncio.sleep(1)
                        state.cap = await self._open_capture(sc)
                        if state.cap is not None:
                            state.reset_tracker = True
                    # Throttle loop on invalid frames to prevent spin
                    await asyncio.sleep(frame_interval)
                    continue
                else:
                    state.validation_fail_count = 0

                state.frame_cnt += 1
                _, state.frame_timestamp = self._sync_clock.stamp()

                # --- Processing block: wrapped so transient errors don't kill the stream ---
                try:
                    # --- YOLO tracking with adaptive frame skip ---
                    if needs_yolo:
                        run_yolo = True
                        state._frames_since_yolo += 1

                        # 1) 컨텐츠 기반 skip (AdaptiveFPSController) — 조용할 때 YOLO 자체를 건너뜀
                        if state.adaptive_fps_ctrl is not None:
                            if not state.adaptive_fps_ctrl.should_analyze(state.last_analysis_time):
                                run_yolo = False
                                boxes = self._interpolate_boxes(state)
                                track_ids = state.last_track_ids
                                bowl_boxes = state.last_bowl_boxes
                                if sc.privacy and state.last_person_boxes:
                                    for px1, py1, px2, py2 in state.last_person_boxes:
                                        if sc.privacy_method == "mosaic":
                                            frame = apply_mosaic(frame, px1, py1, px2, py2)
                                        elif sc.privacy_method == "black":
                                            frame = apply_black_box(frame, px1, py1, px2, py2)
                                        else:
                                            frame = apply_blur(frame, px1, py1, px2, py2)

                        # 2) 처리시간 기반 skip (enable_adaptive_skip) — 처리가 늦을 때 프레임 skip
                        if run_yolo and self.config.processing.enable_adaptive_skip and state.skip_counter > 0:
                            # Skip YOLO — reuse cached results with linear interpolation
                            state.skip_counter -= 1
                            boxes = self._interpolate_boxes(state)
                            track_ids = state.last_track_ids
                            bowl_boxes = state.last_bowl_boxes
                            run_yolo = False
                            # Apply cached privacy filter on skip frames
                            if sc.privacy and state.last_person_boxes:
                                for px1, py1, px2, py2 in state.last_person_boxes:
                                    if sc.privacy_method == "mosaic":
                                        frame = apply_mosaic(frame, px1, py1, px2, py2)
                                    elif sc.privacy_method == "black":
                                        frame = apply_black_box(frame, px1, py1, px2, py2)
                                    else:
                                        frame = apply_blur(frame, px1, py1, px2, py2)

                        if run_yolo:
                            # Record frame gap for velocity normalization
                            state._yolo_frame_gap = max(1, state._frames_since_yolo)
                            state._frames_since_yolo = 0

                            t_yolo = time.perf_counter()
                            frame, boxes, track_ids, bowl_boxes = await loop.run_in_executor(
                                self._executor, self._run_tracking, sc, state, frame
                            )
                            state.last_proc_time = time.perf_counter() - t_yolo

                            # EMA smoothing to reduce bbox jitter
                            if len(boxes) > 0 and len(track_ids) > 0:
                                boxes = self._smooth_boxes(state, boxes, track_ids)

                            # Update velocities from smoothed centers
                            self._update_velocities(state, boxes, track_ids)

                            # Cache results for skip frames
                            state.last_boxes = boxes
                            state.last_track_ids = track_ids
                            state.last_bowl_boxes = bowl_boxes

                            # Update AdaptiveFPSController (컨텐츠 기반 다음 skip 결정)
                            if state.adaptive_fps_ctrl is not None:
                                state.last_analysis_time = time.time()
                                num_objs = len(track_ids) if track_ids else 0
                                # 평균 변위 계산 (track_id별 이전 중심과의 거리)
                                avg_disp = 0.0
                                if num_objs > 0 and state.prev_centers:
                                    disps = []
                                    for tid, box in zip(track_ids, boxes):
                                        prev = state.prev_centers.get(tid)
                                        if prev is not None:
                                            cx = box[0] if not hasattr(box, 'tolist') else box.tolist()[0]
                                            cy = box[1] if not hasattr(box, 'tolist') else box.tolist()[1]
                                            dx, dy = cx - prev[0], cy - prev[1]
                                            disps.append((dx * dx + dy * dy) ** 0.5)
                                    if disps:
                                        avg_disp = sum(disps) / len(disps)
                                state.adaptive_fps_ctrl.update(
                                    num_objects=num_objs,
                                    avg_displacement=avg_disp,
                                    has_event=False,  # 행동 감지 후 갱신 필요 시 _run_behavior_detection에서 처리
                                )

                            # Decide how many frames to skip based on processing load
                            if self.config.processing.enable_adaptive_skip:
                                max_skip = self.config.processing.max_frame_skip
                                priority_factor = max(1, sc.priority)
                                if state.last_proc_time > frame_interval * 2:
                                    # Severely behind — skip maximum to catch up
                                    state.skip_counter = max_skip
                                elif state.last_proc_time > frame_interval:
                                    state.skip_counter = max(1, max_skip - priority_factor + 1)
                                else:
                                    state.skip_counter = 0

                        # --- ReID correction (optional, only on YOLO frames) ---
                        if run_yolo and state.reid_tracker is not None and len(boxes) > 0:
                            # --- Proximity lock: find track_ids whose bboxes are close ---
                            PROXIMITY_LOCK_DIST = 250  # pixels — lock corrections when closer
                            proximity_locked = set()
                            box_centers = []
                            for box in boxes:
                                if hasattr(box, 'tolist'):
                                    cx, cy = box.tolist()[:2]
                                else:
                                    cx, cy = box[0], box[1]
                                box_centers.append((cx, cy))
                            for i in range(len(box_centers)):
                                for j in range(i + 1, len(box_centers)):
                                    dx = box_centers[i][0] - box_centers[j][0]
                                    dy = box_centers[i][1] - box_centers[j][1]
                                    dist = (dx * dx + dy * dy) ** 0.5
                                    if dist < PROXIMITY_LOCK_DIST:
                                        proximity_locked.add(track_ids[i])
                                        proximity_locked.add(track_ids[j])

                            reid_result = state.reid_tracker.process(
                                frame, boxes, track_ids,
                                proximity_locked=proximity_locked,
                            )
                            raw_corrected = reid_result['corrected_ids']

                            # ID stabilization: only accept correction after N consistent frames
                            stable_ids = []
                            for orig_tid, corr_tid in zip(track_ids, raw_corrected):
                                if orig_tid == corr_tid:
                                    # No correction — clear buffer and keep original
                                    state.id_stable_buffer.pop(orig_tid, None)
                                    stable_ids.append(orig_tid)
                                elif orig_tid in proximity_locked:
                                    # Proximity lock — reject correction, keep original
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

                            # Global ID assignment — enforce uniqueness within frame
                            if reid_result.get('global_ids'):
                                # Collect gids already claimed by current-frame track_ids
                                frame_gids_used = {}  # gid -> tid that owns it
                                for tid in track_ids:
                                    existing = state.global_id_map.get(tid)
                                    if existing is not None:
                                        frame_gids_used[existing] = tid

                                for tid, gid in zip(track_ids, reid_result['global_ids']):
                                    if tid in proximity_locked and tid in state.global_id_map:
                                        continue  # Keep existing global_id when close
                                    if gid is None:
                                        continue
                                    # Check if this gid is already taken by another track in this frame
                                    owner = frame_gids_used.get(gid)
                                    if owner is not None and owner != tid:
                                        # Already assigned to a different track — skip
                                        continue
                                    # Revoke old gid from frame_gids_used if tid is switching
                                    old_gid = state.global_id_map.get(tid)
                                    if old_gid is not None and old_gid != gid:
                                        if frame_gids_used.get(old_gid) == tid:
                                            del frame_gids_used[old_gid]
                                    state.global_id_map[tid] = gid
                                    frame_gids_used[gid] = tid

                                # Prune stale entries: remove tracks no longer in frame
                                active_tids = set(track_ids)
                                stale = [t for t in state.global_id_map if t not in active_tids]
                                for t in stale:
                                    del state.global_id_map[t]

                        # --- Behavior detection (only on YOLO frames) ---
                        if needs_behavior and len(boxes) > 0:
                            if run_yolo:
                                frame = await loop.run_in_executor(
                                    self._executor, self._run_behavior_detection,
                                    sc, state, frame, boxes, track_ids, bowl_boxes,
                                )
                            else:
                                # Skip frames — draw cached labels without re-running detection
                                frame = self._draw_cached_labels(state, frame, boxes, track_ids)

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

                # --- Frame rate control (cumulative — no sleep when behind) ---
                next_frame_time += frame_interval
                now = time.perf_counter()
                if now < next_frame_time:
                    await asyncio.sleep(next_frame_time - now)
                else:
                    # Cap debt after reconnection / long stall to avoid burst
                    if now - next_frame_time > frame_interval * 5:
                        next_frame_time = now
                    await asyncio.sleep(0)

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
    def _validate_frame(
        frame: np.ndarray,
        state: "StreamState",
        stuck_threshold: int = 90,
    ) -> Tuple[bool, str]:
        """Validate a frame for corruption, stuck, or blank conditions.

        Returns (is_valid, reason). Runs in < 1ms using downsampled checks.
        """
        import zlib

        # 1. Shape check
        if frame.ndim != 3 or frame.shape[2] != 3:
            return False, "invalid_shape"
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return False, "zero_dimensions"

        # 2. Stuck frame detection via CRC32 of 16x16 thumbnail
        thumb = cv2.resize(frame, (16, 16), interpolation=cv2.INTER_NEAREST)
        frame_hash = zlib.crc32(thumb.tobytes())

        if frame_hash == state.prev_frame_hash:
            state.stuck_frame_count += 1
            if state.stuck_frame_count >= stuck_threshold:
                return False, "stuck_frame"
        else:
            state.stuck_frame_count = 0
            state.prev_frame_hash = frame_hash

        # 3. Green corruption check (FFmpeg decode artifact)
        cy, cx = h // 2, w // 2
        if h >= 32 and w >= 32:
            patch = frame[cy - 16:cy + 16, cx - 16:cx + 16]
        else:
            patch = thumb
        b_mean = float(patch[:, :, 0].mean())
        g_mean = float(patch[:, :, 1].mean())
        r_mean = float(patch[:, :, 2].mean())
        if g_mean > 200 and (g_mean - b_mean) > 80 and (g_mean - r_mean) > 80:
            return False, "green_corruption"

        # 4. Black/blank frame check
        if float(thumb.mean()) < 5.0:
            return False, "black_frame"

        return True, "ok"

    @staticmethod
    def _update_velocities(state: "StreamState", boxes, track_ids):
        """Compute per-frame velocity from previous centers for bbox interpolation.

        Divides displacement by the number of frames since last YOLO run
        so that each skip frame shifts by exactly 1 frame worth of motion.
        """
        # frames_gap = 1 (no skip) or 1 + skipped frames since last YOLO
        frames_gap = getattr(state, '_yolo_frame_gap', 1)
        new_centers = {}
        for tid, box in zip(track_ids, boxes):
            if hasattr(box, 'tolist'):
                cx, cy = box.tolist()[:2]
            else:
                cx, cy = box[0], box[1]
            new_centers[tid] = (cx, cy)
            if tid in state.prev_centers:
                px, py = state.prev_centers[tid]
                state.last_velocities[tid] = (
                    (cx - px) / frames_gap,
                    (cy - py) / frames_gap,
                )
            else:
                state.last_velocities[tid] = (0.0, 0.0)
        state.prev_centers = new_centers

    @staticmethod
    def _smooth_boxes(state: "StreamState", boxes, track_ids, alpha: float = 0.6):
        """Apply EMA smoothing to reduce bbox jitter.

        smoothed = alpha * new + (1 - alpha) * previous
        alpha=0.6: responsive but stable. Lower = smoother but laggier.
        Returns list of smoothed boxes (same type as input).
        """
        if not track_ids or len(boxes) == 0:
            return boxes

        is_tensor = hasattr(boxes[0], 'clone') if len(boxes) > 0 else False
        smoothed = []
        for tid, box in zip(track_ids, boxes):
            curr = box.tolist() if hasattr(box, 'tolist') else list(box)
            prev = state.smoothed_boxes.get(tid)
            if prev is not None:
                s = [alpha * curr[i] + (1 - alpha) * prev[i] for i in range(4)]
            else:
                s = curr
            state.smoothed_boxes[tid] = s
            smoothed.append(s)

        # Clean up stale tracks
        active = set(track_ids)
        state.smoothed_boxes = {k: v for k, v in state.smoothed_boxes.items() if k in active}

        if is_tensor:
            return torch.tensor(smoothed, dtype=boxes[0].dtype)
        return smoothed

    @staticmethod
    def _interpolate_boxes(state: "StreamState"):
        """Shift cached bboxes by their velocity for skip frames."""
        if state.last_boxes is None or len(state.last_boxes) == 0 or not state.last_track_ids:
            return state.last_boxes
        interpolated = []
        for tid, box in zip(state.last_track_ids, state.last_boxes):
            dx, dy = state.last_velocities.get(tid, (0.0, 0.0))
            if hasattr(box, 'clone'):
                new_box = box.clone()
                new_box[0] += dx
                new_box[1] += dy
            else:
                new_box = list(box)
                new_box[0] += dx
                new_box[1] += dy
            interpolated.append(new_box)
        # Update cached boxes so next skip frame continues from interpolated position
        if interpolated:
            if hasattr(state.last_boxes[0], 'clone'):
                state.last_boxes = torch.stack(interpolated) if len(interpolated) > 0 else []
            else:
                state.last_boxes = interpolated
        return state.last_boxes

    @staticmethod
    def _draw_cached_labels(
        state: "StreamState",
        frame: np.ndarray,
        boxes,
        track_ids: list,
    ) -> np.ndarray:
        """Draw bboxes with cached behavior labels (lightweight, for skip frames)."""
        COLOR_MAP = {
            "fight": (0, 0, 255), "escape": (0, 255, 255),
            "sleeping": (128, 0, 128), "bathroom": (235, 206, 135),
            "feeding": (180, 105, 255), "playing": (0, 165, 255),
            "inactive": (180, 0, 0), "normal": (0, 200, 0),
            "none": (144, 238, 144),
        }
        sc = state.config
        tid_to_bid = {}
        used_bids = set()
        for tid in track_ids:
            gid = state.global_id_map.get(tid)
            bid = gid if gid is not None else tid
            if bid in used_bids:
                bid = tid  # Duplicate global_id — fall back to track_id
            used_bids.add(bid)
            tid_to_bid[tid] = bid

        for tid, box in zip(track_ids, boxes):
            bid = tid_to_bid.get(tid, tid)
            behavior = state.last_dog_behavior.get(bid, "normal")
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
                label = f"{pet_name} {behavior}" if sc.show_track_id else behavior
            elif sc.label_registered_only:
                cv2.rectangle(frame, pt1, pt2, COLOR_MAP["none"], 1)
                continue
            elif not sc.show_track_id:
                label = behavior
            elif gid is not None:
                label = f"G:{gid} {behavior}"
            else:
                label = f"#{tid} {behavior}"

            cv2.rectangle(frame, pt1, pt2, color, 2)
            if behavior != "normal" or pet_name:
                display_label = pet_name if (behavior == "normal" and pet_name) else label
                (tw, th), _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (pt1[0], pt1[1] - th - 10), (pt1[0] + tw, pt1[1]), color, -1)
                cv2.putText(frame, display_label, (pt1[0], pt1[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return frame

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
            with _capture_env_lock:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                    "rtsp_transport;tcp"
                    "|fflags;nobuffer"
                    "|analyzeduration;5000000"
                    "|probesize;5000000"
                    "|stimeout;5000000"
                    "|max_delay;500000"
                    "|reorder_queue_size;150"
                )
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            # Flush initial frames to skip past incomplete GOP (outside lock)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                for _ in range(60):
                    cap.grab()
        elif source.lower().startswith("rtmp://"):
            with _capture_env_lock:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                    "fflags;nobuffer"
                    "|flags;low_delay"
                    "|analyzeduration;500000"
                    "|probesize;500000"
                    "|timeout;5000000"
                )
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(source)
        elapsed = time.perf_counter() - t0
        if cap.isOpened():
            logger.info(f"[capture] Opened in {elapsed:.1f}s: {source}")
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

        When ``sc.privacy`` is enabled, person (class 0) is detected in the
        same inference pass and privacy filter is applied to the frame before
        returning — no separate model call needed.

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

            # Build detection classes: pets + bowls + person (if privacy)
            det_classes = list(sc.yolo_classes or [1])
            if sc.privacy and 0 not in det_classes:
                det_classes.append(0)

            with self._gpu_sem:
                results = state.model.track(
                    frame, persist=use_persist, conf=sc.yolo_conf, iou=sc.yolo_iou,
                    classes=det_classes, tracker=tracker_yaml,
                    device=device, half=half, verbose=False,
                )
            if state.reset_tracker:
                state.reset_tracker = False
                state.track_class_buffer.clear()
            ids = results[0].boxes.id
            cls = results[0].boxes.cls.cpu() if results[0].boxes.cls is not None else None

            if ids is None:
                state.no_detection_count += 1
                # Auto-reset tracker if no detections for 90+ consecutive frames
                if state.no_detection_count >= 90:
                    state.reset_tracker = True
                    state.no_detection_count = 0
                boxes, track_ids, bowl_boxes = [], [], []
                # Privacy fallback: blur raw person detections even without tracking
                if sc.privacy and cls is not None:
                    all_xyxy_np = results[0].boxes.xyxy.cpu().numpy()
                    person_boxes = []
                    for i, c in enumerate(cls):
                        if int(c) == 0:
                            x1, y1, x2, y2 = map(int, all_xyxy_np[i])
                            pad = 10
                            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                            x2 = min(frame.shape[1], x2 + pad)
                            y2 = min(frame.shape[0], y2 + pad)
                            person_boxes.append([x1, y1, x2, y2])
                            if sc.privacy_method == "mosaic":
                                frame = apply_mosaic(frame, x1, y1, x2, y2)
                            elif sc.privacy_method == "black":
                                frame = apply_black_box(frame, x1, y1, x2, y2)
                            else:
                                frame = apply_blur(frame, x1, y1, x2, y2)
                    state.last_person_boxes = person_boxes
                elif sc.privacy:
                    state.last_person_boxes = []
            else:
                state.no_detection_count = 0
                all_xywh = results[0].boxes.xywh.cpu()
                all_xyxy = results[0].boxes.xyxy.cpu().numpy()
                all_ids = results[0].boxes.id.int().cpu().tolist()
                all_conf = results[0].boxes.conf.cpu()

                # --- Class stabilization per track ID ---
                # Require 3 consecutive frames of new class before accepting change.
                # Prevents person↔dog flickering from creating ghost tracks or
                # dropping privacy filter.
                CLASS_STABLE_FRAMES = 3
                stable_cls = []
                for i in range(len(cls)):
                    tid = all_ids[i]
                    raw_c = int(cls[i])
                    buf = state.track_class_buffer.get(tid)
                    if buf is None:
                        # New track — accept class immediately
                        state.track_class_buffer[tid] = {'cls': raw_c}
                        stable_cls.append(raw_c)
                    elif raw_c == buf['cls']:
                        # Same as stabilized class — clear pending change
                        buf.pop('candidate', None)
                        buf.pop('count', None)
                        stable_cls.append(raw_c)
                    elif buf.get('candidate') == raw_c:
                        buf['count'] = buf.get('count', 0) + 1
                        if buf['count'] >= CLASS_STABLE_FRAMES:
                            # Confirmed class change
                            buf['cls'] = raw_c
                            buf.pop('candidate', None)
                            buf.pop('count', None)
                            stable_cls.append(raw_c)
                        else:
                            # Not yet confirmed — keep old class
                            stable_cls.append(buf['cls'])
                    else:
                        # New candidate class
                        buf['candidate'] = raw_c
                        buf['count'] = 1
                        stable_cls.append(buf['cls'])

                # Privacy filter using stabilized classes
                if sc.privacy:
                    person_boxes = []
                    for i, c in enumerate(stable_cls):
                        if c == 0:
                            x1, y1, x2, y2 = map(int, all_xyxy[i])
                            pad = 10
                            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                            x2 = min(frame.shape[1], x2 + pad)
                            y2 = min(frame.shape[0], y2 + pad)
                            person_boxes.append([x1, y1, x2, y2])
                            if sc.privacy_method == "mosaic":
                                frame = apply_mosaic(frame, x1, y1, x2, y2)
                            elif sc.privacy_method == "black":
                                frame = apply_black_box(frame, x1, y1, x2, y2)
                            else:
                                frame = apply_blur(frame, x1, y1, x2, y2)
                    state.last_person_boxes = person_boxes

                # Separate pets (class 1) and bowls (class 3) using stabilized classes
                pet_entries = []  # (conf, xywh, id)
                bowl_boxes = []
                for i, c in enumerate(stable_cls):
                    if c == 1:
                        pet_entries.append((float(all_conf[i]), all_xywh[i], all_ids[i]))
                    elif c == 3:
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
        # Enforce uniqueness: if two track_ids share the same global_id,
        # only the first keeps it; the second falls back to raw track_id.
        behavior_ids = []
        tid_to_bid = {}
        used_bids = set()
        for tid in track_ids:
            gid = state.global_id_map.get(tid)
            bid = gid if gid is not None else tid
            if bid in used_bids:
                # Duplicate global_id in this frame — fall back to track_id
                bid = tid
            used_bids.add(bid)
            behavior_ids.append(bid)
            tid_to_bid[tid] = bid

        def _resolve_dog_id(bid):
            """Return dog ID for API events.

            Always returns bid so events are sent for all detected pets,
            whether they have a global_id or just a corrected track_id.
            """
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
            state.sleep_coor[bid].append([state.frame_timestamp, xc, yc])
            state.sleep_bbox[bid].append([state.frame_timestamp, w, h])
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
            fight_indices, state.close_count, state.far_count = detect_fight(
                x_arr, y_arr, track_ids, state.close_count, state.far_count,
                sc.threshold, sc.reset_frames, sc.flag_frames,
                last_w, last_h,
                speeds=speeds,
                fight_speed_threshold_px_sec=sc.fight_speed_threshold,
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

        # Inert detection (sleep 감지된 ID 제외)
        inert_set = set()
        if sc.task_inert:
            inert_ids = detect_inert(state.inert_coor, sc.inert_threshold, sc.inert_frames)
            inert_set = set(inert_ids) - sleep_set  # sleep이 우선
            if len(inert_set) > 0:
                for bid in (inert_set - state.prev_inert_ids):
                    _send_event({"dogId": _resolve_dog_id(bid), "behaviorType": BEHAVIOR_TYPE_MAP["inert"]})
                stats = self._stats.get_stream_stats(sc.stream_id)
                if stats:
                    stats.detections["inert"] += len(inert_set)
            state.prev_inert_ids = inert_set
            for bid in inert_set:
                if bid not in dog_behavior:
                    dog_behavior[bid] = "inactive"

        # Cache behavior labels for skip frames
        state.last_dog_behavior = dict(dog_behavior)

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
                label = f"{pet_name} {behavior}" if sc.show_track_id else behavior
            elif registered_only:
                # Unregistered pet — bbox only, no label
                cv2.rectangle(frame, pt1, pt2, COLOR_MAP["none"], 1)
                continue
            elif not sc.show_track_id:
                label = behavior
            elif gid is not None:
                label = f"G:{gid} {behavior}"
            else:
                label = f"#{tid} {behavior}"

            cv2.rectangle(frame, pt1, pt2, color, 2)
            if behavior != "normal" or pet_name:
                # Label background
                display_label = pet_name if (behavior == "normal" and pet_name) else label
                (tw, th), _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (pt1[0], pt1[1] - th - 10), (pt1[0] + tw, pt1[1]), color, -1)
                cv2.putText(frame, display_label, (pt1[0], pt1[1] - 6),
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
