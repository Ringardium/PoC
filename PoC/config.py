"""Configuration management for multi-stream processing"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class StreamConfig:
    """Configuration for a single stream"""
    stream_id: str
    input_source: str  # video file, RTSP URL, or webcam index (e.g., "0")
    output_path: Optional[str] = None
    method: str = "bytetrack"  # bytetrack, botsort, deepsort, ocsort
    task_fight: bool = True
    task_escape: bool = False
    task_inert: bool = True
    task_sleep: bool = False
    task_eat: bool = False
    task_bathroom: bool = False
    task_active: bool = False
    threshold: float = 0.3
    fight_speed_threshold: float = 5.0
    inert_threshold: float = 1.0   # avg speed (px/frame) — below this = inert
    inert_frames: int = 150
    sleep_threshold: float = 30.0
    sleep_frames: int = 200
    sleep_aspect_ratio: float = 1.2
    sleep_area_stability: float = 0.15
    eat_iou_threshold: float = 0.3
    eat_dwell_frames: int = 30
    eat_direction_frames: int = 10
    bowl_conf: float = 0.5
    bathroom_cls_model: str = "../weights/bathroom_cls.pt"
    bathroom_trigger_frames: int = 30
    bathroom_height_drop: float = 0.25
    bathroom_cls_conf: float = 0.5
    active_threshold: float = 800.0  # ignored (kept for backward compat)
    active_frames: int = 90
    active_speed_threshold: float = 5.0  # avg speed (px/frame) — above this = active
    reset_frames: int = 20
    flag_frames: int = 40
    priority: int = 1  # 1=highest, 3=lowest
    target_fps: int = 30
    escape_polygon: Optional[List[Tuple[int, int]]] = None
    use_reid: bool = False
    reid_method: str = "adaptive"       # adaptive, histogram, mobilenet
    reid_threshold: float = 0.5
    reid_global_id: bool = False
    reid_freeze_registered: bool = True  # True: 등록 프로필 특징 고정 (드리프트 방지)
    # YOLO inference settings
    yolo_conf: float = 0.7              # confidence threshold
    yolo_iou: float = 0.5               # IoU threshold for NMS
    yolo_classes: List[int] = field(default_factory=lambda: [1])  # detection classes (1=pet)
    yolo_augment: bool = False           # test-time augmentation (slower but more accurate)
    yolo_agnostic_nms: bool = False      # class-agnostic NMS
    yolo_verbose: bool = False           # verbose inference logging
    yolo_persist: bool = True            # persist tracks across frames
    # Label display
    label_registered_only: bool = False  # True = 등록된 반려동물(pet_profiles)만 라벨 표시
    show_track_id: bool = True           # True = 라벨에 track/global ID 표시, False = 행동명만 표시
    # Privacy filter
    privacy: bool = False                # 사람 감지 후 프라이버시 필터 적용
    privacy_method: str = "blur"         # blur, mosaic, black
    privacy_model: str = "yolo11n.pt"    # 사람 감지용 YOLO 모델 경로
    # Adaptive FPS (컨텐츠 기반 YOLO 추론 주기 자동 조절)
    # ProcessingConfig.enable_adaptive_skip(처리시간 기반)과 독립적으로 동작
    adaptive_fps_enabled: bool = False   # False = 항상 매 프레임 YOLO 실행
    adaptive_fps_max: float = 10.0       # 활발할 때 최대 분석 FPS
    adaptive_fps_min: float = 0.1        # 객체 없을 때 최소 분석 FPS (= 10초에 1회)
    adaptive_fps_idle: float = 1.0       # 객체 있지만 조용할 때 분석 FPS
    adaptive_fps_displacement_low: float = 5.0   # 저움직임 임계값 (px/frame)
    adaptive_fps_displacement_high: float = 50.0 # 고움직임 임계값 (px/frame)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["yolo_classes"] = list(self.yolo_classes)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "StreamConfig":
        if "yolo_classes" in data:
            data["yolo_classes"] = list(data["yolo_classes"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class GPUConfig:
    """GPU-specific configuration"""
    device_id: int = 0
    memory_fraction: float = 0.9  # Use 90% of GPU memory
    half_precision: bool = True  # FP16 for A100
    batch_size: int = 6  # Optimal for 6 streams at 30fps
    enable_cudnn_benchmark: bool = True
    enable_tf32: bool = True  # A100 TF32 support


@dataclass
class ProcessingConfig:
    """Processing pipeline configuration"""
    max_streams: int = 6
    target_resolution: Tuple[int, int] = (640, 640)
    frame_buffer_size: int = 60  # Increased for 30fps
    inference_timeout: float = 0.033  # 33ms for 30fps real-time
    enable_adaptive_skip: bool = True
    max_frame_skip: int = 2  # Reduced for better real-time
    min_fps_threshold: int = 25  # Minimum acceptable FPS for 30fps target


@dataclass
class SystemConfig:
    """Overall system configuration"""
    model_path: str = "../weights/best.pt"
    gpu: GPUConfig = field(default_factory=GPUConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    streams: List[StreamConfig] = field(default_factory=list)
    log_level: str = "INFO"
    stats_interval: float = 5.0  # seconds
    event_api_url: Optional[str] = None  # Backend URL for behavior events (None = disabled)
    hls_s3_bucket: Optional[str] = None  # S3 bucket for HLS output (None = disabled)
    hls_s3_prefix: str = "hls/live"      # S3 key prefix
    clip_s3_prefix: str = "clips/events" # S3 key prefix for event clips
    clip_pre_seconds: float = 2.0        # 이벤트 전 녹화 시간 (초)
    clip_post_seconds: float = 2.0       # 이벤트 후 녹화 시간 (초)
    cdn_base_url: Optional[str] = None   # CDN base URL (e.g. "https://cdn.mungwoofai.cloud")

    def to_dict(self) -> dict:
        data = {
            "model_path": self.model_path,
            "gpu": asdict(self.gpu),
            "processing": {
                **asdict(self.processing),
                "target_resolution": list(self.processing.target_resolution)
            },
            "streams": [s.to_dict() for s in self.streams],
            "log_level": self.log_level,
            "stats_interval": self.stats_interval,
            "event_api_url": self.event_api_url,
            "hls_s3_bucket": self.hls_s3_bucket,
            "hls_s3_prefix": self.hls_s3_prefix,
            "clip_s3_prefix": self.clip_s3_prefix,
            "clip_pre_seconds": self.clip_pre_seconds,
            "clip_post_seconds": self.clip_post_seconds,
            "cdn_base_url": self.cdn_base_url,
        }
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SystemConfig":
        gpu = GPUConfig(**data.get("gpu", {}))

        proc_data = data.get("processing", {})
        if "target_resolution" in proc_data:
            proc_data["target_resolution"] = tuple(proc_data["target_resolution"])
        processing = ProcessingConfig(**proc_data)

        streams = [StreamConfig.from_dict(s) for s in data.get("streams", [])]

        return cls(
            model_path=data.get("model_path", "../weights/best.pt"),
            gpu=gpu,
            processing=processing,
            streams=streams,
            log_level=data.get("log_level", "INFO"),
            stats_interval=data.get("stats_interval", 5.0),
            event_api_url=data.get("event_api_url"),
            hls_s3_bucket=data.get("hls_s3_bucket"),
            hls_s3_prefix=data.get("hls_s3_prefix", "hls/live"),
            clip_s3_prefix=data.get("clip_s3_prefix", "clips/events"),
            clip_pre_seconds=data.get("clip_pre_seconds", 2.0),
            clip_post_seconds=data.get("clip_post_seconds", 2.0),
            cdn_base_url=data.get("cdn_base_url"),
        )

    def save(self, path: str):
        """Save configuration to JSON file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "SystemConfig":
        """Load configuration from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


def create_sample_config(num_streams: int = 4) -> SystemConfig:
    """Create a sample configuration for 30fps real-time processing"""
    if num_streams > 6:
        num_streams = 6  # Max 6 streams for 30fps real-time

    streams = []
    for i in range(num_streams):
        stream = StreamConfig(
            stream_id=f"camera_{i+1}",
            input_source=f"rtsp://192.168.1.{100+i}:8080/video",
            output_path=f"output/stream_{i+1}.mp4",
            method="bytetrack",
            task_fight=True,
            task_inert=True,
            priority=1 if i < 2 else 2,  # First 2 streams have higher priority
            target_fps=30
        )
        streams.append(stream)

    config = SystemConfig(
        model_path="../weights/best.pt",
        gpu=GPUConfig(
            device_id=0,
            memory_fraction=0.9,
            half_precision=True,
            batch_size=min(num_streams, 6)
        ),
        processing=ProcessingConfig(
            max_streams=6,
            target_resolution=(640, 640),
            frame_buffer_size=60,
            enable_adaptive_skip=True,
            max_frame_skip=2,
            min_fps_threshold=25
        ),
        streams=streams
    )
    return config
