# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

**Running the application:**
```bash
python main.py --method [bytetrack|botsort|deepsort] --input <video_file> --output <output_file> [options]
```

**Installing dependencies:**
```bash
pip install -r requirements.in
```

**Example usage:**
```bash
# Basic tracking with fight detection
python main.py --method bytetrack --input play.mp4 --output output.mp4 --task-fight

# Multiple behavior detection
python main.py --method botsort --input video.mp4 --output result.mp4 --task-fight --task-escape --task-inert

# Custom thresholds
python main.py --method deepsort --input input.mp4 --output output.mp4 --task-fight --threshold 0.15 --inert-threshold 30

# Sleep detection
python main.py --method bytetrack --input video.mp4 --output result.mp4 --task-sleep --sleep-threshold 20 --sleep-frames 300

# Eat detection (requires bowl class 3 in YOLO model)
python main.py --method bytetrack --input video.mp4 --output result.mp4 --task-eat --eat-iou-threshold 0.3 --eat-dwell-frames 30

# Bathroom detection (requires trained YOLO classify model)
python main.py --method bytetrack --input video.mp4 --output result.mp4 --task-bathroom --bathroom-cls-model weights/bathroom_cls.pt

# All behaviors at once
python main.py --method bytetrack --input video.mp4 --output result.mp4 --task-fight --task-inert --task-sleep --task-eat --task-bathroom
```

## Architecture Overview

This is a pet behavior tracking and analysis system using computer vision and object tracking algorithms. The system processes video files to detect and track pets, then analyzes their behavior patterns.

**Core Components:**

1. **main.py** - Entry point with CLI interface using Click. Orchestrates the entire pipeline from video input to processed output with detected behaviors.

2. **tracking.py** - Contains three tracking implementations:
   - `track_with_bytetrack()` - ByteTrack algorithm implementation
   - `track_with_botsort()` - BoT-SORT algorithm implementation
   - `track_with_deepsort()` - DeepSORT algorithm implementation
   All use YOLO for object detection (class 1 = pets) with configurable confidence/IoU thresholds.

3. **Behavior Detection Modules:**
   - **detect_fight.py** - Detects aggressive interactions between pets using pairwise IoU and sustained proximity counting
   - **detect_escape2.py** - Detects when pets leave a user-defined polygon ROI area (point-in-polygon)
   - **detect_inert.py** - Detects when pets remain stationary for extended periods (displacement threshold)
   - **detect_sleep.py** - Detects sleeping behavior: low displacement + bbox aspect ratio ≥ 1.2 (lying posture) + stable bbox area (CV ≤ 0.15)
   - **detect_eat.py** - Detects eating behavior using bowl (YOLO class 3) interaction: bowl-relative overlap (intersection/bowl_area), movement direction toward bowl (cosine similarity), and dwell time
   - **detect_bathroom.py** - Detects bathroom behavior with 2-phase pipeline: rule-based trigger (bbox height drop ≥ 25% + stationary) → YOLO classify model for confirmation

4. **deep_sort/** - Complete DeepSORT implementation with Kalman filtering, Hungarian assignment, and appearance descriptors for robust tracking.

**Key Architecture Patterns:**

- **Modular behavior detection**: Each behavior (fight, escape, inert, sleep, eat, bathroom) is implemented as a separate module with its own detection logic
- **Tracker abstraction**: All three tracking methods return the same interface (boxes, track_ids, frame)
- **State management**: Uses deques for coordinate history, torch tensors for interaction counting, and configurable frame buffers
- **Video processing pipeline**: Reads frames → detect/track objects → analyze behaviors → annotate frame → save output

**Configuration:**
- Tracker parameters are defined in `bytetrack.yaml` and `botsort.yaml`
- Model weights available in `weights/` directory (modelv11x.pt, modelv9e.pt)
- Default model path: `weights/best.pt` (configurable via --model parameter)
- All thresholds and frame counts are configurable via CLI parameters

## Multi-Stream Processing

**New parallel processing system for handling multiple video streams:**

**Running multi-stream processing:**
```bash
# From JSON config file
python main_multi.py run-config --config multi_stream_config.json

# Direct CLI with multiple streams
python main_multi.py run-streams --streams "video1.mp4,video2.mp4,0" --outputs "out1.mp4,out2.mp4,webcam.mp4"

# RTSP streams
python main_multi.py rtsp --rtsp-urls "rtsp://camera1,rtsp://camera2"

# Create sample config
python main_multi.py create-config
```

**Key Features:**
- **Async Processing**: Uses asyncio for concurrent stream handling
- **Batch Inference**: Optimizes GPU usage by processing multiple frames together
- **Memory Management**: Efficient frame buffering with compression
- **Resource Monitoring**: Real-time CPU/GPU/memory usage tracking
- **Auto-scaling**: Adaptive frame skipping based on system load
- **Real-time Support**: RTSP streams and webcam input

**Performance Optimizations:**
- `optimized_tracking.py` - Batch processing, memory management, adaptive frame skipping
- `multi_stream_processor.py` - Async architecture with thread pool execution
- GPU memory optimization with half-precision inference
- Compressed frame buffering to reduce memory usage

## Model Conversion and Optimization

**PyTorch Mobile Export (No TensorFlow dependency required):**
```bash
# Export to TorchScript for PyTorch Mobile/ExecuTorch
python model_cli.py mobile export --model weights/modelv11x.pt --format torchscript --half

# Export to ONNX for cross-platform mobile with INT8 quantization
python model_cli.py mobile export --model weights/modelv11x.pt --format onnx --optimize --int8

# Export to CoreML for iOS
python model_cli.py mobile export --model weights/modelv11x.pt --format coreml

# Apply pruning to reduce model size (50% sparsity)
python model_cli.py mobile prune --model weights/modelv11x.pt --sparsity 0.5

# Apply post-training quantization with calibration data
python model_cli.py mobile quantize --model weights/modelv11x.pt --calibration-data ./calibration_images

# Auto-optimize for mobile with size constraints
python model_cli.py mobile optimize --model weights/modelv11x.pt --target-size-mb 30

# Benchmark mobile model performance
python model_cli.py mobile benchmark --model model.pt --image test.jpg --format torchscript
```

**PyTorch Lightning Training:**
```bash
# Train with Lightning
python model_cli.py lightning train --data data.yaml --epochs 100 --batch-size 16

# Inference with Lightning model
python model_cli.py lightning infer --checkpoint model.ckpt --image test.jpg --output result.jpg

# Benchmark Lightning model
python model_cli.py lightning benchmark-lightning --checkpoint model.ckpt --test-images ./test_images
```

**Model Information:**
```bash
# Get model details
python model_cli.py info --model weights/best.pt
```

**Key Model Features:**
- **PyTorch Mobile/ExecuTorch**: TorchScript export optimized for mobile deployment
- **ONNX Export**: Cross-platform compatibility with ONNX Runtime Mobile
- **CoreML**: iOS-optimized models for Apple devices
- **Pruning**: Magnitude-based unstructured pruning for model compression
- **Quantization**: Post-training quantization (PTQ) with INT8/FP16 support
- **PyTorch Lightning**: Advanced training with callbacks, logging, and distributed training
- **Auto-Optimization**: Automatically find best mobile model within size constraints
- **Performance Benchmarking**: Compare different mobile model formats

**Supported Model Formats:**
- Original YOLO (.pt)
- TorchScript (.pt) - PyTorch Mobile/ExecuTorch
- ONNX (.onnx) - Cross-platform mobile
- CoreML (.mlpackage) - iOS optimized
- PyTorch Lightning (.ckpt) - Advanced training

## Development Workflow

**Testing the system:**
```bash
# Quick functionality test with sample video
python main.py --method bytetrack --input test_video.mp4 --output test_output.mp4 --task-fight

# Test multi-stream processing
python main_multi.py create-config
python main_multi.py run-config --config multi_stream_config.json

# Test mobile model export pipeline
python model_cli.py info --model weights/modelv11x.pt
python model_cli.py mobile export --model weights/modelv11x.pt --format torchscript
```

**Key Dependencies and Setup:**
- Python packages listed in `requirements.in`
- YOLO models in `weights/` directory
- Compatible with CUDA for GPU acceleration
- OpenCV for video processing, PyTorch for deep learning

**Important Notes:**
- Pet class ID is hardcoded as `1` in all tracking methods
- Bowl class ID is `3` (used by detect_eat for bowl detection)
- Video output uses H.264 encoding via PyAV library
- Interactive polygon selection required for escape detection
- All behavior detection modules are optional and can be enabled via CLI flags
- detect_eat requires bowl (class 3) to be in the YOLO detection model
- detect_bathroom requires a separate YOLO classify model (`weights/bathroom_cls.pt`)

## Behavior Detection Module Reference

| Module | File | Method | Detection Logic | Additional Model |
|--------|------|--------|----------------|-----------------|
| Fight | `detect_fight.py` | Rule-based | Pairwise IoU + sustained proximity counting | None |
| Escape | `detect_escape2.py` | Rule-based | Point-in-polygon ROI check | None |
| Inert | `detect_inert.py` | Rule-based | Displacement < threshold over N frames | None |
| Sleep | `detect_sleep.py` | Rule-based | Displacement + bbox aspect ratio (lying) + area stability (CV) | None |
| Eat | `detect_eat.py` | Rule + YOLO detect | Bowl overlap (intersection/bowl_area) + movement direction (cosine) + dwell time | Bowl detection (class 3) |
| Bathroom | `detect_bathroom.py` | Rule trigger + YOLO classify | Phase 1: bbox height drop + stationary → Phase 2: crop + classify | YOLO classify model |

**CLI flags for each module:**
- `--task-fight` / `--threshold`, `--reset-frames`, `--flag-frames`
- `--task-escape`
- `--task-inert` / `--inert-threshold`, `--inert-frames`
- `--task-sleep` / `--sleep-threshold`, `--sleep-frames`, `--sleep-aspect-ratio`, `--sleep-area-stability`
- `--task-eat` / `--eat-iou-threshold`, `--eat-dwell-frames`, `--eat-direction-frames`, `--bowl-conf`
- `--task-bathroom` / `--bathroom-cls-model`, `--bathroom-trigger-frames`, `--bathroom-height-drop`, `--bathroom-cls-conf`
- `--privacy` / `--privacy-method` (blur|mosaic|black), `--privacy-model`

## Privacy Filter

사람을 자동 감지하여 프라이버시를 보호하는 필터. 독립 CLI 또는 main.py 통합 사용 가능.

**독립 실행:**
```bash
python privacy_filter.py --input video.mp4 --output output.mp4 --method blur
python privacy_filter.py --input video.mp4 --output output.mp4 --method mosaic --mosaic-size 30
```

**main.py 파이프라인 통합:**
```bash
# 펫 트래킹 + 사람 블러 처리
python main.py --method bytetrack --input video.mp4 --output result.mp4 --task-fight --privacy
python main.py --method bytetrack --input video.mp4 --output result.mp4 --privacy --privacy-method mosaic
```

## Emoji Rendering

행동 감지 시 bbox 옆에 이모지를 표시. `emoji/` 디렉토리에 PNG 파일 필요.

```bash
# 이모지 PNG 생성 (최초 1회)
python generate_emoji.py

# emoji/ 폴더가 있으면 main.py 실행 시 자동 로드
```

생성되는 파일: fight.png (🥊), escape.png (⚠️), inert.png (❄️), play.png (🎾), sleep.png (😴), eat.png (🍽️), bathroom.png (🚽)

## ReID / Global ID Module Reference

**Re-Identification 및 글로벌 ID 관리 모듈 구조:**

| 모듈 | 파일 | 역할 |
|------|------|------|
| ReID Tracker | `reid_tracker.py` | ID 보정 + 글로벌 ID 파이프라인 (main.py에서 사용) |
| Global ID Manager | `global_id_manager.py` | 크로스 채널/스트림 글로벌 ID 매핑 |
| ReID Features (GPU) | `reid_features.py` | OSNet, EfficientNet, Histogram 기반 특징 추출 |
| ReID Lightweight (CPU) | `reid_lightweight.py` | FastHistogram, MobileNetV3, Adaptive 특징 추출 |
| ReID Image Matcher | `reid_image_matcher.py` | 레퍼런스 이미지 기반 펫 식별 CLI 도구 |
| Feature Framework | `features/` | 플러그인 방식 특징 추출/융합/매칭 프레임워크 |
| Pet Profile Store | `pet_profiles.py` | JSON 기반 펫 프로필(이름, 이미지, 정보) CRUD |

**펫 프로필 저장소 (`pet_profiles.py`):**
```bash
# 펫 프로필은 references/ 디렉토리에 저장
references/
├── pets.json           # 펫 메타데이터 (이름, 종, 정보)
└── images/             # 레퍼런스 이미지
    ├── poppi_001.jpg
    └── mimi_001.jpg
```

```python
from pet_profiles import PetProfileStore

store = PetProfileStore("references")
gid = store.add_pet(name="뽀삐", species="dog", breed="골든 리트리버")
store.add_reference_image(gid, "path/to/poppi.jpg")
store.save()

# ReID Image Matcher와 연동
refs = store.to_reid_references()  # reid_image_matcher 호환 형식
name_map = store.get_name_map()    # {global_id: name} 매핑
```

**ReID CLI 옵션:**
- `--use-reid` / `--reid-method` (adaptive|histogram|mobilenet), `--reid-threshold`
- `--reid-global-id` — 글로벌 ID 파이프라인 활성화

**Features 프레임워크 (`features/`):**
- `base.py` — FeatureExtractor ABC, TrackContext, FeatureOutput
- `appearance.py` — Histogram, MobileNet, Adaptive, ColorLayout 추출기
- `motion.py` — Motion, OpticalFlow, Trajectory 추출기
- `behavior.py` — Activity, Posture, Interaction, BehaviorPattern 추출기
- `fusion.py` — WeightedConcat, Attention, Adaptive 융합 전략
- `matching.py` — Cosine, Euclidean, Cascade, Greedy 매칭 전략
- `events.py` — TrackEventBus, IDSwitchHandler, OcclusionHandler