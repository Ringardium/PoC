# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Structure

```
├── main.py                  # 싱글 스트림 CLI (Click)
├── tracking.py              # ByteTrack / BotSORT / DeepSORT 트래커
├── bytetrack.yaml           # ByteTrack 설정
├── botsort.yaml             # BotSORT 설정
│
├── detection/               # 행동 감지 모듈
│   ├── fight.py             # 싸움 감지 (IoU + proximity)
│   ├── inert.py             # 비활동 감지 (displacement)
│   ├── sleep.py             # 수면 감지 (aspect ratio + stability)
│   ├── eat.py               # 식사 감지 (bowl overlap + direction)
│   ├── bathroom.py          # 배변 감지 (height drop + classify)
│   ├── active.py            # 활동 감지 (high displacement)
│   ├── escape.py            # 탈출 감지 (polygon ROI)
│   └── utils.py             # IoU 유틸리티
│
├── reid/                    # Re-ID 시스템
│   ├── tracker.py           # ReIDTracker (ID 보정 + 글로벌 ID)
│   ├── image_matcher.py     # 레퍼런스 이미지 매칭 CLI
│   ├── extractor.py         # GPU 특징 추출 (OSNet, Histogram)
│   ├── lightweight.py       # CPU 특징 추출 (FastHistogram, MobileNet)
│   ├── global_id.py         # 크로스 스트림 글로벌 ID
│   └── features/            # 플러그인 특징 프레임워크
│       ├── base.py, appearance.py, motion.py
│       ├── behavior.py, fusion.py, matching.py, events.py
│
├── models/                  # 모델 관리/최적화
│   ├── cli.py               # 모델 변환 CLI (export, prune, quantize)
│   ├── converter.py         # 포맷 변환 (TorchScript, ONNX, CoreML)
│   ├── optimization.py      # 최적화 (pruning, quantization)
│   ├── lightning_training.py # PyTorch Lightning 학습
│   └── lightning_inference.py
│
├── tools/                   # 유틸리티
│   ├── privacy_filter.py    # 프라이버시 필터 (blur/mosaic/black)
│   ├── pet_profiles.py      # 펫 프로필 CRUD (JSON)
│   ├── coord_picker.py      # ROI 좌표 선택기
│   └── generate_emoji.py    # 이모지 PNG 생성
│
├── deep_sort/               # DeepSORT 구현
├── assets/                  # 정적 에셋
│   ├── emoji/               # 행동 이모지 PNG
│   └── fonts/               # 폰트 파일
│
├── PoC/                     # 멀티스트림 서버 시스템
│   ├── main.py              # 서버 CLI
│   ├── stream_processor.py  # 비동기 멀티스트림 처리
│   ├── config.py            # SystemConfig / StreamConfig
│   ├── web_server.py        # FastAPI + WebSocket
│   ├── monitor.py           # 실시간 성능 모니터링
│   ├── event_sender.py      # 행동 이벤트 API
│   ├── hls_uploader.py      # HLS → S3 업로드
│   └── templates/           # 웹 UI
│
└── legacy/                  # Deprecated 파일
```

## Common Development Commands

```bash
# 싱글 스트림 추적 + 행동 감지
python main.py --method bytetrack --input video.mp4 --output result.mp4 --task-fight --task-sleep --task-eat

# 프라이버시 필터 포함
python main.py --method bytetrack --input video.mp4 --output result.mp4 --task-fight --privacy --privacy-method blur

# 멀티스트림 서버 실행
cd PoC && python main.py run --config sample_config.json

# 모델 최적화
python -m models.cli mobile export --model weights/best.pt --format torchscript --half

# 이모지 PNG 생성
python tools/generate_emoji.py

# 프라이버시 필터 독립 실행
python tools/privacy_filter.py --input video.mp4 --output output.mp4 --method blur
```

## Architecture

### 트래킹 파이프라인
```
프레임 → 프라이버시 필터 → YOLO 감지/추적 → ReID 보정 → 행동 분석 → 시각화
```

### 핵심 규칙
- Pet class ID = `1`, Bowl class ID = `3`
- `task_eat=True` → yolo_classes에 class 3 자동 추가
- 모든 행동 감지 모듈은 독립적 (cross-import 없음)
- tracking.py의 3개 트래커는 동일 인터페이스 반환: `(boxes, track_ids, frame)`

## Import 패턴

```python
# 행동 감지
from detection import detect_fight, detect_sleep, detect_eat

# ReID
from reid import ReIDTracker

# 유틸리티
from tools import apply_blur, apply_mosaic, apply_black_box
from tools.pet_profiles import PetProfileStore
```

## Behavior Detection Reference

| Module | File | Detection Logic |
|--------|------|----------------|
| Fight | `detection/fight.py` | Pairwise IoU + sustained proximity counting |
| Inert | `detection/inert.py` | Displacement < threshold over N frames |
| Sleep | `detection/sleep.py` | Low displacement + aspect ratio + area stability |
| Eat | `detection/eat.py` | Bowl overlap + movement direction + dwell time |
| Bathroom | `detection/bathroom.py` | Height drop trigger → YOLO classify confirm |
| Active | `detection/active.py` | High displacement over sustained period |
| Escape | `detection/escape.py` | Point-in-polygon ROI check |

## Pet Profile Storage

```
references/
├── pets.json         # 펫 프로필 데이터
└── images/           # 레퍼런스 이미지
```

```python
from tools.pet_profiles import PetProfileStore
store = PetProfileStore("references")
gid = store.add_pet(name="뽀삐", species="dog")
store.add_reference_image(gid, "photo.jpg")
store.save()
```
