# PoC - Multi-Stream Pet Tracking System

A100 GPU 최적화된 멀티스트림 반려동물 추적 시스템.
최대 6개 스트림 동시 처리 (30fps 실시간 보장), 웹 기반 실시간 모니터링 지원.

## 빠른 시작

```bash
cd PoC

# 샘플 설정으로 실행
python main.py run --config sample_config.json

# CLI 옵션으로 직접 실행
python main.py run \
    --streams "../data/fight/fight1.mp4,../data/fight/fight2.mp4" \
    --outputs "output/out1.mp4,output/out2.mp4" \
    --model ../weights/best.pt

# 웹 뷰어로 실시간 모니터링
python web_server.py --config sample_config.json --port 8000
# 브라우저에서 http://localhost:8000 접속
```

## CLI 명령어

### run - 멀티스트림 처리 실행

```bash
# 설정 파일로 실행
python main.py run --config config.json

# CLI 옵션으로 실행
python main.py run \
    --streams "video1.mp4,video2.mp4,rtsp://camera1" \
    --outputs "out1.mp4,out2.mp4," \
    --method bytetrack \
    --task-fight \
    --task-inert \
    --half \
    --batch-size 6

# ReID 활성화
python main.py run \
    --streams "video1.mp4,video2.mp4" \
    --use-reid \
    --reid-method adaptive \
    --reid-threshold 0.5
```

**주요 옵션:**

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--config` | - | JSON 설정 파일 경로 |
| `--streams` | - | 쉼표 구분 입력 소스 (파일, RTSP, 웹캠 인덱스) |
| `--outputs` | - | 쉼표 구분 출력 파일 경로 |
| `--model` | `../weights/best.pt` | YOLO 모델 경로 |
| `--method` | `bytetrack` | 추적 알고리즘 (`bytetrack`, `botsort`, `deepsort`) |
| `--max-streams` | `6` | 최대 스트림 수 |
| `--target-fps` | `30` | 스트림별 목표 FPS |
| `--task-fight` | `True` | 싸움 감지 활성화 |
| `--task-inert` | `True` | 정지 감지 활성화 |
| `--task-escape` | `False` | 이탈 감지 활성화 |
| `--half` | `True` | FP16 반정밀도 추론 |
| `--batch-size` | `6` | 배치 추론 크기 |
| `--use-reid` | `False` | ReID 기반 ID 보정 |
| `--reid-method` | `adaptive` | ReID 방식 (`adaptive`, `histogram`, `mobilenet`) |
| `--reid-threshold` | `0.5` | ReID 유사도 임계값 |
| `--reid-global-id` | `False` | 스트림 간 글로벌 ID 할당 |

### create-config - 설정 파일 생성

```bash
# 4개 스트림 설정 생성
python main.py create-config --streams 4 --output my_config.json

# 템플릿 지정 (video, rtsp, webcam, mixed)
python main.py create-config --streams 6 --template rtsp --output rtsp_config.json
```

### benchmark - 성능 벤치마크

```bash
python main.py benchmark --config config.json --duration 60 --warmup 10
```

벤치마크 결과: 평균/최대/최소 FPS, 스트림 수, 효율(%) 출력.

### info - 시스템 정보

```bash
python main.py info
```

Python, PyTorch, CUDA 버전 및 GPU 정보 출력. A100 감지 시 최적화 권장사항 표시.

### list-streams - 설정된 스트림 목록

```bash
python main.py list-streams --config config.json
```

## 웹 뷰어 (실시간 모니터링)

`web_server.py`로 브라우저에서 전체 스트림을 실시간 모니터링할 수 있습니다.

```bash
python web_server.py --config sample_config.json --port 8000 --host 0.0.0.0
```

### 웹 UI 기능

- **멀티스트림 그리드 뷰**: WebSocket을 통한 실시간 JPEG 프레임 스트리밍 (~15fps)
- **시스템 리소스 모니터링**: CPU, GPU 사용률, GPU 메모리, 온도 실시간 표시
- **스트림별 상태**: FPS, 지연시간, 추적 객체 수, 행동 감지 카운트
- **동적 스트림 관리**: 웹 UI에서 스트림 추가/삭제 가능

### REST API

| Method | Endpoint | 설명 |
|--------|----------|------|
| `GET` | `/` | 웹 뷰어 페이지 |
| `GET` | `/api/stats` | 시스템 및 스트림 통계 |
| `GET` | `/api/streams` | 활성 스트림 목록 |
| `POST` | `/api/streams` | 스트림 동적 추가 |
| `DELETE` | `/api/streams/{stream_id}` | 스트림 동적 삭제 |

### WebSocket

| Endpoint | 설명 |
|----------|------|
| `/ws/frames/{stream_id}` | 특정 스트림의 JPEG 프레임 |
| `/ws/frames/all` | 전체 스트림 프레임 (stream_id 태그 포함) |

## 설정 파일 구조

```json
{
  "model_path": "../weights/modelv11x.pt",
  "gpu": {
    "device_id": 0,
    "memory_fraction": 0.9,
    "half_precision": true,
    "batch_size": 6,
    "enable_cudnn_benchmark": true,
    "enable_tf32": true
  },
  "processing": {
    "max_streams": 8,
    "target_resolution": [640, 640],
    "frame_buffer_size": 60,
    "inference_timeout": 0.033,
    "enable_adaptive_skip": true,
    "max_frame_skip": 2,
    "min_fps_threshold": 25
  },
  "streams": [
    {
      "stream_id": "camera_1",
      "input_source": "video.mp4",
      "output_path": "output/result.mp4",
      "method": "bytetrack",
      "task_fight": true,
      "task_escape": false,
      "task_inert": true,
      "threshold": 0.1,
      "inert_threshold": 50.0,
      "inert_frames": 100,
      "priority": 1,
      "target_fps": 30,
      "use_reid": true,
      "reid_method": "adaptive",
      "reid_threshold": 0.5,
      "reid_global_id": false
    }
  ],
  "log_level": "INFO",
  "stats_interval": 5.0
}
```

### 스트림 설정 필드

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `stream_id` | string | (필수) | 스트림 고유 식별자 |
| `input_source` | string | (필수) | 입력 소스 (파일 경로, RTSP URL, 웹캠 인덱스) |
| `output_path` | string | `null` | 출력 파일 경로 (없으면 저장 안 함) |
| `method` | string | `bytetrack` | 추적 알고리즘 |
| `task_fight` | bool | `true` | 싸움 감지 |
| `task_escape` | bool | `false` | 이탈 감지 |
| `task_inert` | bool | `true` | 정지 감지 |
| `threshold` | float | `0.1` | 싸움 감지 임계값 |
| `inert_threshold` | float | `50.0` | 정지 판정 변위 임계값 |
| `inert_frames` | int | `100` | 정지 판정 프레임 수 |
| `priority` | int | `1` | 처리 우선순위 (1=최고, 3=최저) |
| `target_fps` | int | `30` | 목표 FPS |
| `use_reid` | bool | `false` | ReID ID 보정 활성화 |
| `reid_method` | string | `adaptive` | ReID 방식 |
| `reid_threshold` | float | `0.5` | ReID 유사도 임계값 |
| `reid_global_id` | bool | `false` | 스트림 간 글로벌 ID |

## 주요 기능

### GPU 최적화 (A100)
- TF32 연산 활성화
- FP16 (Half Precision) 추론
- cuDNN 벤치마크 모드
- 배치 추론 (최대 6 프레임 동시 처리)
- GPU 메모리 90% 활용 설정

### 멀티스트림 처리
- 비동기 프레임 캡처
- 배치 모델 추론
- 스트림별 독립 상태 관리
- 우선순위 기반 처리
- 33ms 타임아웃으로 30fps 실시간 보장

### 적응형 프레임 스킵
- 시스템 부하에 따른 자동 조절
- 최소 25 FPS 보장 (30fps 타겟)
- 최대 2프레임 스킵으로 실시간성 유지

### 행동 감지
- **Fight**: 반려동물 간 싸움 감지 (IoU 기반)
- **Inert**: 장시간 정지 상태 감지 (변위 기반)
- **Escape**: 지정 영역 이탈 감지 (폴리곤 기반)

### ReID (Re-Identification)
- **adaptive**: 적응형 특징 추출
- **histogram**: 히스토그램 기반 외형 매칭
- **mobilenet**: MobileNet 기반 딥러닝 특징 추출
- 스트림 간 글로벌 ID 할당 지원

### 리소스 모니터링
- CPU/메모리 사용률 (psutil)
- GPU 사용률, 메모리, 온도 (pynvml)
- 스트림별 FPS, 지연시간, 처리/드롭 프레임 수
- 시스템 과부하 자동 감지

## 성능 예상치 (A100 40GB)

| 스트림 수 | 해상도 | FPS/스트림 | GPU 사용률 | 실시간 보장 |
|-----------|--------|------------|------------|-------------|
| 2 | 640p | 30 | ~20% | O |
| 4 | 640p | 30 | ~40% | O |
| 6 | 640p | 30 | ~60% | O |

6스트림 30fps가 실시간 처리 보장 최대치입니다.

## 파일 구조

```
PoC/
├── main.py               # CLI 진입점 (run, create-config, benchmark, info, list-streams)
├── stream_processor.py   # 핵심 멀티스트림 처리기 (MultiStreamProcessor)
├── config.py             # 설정 관리 (SystemConfig, StreamConfig, GPUConfig, ProcessingConfig)
├── monitor.py            # 리소스 모니터링, 성능 프로파일링, 통계 집계
├── web_server.py         # FastAPI 웹 서버 (REST API + WebSocket 프레임 스트리밍)
├── templates/
│   └── index.html        # 웹 모니터링 UI (멀티스트림 그리드 뷰)
├── sample_config.json    # 6스트림 샘플 설정 파일
├── output/               # 출력 영상 저장 디렉토리
└── README.md
```

## 의존성

상위 폴더의 `requirements.in` 참조. 추가 권장:

```bash
pip install pynvml   # GPU 모니터링용 (선택)
pip install psutil   # CPU/메모리 모니터링용
pip install fastapi  # 웹 서버
pip install uvicorn  # ASGI 서버
```
