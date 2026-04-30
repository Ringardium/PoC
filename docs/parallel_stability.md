# 병렬 스트림 안정화 플레이북

GPU 서버에서 여러 카메라 스트림을 동시에 처리할 때 FPS 덜컥임, VRAM 증가, 지연 축적 등을 줄이기 위한 개선 항목을 ROI(투자 대비 효과) 기준으로 정리한 문서.

## 현재 병렬 처리 구조

```
MultiStreamProcessor
├── ThreadPoolExecutor(max_workers = max(len(streams)*3, 8))
├── _gpu_sem = threading.Semaphore(2)   ← GPU 추론 동시 실행 제한
├── 스트림별 async task (_stream_loop)
├── 스트림별 YOLO 인스턴스 (tracker state 격리용)
└── 전역 asyncio loop
```

### 알려진 병목
1. **GPU 직렬화**: `_gpu_sem(2)`로 2개 스트림만 동시 YOLO 추론. 스트림 4개 이상이면 큐 쌓임.
2. **VRAM 선형 증가**: 각 스트림이 자체 YOLO 인스턴스 유지 → N × 모델 크기.
3. **ReID 오버헤드**: DINOv2 feature extraction이 모든 YOLO 프레임마다 돎 (CPU 부담).
4. **단일 asyncio 루프**: 스트림 수 많아지면 selector-based asyncio가 병목 가능.
5. **배치 추론 미사용**: 매 프레임 단건 inference.
6. **bathroom classifier cold start**: 첫 트리거 시 모델 로드 1~2초 stall (✅ 해결됨 — startup eager-load).

## 개선 항목 (ROI 순)

### 🔝 즉효 (수 시간 작업)

#### 1. GPU Semaphore 설정화
- 현재 `Semaphore(2)` 하드코딩 → `GPUConfig.inference_concurrency` 필드로 노출
- A100 기준 4~6까지 안전. 고급 GPU면 배치로 가는 게 낫지만 임시 튜닝 knob.

#### 2. uvloop 채택
- asyncio 기본 selector-loop → `uvloop`로 교체 (리눅스만)
- 이벤트 루프 처리량 2~3배, 코드 변경 3줄
- macOS나 uvloop 미설치 환경에선 자동으로 기본 asyncio로 fallback

### 🥈 중간 (반나절 작업)

#### 3. ~~YOLO 모델 공유~~ ❌ 채택 불가
- ultralytics `model.track(persist=True)`는 **tracker state(Kalman, ID pool, frame_cnt)를
  model.predictor 내부에 저장** — 인스턴스 공유 시 스트림 간 ID 충돌 / Kalman 오염 / ID 스위치 폭증
- per-stream YOLO 인스턴스는 설계상 올바름. **유지**
- VRAM 절감 원하면 5번(배치) 경로로 가는 게 맞음 (tracker는 detection 결과를 stream-local Python wrapper로 처리하는 구조 변경 필요)

#### 3'. ReID 빈도 조절  ← 3번 대체
- 현재: 매 YOLO 프레임마다 DINOv2 feature extraction
- 개선: 매 N프레임(예: 3~5) 또는 ID 혼동(proximity_locked 많을 때) 시에만 재추출
- CPU 부담 대폭 감소, global_id 안정성은 약간 느려질 수 있음
- per-stream tracker 영향 없음

#### 4. bathroom classifier eager-load
- ✅ 적용 완료 (Phase 0)

### 🥉 큰 작업 (며칠)

#### 5. 배치 추론 (detection만)
- N개 스트림의 프레임을 한 번에 `model.predict([f1, f2, ...])` — **detection만**
- 이후 각 스트림의 자체 tracker(ByteTrack/BoT-SORT wrapper)에 detection 결과 주입
  → tracker state는 여전히 per-stream, 단 YOLO inference는 공유 batch
- 장점: 처리량 2~3배 (A100), VRAM 1× (모델 1개만)
- 필요 작업:
  - ultralytics `model.track()` 의존 제거 → `model.predict()` + standalone tracker
  - `supervision.ByteTrack` 같은 lib 사용 또는 커스텀 wrapper
  - 프레임 수집 큐 + 배치 윈도우 + 결과 재분배
- tracker state 격리 여전히 성립

#### 6. TensorRT Dynamic Batch
- 현재 `best.engine`이 고정 batch=1이면 5번 불가능
- `trtexec --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640` 같은 형식으로 재빌드
- 5번의 전제 조건

#### 7. Multiprocessing per Stream
- 스트림별 독립 프로세스 → GIL / asyncio 간섭 제로
- 각 프로세스가 자체 CUDA context 가짐 → VRAM 중복 (MPS 쓰면 완화 가능)
- 내결함성 최강: 한 스트림 크래시해도 다른 스트림 무영향
- 복잡도 크고 state 공유 어려움

## 실행 계획

- [x] Phase 0: bathroom classifier eager-load
- [x] **Phase 1**: GPU semaphore 설정화 (1번)
- [x] **Phase 2**: uvloop 채택 (2번)
- [x] **Phase 3**: ReID 빈도 조절 (3'번) — `StreamConfig.reid_every_n_frames` (default 1, 권장 3-5)
- [x] **Phase 4**: 배치 detection + standalone tracker (5+6번) — opt-in via `SystemConfig.batched_detection_enabled`
- [ ] Phase 5 (조건부): multiprocessing 격리 (7번)

**제외**: YOLO 모델 공유 (legacy `model.track(persist=True)` 경로) — ultralytics tracker state가 model 내부에 있어서 불가능.
공유는 Phase 4 경로(detection만 batch + per-stream standalone tracker)로 해결됨.

Phase 1~2는 동작 변경 최소, 즉시 적용 가능. Phase 3부터는 감지 로직에 영향 — 테스트 필수.

---

## Phase 4 사용 가이드

### 동작 원리

```
[stream A]──┐
[stream B]──┼─→ BatchedDetector (asyncio coalesce + run_in_executor)
[stream C]──┤        ↓
[stream D]──┘   shared YOLO.predict([f1, f2, f3, f4])  ← single CUDA launch
                     ↓
              per-stream filter (classes, conf)
                     ↓
[stream A] ← StandaloneTracker(byterack/botsort) per stream
[stream B] ← StandaloneTracker
[stream C] ← StandaloneTracker
[stream D] ← StandaloneTracker
```

- 한 번의 `YOLO.predict([f1..fN])` 호출로 N 스트림의 detection을 동시에 처리 → CUDA launch 오버헤드 N배 절감
- 각 스트림은 자기만의 ByteTrack/BoT-SORT 인스턴스 보유 → ID 격리 유지
- 이후 ReID / 행동 감지 / 메타데이터 송출 흐름은 legacy 경로와 100% 동일

### 활성화

`config.json` 의 SystemConfig 레벨에 추가:
```json
{
  "model_path": "../weights/modelv11x.pt",
  "batched_detection_enabled": true,
  "batched_detection_max_batch": 4,
  "batched_detection_wait_ms": 5.0,
  "gpu": { "inference_concurrency": 1, ... },
  "streams": [ ... ]
}
```

| 필드 | 기본값 | 비고 |
|---|---|---|
| `batched_detection_enabled` | `false` | true 면 Phase 4 경로, false면 legacy `model.track()` per-stream |
| `batched_detection_max_batch` | `4` | predict() 한 번당 최대 batch. A100 기준 4-8 권장 |
| `batched_detection_wait_ms` | `5.0` | 첫 요청 도착 후 추가 요청 합치려고 기다리는 시간 (ms). 너무 작으면 batch 못 모음, 너무 크면 latency ↑ |

> **`inference_concurrency` 는 batched 모드에서 의미가 작아짐**. BatchedDetector 내부 `_infer_lock`이 GPU 직렬화를 처리하므로, 1로 두는 게 깔끔. legacy 모드에선 그대로 4-6 권장.

### `model_path` 호환성

| 모델 포맷 | Phase 4 호환성 |
|---|---|
| `.pt` (PyTorch) | ✅ 그대로 동작 (PyTorch는 dynamic batch 자유) |
| `.engine` (TensorRT) | ⚠️ **반드시 dynamic 모드로 재빌드** 필요. 고정 batch=1 engine은 batch 못 받음 |
| `.onnx` | ⚠️ dynamic axis (batch 차원) 명시해서 재export 필요 |

TensorRT 재빌드 예:
```bash
yolo export model=weights/best.pt format=engine dynamic=True batch=8 half=True imgsz=640
# 또는
trtexec --onnx=best.onnx \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:4x3x640x640 \
  --maxShapes=images:8x3x640x640 \
  --fp16 --saveEngine=best.engine
```

### 동작 확인 포인트

활성화 후 첫 실행 시 로그에서 확인:
```
INFO  Phase 4 batched detection ENABLED — shared model + standalone trackers
INFO  BatchedDetector started — device=cuda:0, half=True, max_batch=4, batch_wait=5.0ms
```

`/health` 엔드포인트에 batched stats 추가 안 됐다면, `batched_detector.get_stats()` 가 `{batches, frames, avg_batch, avg_infer_ms}` 반환하니 모니터링 hook에 붙일 수 있음.

### 알려진 trade-off

1. **NMS IoU 공유**: batched predict는 한 번의 NMS를 모든 frame에 동일 IoU로 적용. `BatchedDetector` 가 스트림들의 `yolo_iou` median을 골라 사용 → per-stream IoU 다양성은 희생됨 (대부분의 경우 0.5로 통일되어 있어 문제 없음)
2. **Conf threshold 공유**: 가장 작은 conf로 predict 후 per-stream에서 다시 필터 → 정확도 동일, 불필요 NMS 약간 발생
3. **첫 프레임 latency**: `batch_wait` 만큼의 합치기 대기 → max 5ms 추가. 거의 무시 가능
4. **동적 stream 추가/제거**: `add_stream_dynamic()` 호출 시 자동으로 StandaloneTracker만 새로 만듦. BatchedDetector 자체는 무관 (batch 차원이 자유로워서)

### 롤백

`batched_detection_enabled: false` 로 변경 후 재시작. legacy 경로 (`model.track(persist=True)` per-stream YOLO 인스턴스) 그대로 동작. **두 경로는 코드 상에서 완전 분기**되어 있어 안전한 토글.

### 새 코드 위치

| 파일 | 역할 |
|---|---|
| [PoC/standalone_tracker.py](../PoC/standalone_tracker.py) | ultralytics BYTETracker / BOTSORT를 외부 detection으로 구동하는 wrapper |
| [PoC/batched_detector.py](../PoC/batched_detector.py) | shared YOLO + asyncio batching coordinator |
| [PoC/stream_processor.py](../PoC/stream_processor.py) `_run_tracking_batched` | Phase 4 trackign 경로 (legacy `_run_tracking` 와 분기) |
| [PoC/config.py](../PoC/config.py) `SystemConfig.batched_detection_*` | 활성화 플래그 3개 |

## 참고 수치 (예상)

| 변경 | 스트림 4개 목표 FPS | VRAM | CPU |
|---|---|---|---|
| 현재 (legacy) | 15×2 = 30 효과적 (나머지 대기) | N × 112MB | 80% |
| +semaphore(4) | 15×4 = 60 | 동일 | 90% |
| +uvloop | +10% | 동일 | 85% |
| +ReID N프레임 (Phase 3) | 동일 | 동일 | 40~60% |
| +batched detect (Phase 4) | 15×4 (지연↓) | **1 × 112MB** | 70% |

**Phase 1-4 누적 적용 시 권장 config 스타일**:
```json
{
  "batched_detection_enabled": true,
  "batched_detection_max_batch": 4,
  "gpu": { "inference_concurrency": 1 },
  "streams": [
    { "..." : "...",
      "adaptive_fps_enabled": true,
      "adaptive_fps_idle": 1.0,
      "reid_every_n_frames": 5
    }
  ]
}
```
sample_config.json / whep_test_config.json 에 reid_every_n_frames 와 adaptive_fps 는 이미 적용됨. batched 는 검증 후 켜는 게 안전.
