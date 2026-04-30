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
- [ ] **Phase 4**: 배치 detection + standalone tracker (5+6번) — 진행 중
- [ ] Phase 5 (조건부): multiprocessing 격리 (7번)

**제외**: YOLO 모델 공유 — ultralytics tracker state가 model 내부에 있어서 불가능.
배치 필요하면 Phase 4 경로 (detection만 batch + standalone tracker) 가야 함.

Phase 1~2는 동작 변경 최소, 즉시 적용 가능. Phase 3부터는 감지 로직에 영향 — 테스트 필수.

## 참고 수치 (예상)

| 변경 | 스트림 4개 목표 FPS | VRAM | CPU |
|---|---|---|---|
| 현재 | 15×2 = 30 효과적 (나머지 대기) | N × 112MB | 80% |
| +semaphore(4) | 15×4 = 60 | 동일 | 90% |
| +uvloop | +10% | 동일 | 85% |
| +ReID N프레임 | 동일 | 동일 | 40~60% |
| +batched detect(4) | 15×4 (지연↓) | 1 × 112MB | 70% |
