# YOLO 검출기 모바일 NPU(INT8) 배포 — 디버깅 타임라인

> 펫 모니터링용 YOLO 검출기를 Qualcomm Hexagon NPU에 INT8로 올리고, 성능·정확도·
> 기기 충실도(parity)를 검증·개선한 전 과정의 기술 기록. 각 단계는
> **문제(Problem) → 원인(Cause) → 시도(Attempt) → 결과/해결(Result)**, 그리고 **교훈**으로 구성.
> 포트폴리오/면접용. 모든 수치는 실측값.

---

## 0. 개요

| | |
|---|---|
| **목표** | FP32 YOLO 검출기를 모바일 NPU에서 INT8로 실시간 추론 + 정확도·충실도 검증 |
| **모델** | YOLO 3-class (person/dog/cat), `best_tunning.pt`, FP32 mAP50 = **0.872** |
| **타겟 기기** | Galaxy A36 5G — Snapdragon **6 Gen 3**(SM6475, 중급 SoC), Hexagon HTP **v73** |
| **OS/런타임** | Android 16(SDK36), arm64 · QAIRT(QNN) SDK 2.44.0 · LiteRT `benchmark_model`/`qtld-net-run` |
| **평가셋** | Roboflow 420장 (GT: person 543 / dog 393 / cat 96 인스턴스) |

**최종 결과 한 줄 요약:**
INT8 `split.tflite`를 QNN HTP burst로 **392/392 op 100% NPU 적재, 16.2ms(62fps)** 달성.
정확도 mAP50 0.872→**0.73**(평가셋 한계 보정 전), 그리고 **세 건의 숨은 결함을 디버깅으로 규명**:
① INT8 concat의 cls 말살, ② 평가 코드의 box 포맷 오디코딩, ③ cls 출력 양자화 saturation.

---

## 타임라인

### T1. 어떤 그래프를 NPU에 올릴 것인가 — concat의 INT8 정밀도 파괴

**문제:** INT8 변환 산출물이 3종(`best_tunning`/`raw`/`split`). 무엇을 NPU에 써야 하는가?

**원인 분석:** 데스크톱 TFLite로 출력·op 분석.

| 파일 | 출력 | op | 정체 |
|---|---|---|---|
| `best_tunning.tflite` | `[1,300,6]` | 451 | NMS 박힌 end-to-end |
| `raw.tflite` | `[1,7,8400]` | 404 | NMS 제거 (box4+cls3 **concat**) |
| `split.tflite` | `[1,4,8400]`+`[1,3,8400]` | 401 | NMS 제거 **+ concat 제거** |

- NMS의 `TopK`/`NonMaxSuppression`은 HTP가 못 먹어 CPU 폴백 유발 → end-to-end 부적합.
- **핵심 발견:** INT8 `Concat`은 입력이 **하나의 (scale,zero_point)를 공유**해야 한다.
  box(픽셀 0~684)와 cls(0~1)를 합치면 공유 스케일이 box에 끌려가 step≈684/255≈3.18 →
  cls(최대 0.5)는 반 칸 미만이라 **전부 0으로 반올림**. 실측: `raw.tflite`의 cls range
  = **[0, 0]** (완전 사망). `split.tflite`는 헤드 분리로 cls range [0,0.5], step 1/256 보존.

**해결:** `split.tflite` 채택(이후 전 실험 기준). `raw`는 cls가 죽어 사용 불가.

**교훈:** 멀티헤드 모델 PTQ에서 **dynamic range가 다른 텐서를 한 텐서로 concat하면 작은 쪽이
양자화로 말살**된다. NMS뿐 아니라 출력 그래프 형태가 INT8 정밀도를 좌우.

---

### T2. NPU 접근 경로 확보 — NNAPI는 죽었다

**문제:** NPU에 어떻게 접근? 다운로드 없는 표준 경로(NNAPI)가 되는가?

**시도/결과:**
- `--use_nnapi=true` → `NNAPI accelerators: [nnapi-reference]` (CPU만). 그래프 미델리게이트, 214ms.
- `--use_gpu=true` → `Batch size mismatch` 로 GPU 델리게이트 적용 실패.

**원인:** Android 15+에서 NNAPI deprecated, Qualcomm이 NNAPI HAL을 제거하고 **QNN으로 이전**.
Android 16 기기엔 벤더 NPU 드라이버가 NNAPI에 노출 안 됨. (`/vendor/lib64`의
`libSnpeHtpV73Stub.so`로 HTP arch가 **v73**임을 역으로 확인.)

**해결:** QAIRT SDK의 `libQnnTFLiteDelegate.so` + v73 stub/skel을 기기에 push,
`--external_delegate_path` 로 QNN HTP 델리게이트 적용.

**교훈:** 최신 Qualcomm 기기에서 NPU = **QNN 외길**. 기기 vendor 라이브러리가 HTP arch의
1차 사료.

---

### T3. NPU 성능 — 100% 적재했는데 왜 안 빠른가

**문제:** QNN HTP 적용 시 `392/392 node delegated`(100% NPU)인데 추론이 **52ms** — CPU
4-thread(77ms) 대비 1.5배뿐.

**원인 가설:** 모델이 아니라 **델리게이트 전력 정책**이 throttle.

**시도:** `htp_performance_mode` 조정. (함정: 이 빌드는 **문자열이 아닌 정수 enum**을 받음 —
`kHtpBurst=2`. `:burst` 문자열은 `stoi` 크래시. 헤더 `QnnTFLiteDelegate.h`로 확인.)

**결과/해결:** burst(=2) → **16.2ms (≈62fps)**. 기본 대비 3.2×, CPU 4t 대비 **4.7×**, CPU
1t(214ms) 대비 13×. 추가로 첫 로드 **135초**(HTP 그래프 컴파일)는 `cache_dir`로 컨텍스트
바이너리(12.3MB) 캐싱 → init **0.34초**.

| 모드 | 추론 | fps |
|---|---|---|
| CPU 1t / 4t | 214 / 77 ms | 4.7 / 13 |
| NPU default / **burst** | 52 / **16.2** ms | 19 / **62** |

**교훈:** "NPU 느림"의 흔한 원인은 모델이 아니라 perf 옵션. 단 burst는 최대 전력 →
thermal throttle 가능(지속 부하는 `sustained_high_performance`로 별도 측정 필요, 미완).

---

### T4. 정확도 평가 — "mAP 0.0012, 모델 사망?" → 평가 코드 버그

**문제:** `benchmark.py`로 측정한 INT8 mAP50 = **0.0012** (FP32 0.872 대비 사실상 0).
도구는 "재calibration 필요" 판정.

**원인 규명:** 모델 사망을 의심하기 전에 **측정 자체를 검증**. split의 box 출력을 GT와 직접
대조 → top 검출이 `IoU(as-xyxy)=0.90` vs `IoU(as-xywh)=0.15`. 즉 **box 헤드는 이미 xyxy를
출력**하는데 `benchmark.py`가 **xywh로 가정하고 변환**(`_raw_to_dets`)해 박스를 전부 파괴.
→ **모델이 아니라 평가 코드 버그.**

**해결:** `_raw_to_dets`에 box 포맷 **자동 감지** 추가(고신뢰 박스에서 x2>x1·y2>y1 비율>0.9면
xyxy). 수정 후 전체 420장 재측정:

| | FP32 | INT8 split | drop |
|---|---|---|---|
| **mAP50** | 0.8716 | **0.7316** | -0.140 |
| person | 0.726 | 0.448 | -0.278 |
| dog | 0.914 | 0.810 | -0.104 |
| cat | 0.974 | 0.937 | -0.038 |

**교훈:** 0 mAP는 모델 사망이 아니라 **eval 디코딩 버그**였다. 양자화 평가는 **디코딩
정합성부터** 검증해야 한다. (평가셋은 train=val=test 동일 + person 라벨 노이즈로 절대값은
보수적 해석 — 아래 T5.)

---

### T5. person이 유독 많이 떨어진다 — 가설 검증

**문제:** drop의 대부분이 person(-0.278). 양자화에 person이 민감한가?

**가설 A (데이터 불균형):** **기각.** GT 카운트 결과 person이 **543개로 최다**(dog 393, cat 96).
데이터 양 문제 아님.

**가설 B (GT 불완전, 사용자 제보):** **지지.** FP32가 conf≥0.4로 잡은 검출 중 GT 미매칭 비율:

| 클래스 | GT매칭 | 무라벨 | 무라벨% |
|---|---|---|---|
| **person** | 313 | 59 | **15.9%** |
| dog | 311 | 22 | 6.6% |
| cat | 90 | 3 | 3.2% |

person 무라벨률이 dog의 2.4×, cat의 5×.

**원인:** 라벨 안 된 실제 사람을 모델이 검출 → **FP로 카운트 → person AP가 FP32·INT8 모두
인위적 하락**. 따라서 person의 큰 drop은 **metric 오염**이 섞인 수치. 라벨 품질이 나은
**dog(-0.10)/cat(-0.04)가 진짜 양자화 손실에 더 가까운 추정.**

**교훈:** "어떤 클래스가 양자화에 약하다"를 주장하기 전 **GT 품질을 의심**하라. 클래스 격차는
양자화 민감도가 아니라 라벨 노이즈일 수 있다. (가설을 데이터로 기각/지지한 과정 자체가 핵심.)

---

### T6. 기기 충실도(parity) — 데스크톱 mAP가 폰 NPU에서 재현되는가

**문제:** 지금까지 정확도(0.73)는 전부 **데스크톱 CPU tflite** 숫자. 폰 HTP가 같은 출력을
내는가?

**방법:** `qtld-net-run --backend htp`로 실이미지 5장의 출력 텐서를 덤프, 데스크톱 CPU tflite
출력과 비교. (출력 디렉토리 권한 깨짐·콜론 파일명 등 도구 이슈를 권한 777 + 사전생성으로 우회.)

**결과:**
- **배경 이미지(검출 없음): HTP = CPU 정확 일치** (cls 0.0234 등). 쉬운 곳은 완벽 parity.
- **검출 있는 이미지: 발산.** FP32를 진실로 두고 대조:

| | FP32(진실) | CPU tflite | HTP |
|---|---|---|---|
| img0 person | **0.888** | 0.50 (천장) | 0.863 ✅ |
| img4 dog | **0.825** | 0.50 (천장) | 0.137 ❌(놓침) |

**원인 (이중 결함):**
1. **출력 양자화 saturation (export 결함):** CPU cls 값이 `{0, 0.0039, 0.5}`뿐 — cls 출력
   int8 스케일이 1/256이라 **confidence가 ≈0.496에서 천장**. FP32는 0.88까지 가는데 INT8
   출력 범위가 [0,0.5]로 **under-calibration**됨(calibration 데이터에 고신뢰 객체가 부족해
   출력 range가 낮게 잡힘). → 강한 검출도 0.5로 깎임.
2. **백엔드 발산:** HTP는 천장을 부분적으로 벗어나 img0은 FP32(0.888)에 거의 일치(0.863)하나,
   img4는 자기 오차로 0.137(FP32 0.825) → conf 0.25 threshold에서 **검출 누락**.
- **반증된 가설:** `--htp_disable_conv_hmx`(HMX 근사 의심) 시도 → cls가 0.977로 더 망가짐
  (악화). HMX 원인 **아님**.

**해결 방향:** box localization은 충실(IoU>0.95)하나 **confidence는 신뢰 불가**. 근본 해결은
**export 단계에서 cls 출력 양자화 range를 [0,1]로 교정**(출력 텐서 quant 고정 또는
high-confidence 샘플로 출력 calibration 보강). 그 전엔 on-device confidence 절대값을 신뢰하지
말고 상대순위/박스 위주로 사용.

**교훈:** "데스크톱에서 됐다 ≠ 기기에서 똑같다." parity 검증이 **mAP로는 안 보이던 출력
saturation 결함**을 드러냈다. INT8은 가중치뿐 아니라 **출력 텐서 range 보정**이 정확도를 좌우.

---

### T7. cls saturation 수정 시도 — 로짓 클리핑 (결과: 부분 성공·반증, 진행중)

**문제:** T6에서 confidence가 5단계로 붕괴(cls 로짓 quant scale 1.83). 이걸 고치자.

**원인(확정):** sigmoid 직전 로짓 텐서 `/model.23/Concat_1_output_0`의 range가 **배경 극단
음수(-211)에 지배**되어 per-tensor scale이 폭발(1.83). 의미있는 로짓(max 2.07)이 칸을 거의
못 받음. (배경 로짓이 -211까지 가는 이유 = 부록 A.3.)

**시도:** raw ONNX의 cls sigmoid 입력에 **Clip(-8, 8) 삽입** → onnx2tf → INT8 재변환.
- 무손실 검증(onnxruntime): box maxdiff 0.0, cls maxdiff 0.000335 (극단 배경만 영향). 확정.
- onnx2tf 함정: `download_test_image_data()`가 pickled npy를 `np.load(allow_pickle=False)`로
  읽다 크래시 → cwd에 더미 `calibration_image_sample_data_20x128x128x3_float32.npy` 두고 우회.

**결과(이중적):**
- ✅ **로짓 quant scale 1.83 → 0.046 (40배 미세화)** 확정. range [-463,3.67] → [-8.0, 3.6].
  5단계 붕괴 자체는 해소.
- ❌ **그런데 end-to-end confidence는 오히려 악화.** 25장 top-detection conf 비교:

| | 평균 top-conf | 평균 \|FP32 차이\| |
|---|---|---|
| FP32 | 0.837 | - |
| split(기존) | 0.554 | 0.283 |
| clip(수정) | **0.312** | **0.525** |

**논리적 원인(가설 2개, 미분리):**
1. **혼동 변수:** clip은 326장(jpg) calibration, 기존 split은 420장(png 포함). 데이터가 달라
   공정 비교 아님. → 같은 saved_model을 동일 326장으로 재양자화(split_repro)해 격리 필요(진행중).
2. **더 깊은 가설:** 기존 split의 "높은 conf(0.55)"가 **거친 양자화가 위로 반올림한 인공물**
   (로짓을 1.83칸의 0.5/0.86으로 튕김)이고, clip은 진짜 INT8 로짓(낮음)을 미세히 노출한 것.
   즉 **상류(backbone/head) INT8 양자화가 로짓을 떨어뜨리는 게 진짜 범인**이고, 출력단 클리핑으론
   못 고침 → 진짜 해결은 **cls 헤드 per-channel 양자화 / fp16 헤드 / QAT**.

**교훈(중요):** "메커니즘을 맞게 짚어도(로짓 scale 40배 개선 확정) end-to-end 지표는 안 좋아질 수
있다." 단일 지점 출력 클리핑은 **필요했지만 불충분**. 또한 **실험에 혼동 변수(calibration 수)를
넣지 말 것** — 격리 실험으로 clip 효과를 분리해야 결론 가능. (이 항목은 **미완·진행중**.)

---

## 최종 성과 요약

| 축 | 결과 |
|---|---|
| **성능** | split.tflite, QNN HTP burst, 100% NPU, **16.2ms(62fps)** — 중급기에서 CPU 4t 대비 4.7× |
| **정확도** | INT8 mAP50 0.73(FP32 0.87). 신뢰가능 손실 추정 = dog/cat 4~10pt (person은 라벨노이즈로 과대) |
| **충실도** | 박스 충실(IoU>0.95), **confidence는 출력 saturation+백엔드 발산으로 신뢰 불가** |
| **디버깅 성과** | ① concat의 cls 말살 ② 평가 box 포맷 버그 ③ cls 출력 saturation — 3개 결함 규명 |

## 핵심 역량 어필 포인트 (면접용)

1. **가설→데이터 검증 루프:** person 불균형 가설을 카운트로 기각, under-labeling을 무라벨률로
   입증, HMX 가설을 재실험으로 반증. 추측이 아니라 **반증 가능한 실험**으로 결론.
2. **측정 자체를 의심:** "mAP 0=모델 사망"을 GT 대조로 **eval 버그**로 규명·수정.
3. **INT8 내부 원리로 현상 설명:** concat 공유 스케일, 출력 range under-calibration 등
   양자화 메커니즘으로 인과 규명(단순 "돌려봤다"가 아님).
4. **엔드투엔드 시스템 감각:** 델리게이트 선택·perf 옵션·컴파일 캐시·기기 vendor 라이브러리까지
   배포 전 과정 핸들링.

## 남은 개선 로드맵 (미실행 — 우선순위)

| 후보 | 이유 | 기대 |
|---|---|---|
| **cls 출력 range 재보정** | T6 saturation 결함(0.5 천장) | confidence 신뢰성 회복(최우선) |
| 평가셋 정비 | 순환 + person 라벨노이즈 | 진짜 양자화 손실 측정 |
| Per-channel 양자화 | per-tensor가 채널 분포 차이 못 담음 | box/cls 정밀도↑ |
| Calibration 보강 | 고신뢰 샘플 부족이 출력 range 왜곡 | T6 근본 완화 |
| Mixed precision / QAT | PTQ 한계 시 | drop 최소화(비용↑) |
| sustained 모드 + end-to-end fps | burst thermal·후처리 미포함 | 실배포 지속 fps |

> 각 후보 실행 시 본 문서에 **문제→원인→시도→결과→교훈** 형식으로 append.

---

# 부록 A. INT8 양자화 원리 — 이 프로젝트에서 직접 겪은 것 중심

> 위 타임라인의 결함들(T1 concat, T6 로짓)은 모두 **하나의 원리**에서 나온다.
> 이 부록은 그 원리와 실무 주의점을 정리한다. (면접에서 "왜 그런가"를 묻는 질문 대비)

## A.1 양자화의 기본 — 256칸이 고정 예산

INT8 = 8비트 = **2⁸ = 256개 정수값**(-128~127)만 저장 가능. 양자화는 실수를 이 256칸에 매핑:

```
실제값 ≈ scale × (정수 q − zero_point),   q ∈ [−128, 127]
scale = (범위 max − 범위 min) / 256        ← 한 칸의 크기 = 최소 분해능(step)
```

- **버킷(칸) 개수는 256으로 고정.** 바꿀 수 있는 건 그 256칸을 **어떤 범위에 펼치느냐**뿐.
- **분해능(step) = 범위 / 256.** 같은 256칸을 **좁은 범위**에 펼치면 촘촘(정밀↑), **넓은 범위**면 듬성(정밀↓).

**자(ruler) 비유:** 눈금 256개짜리 자. 213m 길이면 눈금 간격 0.83m(1m도 구분 못 함), 16m 길이면 0.0625m(6cm 구분). **눈금 수는 같은데 자 길이가 분해능을 결정.**

**핵심 통찰 — 분해능은 제로섬:** 256칸은 고정 예산이라, 양자화는 본질적으로
**"어느 값 범위에 정밀도를 쓸지 선택하는 일"**이다. 안 쓰는 범위에 칸을 낭비하면 쓰는 범위가 굶는다.

## A.2 outlier 문제 — 큰 범위가 작은 범위를 죽인다 (이 프로젝트의 반복 주제)

per-tensor 양자화는 **텐서 하나의 min/max**로 scale을 정한다. 그래서 **극단값(outlier) 하나가
scale을 폭발**시키면, 정작 의미있는 대다수 값이 칸을 거의 못 받아 뭉개진다. 본 프로젝트는 이걸 **두 번** 겪음:

| | T1: concat | T6: 로짓 |
|---|---|---|
| 큰 범위 | box 좌표 (0~684 px) | 배경 로짓 (~-211) |
| 죽은 작은 범위 | cls 점수 (0~1) → 전부 0 | 의미있는 로짓 (max 2.07) → sigmoid 5단계 붕괴 |
| 메커니즘 | concat이 둘에 **공유 스케일** 강제 | 한 텐서 내 outlier가 scale 독식 |
| 해결 | concat 제거(헤드 분리) | 로짓 클리핑([-8,8]) |

→ **"dynamic range가 크게 다른 값을 한 양자화 단위에 섞지 마라"**가 두 사건의 공통 교훈.

## A.3 왜 YOLO 로짓이 극단 음수(-211)가 되나

- 검출 헤드는 8400 anchor마다 클래스 **로짓**을 뱉고 sigmoid로 [0,1] confidence가 됨.
- **객체 없는 anchor**는 모델이 sigmoid≈0을 원함 → 로짓을 음수로 밀어냄. 로짓엔 **하한이 없어**(conv 출력 unbounded) 확신하는 배경은 -50, -100, -211까지 드리프트.
- 8400개 중 대다수가 배경 → "easy negative"들이 학습 중 계속 더 음수로 밀림.
- **-211은 "절대 객체 아님"의 극단 확신.** 확률로는 무의미(sigmoid(-211)=sigmoid(-20)=0)하나
  **raw 크기가 거대** → per-tensor scale을 폭발(step 1.83)시키는 범인.

## A.4 클리핑 — 안 쓰는 범위를 버리고 분해능을 몰아주기

- **sigmoid(±8) = 0.99966 / 0.00034.** 로짓 ±8 바깥은 confidence가 0/1로 평평 → **잘라도 무손실**
  (실측 cls maxdiff 0.000335). 모델 실제 max 로짓 2.07이라 진짜 검출은 안 잘림.
- **[-8,8] 선택 근거:** (a) sigmoid 포화로 무손실 보장 + (b) 안전 마진(미지 입력이 로짓 3~5 내도 OK).
- **효과:** 범위 213→16 (13배 축소) → step 1.83→0.0625 (13배 미세화) → 로짓 0 근처 confidence
  해상도 ~0.0156 → 표현 가능 confidence **5단계 → ~64단계**.
- 더 좁히면([-6,6] 등) 해상도 더↑, 단 고confidence 검출 클립 위험과 trade-off.

## A.5 실무 주의점 체크리스트 (PTQ 할 때 매번 본다)

1. **per-tensor vs per-channel:** 채널별 분포 차가 크면 per-channel이 outlier 영향을 가둠. 검출 헤드는 특히.
2. **출력 텐서 range도 본다:** 가중치만 보지 말 것. cls 출력처럼 **출력 quant가 saturation**되면
   confidence 천장이 생김(T6). dequant 직전 int8 텐서의 scale/zp를 확인.
3. **dynamic range가 다른 값을 한 양자화 단위에 섞지 마라:** concat(T1)·outlier 텐서(T6) 모두 같은 함정.
4. **calibration 데이터가 activation 분포를 대표하는가:** 빈 화면/고confidence 객체가 누락되면
   range가 왜곡됨. 단, **outlier(배경 극단 로짓)는 데이터로 못 고침 → 그래프 클리핑이 정답.**
5. **평가는 디코딩 정합성부터:** box xywh/xyxy, 출력 순서, NMS 유무. "0 mAP=모델 사망"이 아니라
   eval 버그인 경우가 많다(T4).
6. **기기 parity를 잰다:** 데스크톱 CPU tflite ≠ 폰 HTP. INT8 반올림이 백엔드별로 달라
   경계 검출이 뒤집힐 수 있음(T6). HMX 같은 가속 옵션은 정확도 trade-off 주의.
7. **그래프 형태가 NPU 적재율을 좌우:** NMS/concat/transpose는 델리게이트 폴백을 유발 →
   conv-only 출력(split)이 100% 적재에 유리(T1, T3).
8. **input/output dtype:** NPU는 uint8 입력 선호. 출력 float32는 좌표 정밀도엔 좋지만,
   내부 int8 텐서의 range가 진짜 정밀도를 결정함(출력 float이라고 안심 금지).
