# YOLO → NPU-friendly INT8 TFLite 변환 가이드

## 변환 파이프라인

```
YOLO (.pt)
  └─▶ ONNX (static shape, NMS 제외)
        └─▶ TF SavedModel (onnx2tf, NCHW→NHWC)
              └─▶ TFLite INT8 (full quantization)
```

---

## Step 1. YOLO → ONNX

### 설정값

| 옵션 | 값 | 이유 |
|---|---|---|
| `imgsz` | 640 | 입력 shape 고정 |
| `dynamic` | False | static shape — NPU는 dynamic shape 미지원 |
| `simplify` | True | onnxslim으로 불필요한 노드 제거 |
| `opset` | 17 | TF/TFLite 변환 호환 opset |
| `nms` | False | NMS 제외 (후처리는 앱 코드에서 처리) |

### NMS 분리 이유
NMS는 `NonMaxSuppression`, `TopK`, `GatherND` 등 **dynamic output op**를 포함한다.
NPU는 출력 shape이 입력값에 따라 변하는 op를 처리하지 못해 CPU fallback이 발생한다.
전처리(resize/letterbox/normalize)도 동일한 이유로 앱 코드에서 처리한다.

### 주의
ultralytics 일부 버전에서 `nms=False`로 export해도 `TopK`가 헤드 후처리에 잔존할 수 있다.
이 경우 백본/넥의 Conv 블록은 NPU에 올라가고 헤드 끝단만 CPU fallback된다.
YOLO 추론 시간의 90% 이상이 백본/넥이므로 실질적 성능 영향은 작다.

---

## Step 2. ONNX → TF SavedModel (onnx2tf)

### 설정값

| 옵션 | 값 | 이유 |
|---|---|---|
| `-ois` | `images:1,3,640,640` | ONNX 원본 NCHW shape 명시 — 미지정 시 shape 추론 오류 |
| `-n` | (non_verbose) | 로그 억제 |

### NCHW vs NHWC
- ONNX/PyTorch: NCHW `(batch, channel, height, width)`
- TensorFlow/TFLite: NHWC `(batch, height, width, channel)`
- onnx2tf가 자동 변환하지만 `-ois`에는 **ONNX 원본 NCHW shape**을 입력해야 한다.
  NHWC `1,640,640,3`으로 넘기면 channel dimension 불일치 오류 발생.

---

## Step 3. TF SavedModel → TFLite INT8

### 설정값

| 설정 | 값 | 이유 |
|---|---|---|
| `optimizations` | `[Optimize.DEFAULT]` | full INT8 quantization 활성화 |
| `representative_dataset` | val set 420장 | 실제 activation 분포 calibration |
| `supported_ops` | `TFLITE_BUILTINS_INT8` | SELECT_TF_OPS 완전 배제 |
| `inference_input_type` | `tf.uint8` | NPU는 uint8 입력 선호, 앱에서 uint8 이미지 직접 전달 가능 |
| `inference_output_type` | `tf.float32` | box 좌표/score 정밀도 유지 (uint8로 하면 좌표 손실) |
| `allow_custom_ops` | False | custom op 명시적 금지 — NPU delegate 불가 op 차단 |
| `experimental_new_converter` | True | 최신 MLIR 기반 변환기 사용 |

### SELECT_TF_OPS 제거 이유
`SELECT_TF_OPS`를 허용하면 TFLite에 없는 op를 TF 런타임으로 fallback 처리한다.
TF 런타임 op는 NPU delegate가 인식하지 못해 전부 CPU에서 실행된다.
NPU delegate coverage를 최대화하려면 반드시 제거해야 한다.

### Representative Dataset 구성 기준
- **출처**: val set (학습에 직접 사용되지 않은 동일 도메인 데이터)
- **수량**: 200장 이상 권장, 이 프로젝트는 420장 전체 사용
- **구성 비율**: 펫 있는 화면 70% + 빈 화면(배경만) 30%
  - 빈 화면 미포함 시 배경 영역 activation이 calibration에서 누락되어 false positive 증가
- **augmentation**: 밝기 ±20%, 좌우 반전 (원본 이미지 수가 적을 때 다양성 확보)

---

## Android 배포 시 주의사항 (2026 기준)

- **NNAPI deprecated**: Android 15에서 NNAPI가 deprecated됨
- **권장 경로**: LiteRT + vendor-specific NPU delegate
  - Qualcomm: Qualcomm AI Engine Direct Delegate
  - 기타 SoC: 해당 벤더 LiteRT delegate
- 같은 TFLite 모델이라도 SoC별로 delegate coverage가 다를 수 있음
- 최적화 완료 기준: 타겟 SoC에서 실제로 몇 개 op가 delegate로 올라가는지 확인 필요

---

## 사용 명령어

```bash
cd /Users/gentiqxpc1/Documents/Garin/code/PoC
source npu/bin/activate

python -m models.cli mobile npu-int8 \
  --model /Users/gentiqxpc1/Documents/Garin/code/weights/best_tunning.pt \
  --data /Users/gentiqxpc1/Documents/Garin/code/calibration/train/images \
  --output ../weights/npu_int8_best_tunning.tflite \
  --samples 420
```

## 환경

- Python: 3.11.15
- tensorflow-macos: 2.16.2
- tensorflow-metal: 1.1.0
- onnx: 1.16.2
- onnx2tf: 1.26.3
- tf_keras: 2.16.0
- ml_dtypes: 0.3.2
