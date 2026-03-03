# YOLO 학습 파이프라인

YOLO 모델의 Fine-tuning, 클래스 확장, Knowledge Distillation, 데이터셋 관리를 위한 통합 CLI 도구.

## 파이프라인 개요

```
┌─────────────────────────────────────────────────────────┐
│                    데이터 준비                            │
│  prepare → merge / gather → analyze → sample            │
└────────────────────────┬────────────────────────────────┘
                         │
            ┌────────────┼────────────────┐
            ▼            ▼                ▼
      ┌──────────┐ ┌──────────┐   ┌────────────┐
      │ finetune │ │  expand  │   │  distill   │
      │ 동일 클래스│ │ 클래스 추가│   │ 모델 경량화 │
      │ 정밀도 향상│ │ head 확장 │   │ teacher→   │
      │          │ │          │   │   student  │
      └────┬─────┘ └────┬─────┘   └─────┬──────┘
           │             │               │
           └─────────────┴───────────────┘
                         │
                    best.pt / last.pt
```

## 학습 모드

### 1. Fine-tune — 동일 클래스 정밀도 향상

기존 모델과 같은 클래스 구조에서 추가 데이터로 성능을 개선합니다.

```bash
# 기본 fine-tune
python -m models.train finetune --data data.yaml --model weights/modelv11x.pt

# Backbone 고정 (전이학습)
python -m models.train finetune --data data.yaml --model weights/modelv11x.pt \
    --freeze-layers 10 --lr0 0.001

# YAML config 사용
python -m models.train finetune --config models/train/configs/finetune_default.yaml \
    --data data.yaml
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--model` | `weights/modelv11x.pt` | 기존 모델 가중치 경로 |
| `--data` | (필수) | data.yaml 경로 |
| `--freeze-layers` | 0 | 앞 N개 레이어 고정 (0=없음) |
| `--lr0` | 0.01 | 초기 학습률 |
| `--lrf` | 0.01 | 최종 학습률 (lr0 대비 비율) |
| `--warmup-epochs` | 3.0 | Warmup 에포크 수 |

### 2. Expand — 새 클래스 추가

기존 모델의 Detection Head를 확장하여 새 클래스를 추가합니다. 기존 클래스 가중치를 보존하면서 새 클래스를 학습합니다.

```bash
# 기존 모델(person,dog,cat)에 bowl 클래스 추가
python -m models.train expand \
    --old-model weights/modelv11x.pt \
    --combined-data data_with_bowl.yaml

# bowl 전용 config 사용
python -m models.train expand \
    --config models/train/configs/expand_bowl_config.yaml \
    --old-model weights/modelv11x.pt \
    --combined-data data_with_bowl.yaml

# 320x320 엣지 디바이스 타겟
python -m models.train expand \
    --config models/train/configs/expand_320.yaml \
    --old-model weights/modelv11x.pt \
    --combined-data data_with_bowl.yaml
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--old-model` | (필수) | 기존 모델 경로 (확장 대상) |
| `--combined-data` | (필수) | 전체 클래스 포함 data.yaml (기존 + 신규) |
| `--freeze-layers` | 10 | Backbone 고정 (기존 지식 보존) |
| `--lr0` | 0.001 | 낮은 학습률 (catastrophic forgetting 방지) |

**동작 원리:**

```
기존 모델 (nc=3)              확장 모델 (nc=4)
┌─────────────────┐          ┌─────────────────┐
│   Backbone      │──복사──▶│   Backbone      │
│   (동결)        │          │   (동결)        │
├─────────────────┤          ├─────────────────┤
│   Neck (FPN)    │──복사──▶│   Neck (FPN)    │
├─────────────────┤          ├─────────────────┤
│   Detect Head   │          │   Detect Head   │
│   cv3: 3 cls    │──확장──▶│   cv3: 4 cls    │
│ [p,d,c]         │          │ [p,d,c,bowl]    │
└─────────────────┘          └─────────────────┘
```

### 3. Distill — Knowledge Distillation

큰 Teacher 모델의 지식을 작은 Student 모델로 전달합니다.

```bash
# 기본 distill (v11x → v11n)
python -m models.train distill \
    --teacher weights/modelv11x.pt \
    --student yolo11n.pt \
    --data data.yaml

# Feature distillation 비활성화
python -m models.train distill \
    --teacher weights/modelv11x.pt \
    --student yolo11n.pt \
    --data data.yaml \
    --no-distill-features

# Temperature 조절
python -m models.train distill \
    --teacher weights/modelv11x.pt \
    --student yolo11s.pt \
    --data data.yaml \
    --temperature 6.0 --alpha 0.7
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--teacher` | (필수) | Teacher 모델 경로 (대형 모델) |
| `--student` | (필수) | Student 모델 경로 또는 YOLO config YAML |
| `--data` | (필수) | data.yaml 경로 |
| `--temperature` | 4.0 | 증류 온도 (높을수록 soft, 2.0~10.0 권장) |
| `--alpha` | 0.5 | 증류 비율 (0=task만, 1=distill만) |
| `--distill-features` | true | 중간 레이어 feature distillation |

**손실 함수 구조:**

```
                  Student Model
                       │
              ┌────────┴────────┐
              ▼                 ▼
        Task Loss         Distill Loss
    (student vs GT)    (student vs teacher)
              │                 │
              │         ┌───────┴───────┐
              │         ▼               ▼
              │    Response Loss   Feature Loss
              │    (KL Divergence)  (L2 on FPN)
              │         │               │
              └────┬────┘               │
                   ▼                    │
           (1-α) × task    +    α × (response + feature)
```

- **ResponseDistillationLoss**: Detection head 출력의 KL divergence (classification + box DFL)
- **FeatureDistillationLoss**: 중간 레이어 feature map의 L2 loss (1×1 conv adapter로 채널 맞춤)

## 데이터셋 관리

### merge — 데이터셋 병합

여러 YOLO 데이터셋을 클래스 인덱스 자동 리매핑으로 병합합니다.

```bash
# 두 데이터셋 병합
python -m models.train merge \
    --datasets dataset_A/data.yaml \
    --datasets dataset_B/data.yaml \
    --output merged

# 기존 모델 클래스 순서 유지
python -m models.train merge \
    --datasets ds1/data.yaml \
    --datasets ds2/data.yaml \
    --old-model weights/modelv11x.pt

# 재분할 (기존 split 무시)
python -m models.train merge \
    --datasets ds1/data.yaml \
    --datasets ds2/data.yaml \
    --resplit --train-ratio 0.8

# 특정 클래스만 포함
python -m models.train merge \
    --datasets ds1/data.yaml \
    --datasets ds2/data.yaml \
    --include-classes dog,cat

# Symlink 모드 (빠르고 디스크 절약)
python -m models.train merge \
    --datasets ds1/data.yaml \
    --datasets ds2/data.yaml \
    --symlink
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--datasets` | (필수, 복수) | data.yaml 경로들 |
| `--output` | `merged` | 출력 디렉토리 |
| `--old-model` | `""` | 기존 모델 (클래스 순서 보존) |
| `--preserve-splits` | true | 원본 train/val/test 유지 |
| `--train-ratio` | 0.7 | 재분할 시 train 비율 |
| `--val-ratio` | 0.15 | 재분할 시 val 비율 |
| `--test-ratio` | 0.15 | 재분할 시 test 비율 |
| `--keep-unmapped` | false | 매핑 안 되는 클래스 유지 |
| `--symlink` | false | 이미지 복사 대신 symlink |
| `--include-classes` | `""` | 포함할 클래스 (쉼표 구분) |

### prepare — 데이터 구조 변환

플랫 폴더(이미지+라벨 혼합)를 YOLO 디렉토리 구조로 변환합니다.

```bash
# 기본 변환
python -m models.train prepare --input data/raw --output data/prepared

# 클래스 이름 지정 + 분할 비율 조정
python -m models.train prepare \
    --input data/raw --output data/prepared \
    --class-names "background,pet,person,bowl" \
    --val-ratio 0.2 --test-ratio 0.1
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--input` | (필수) | 원본 폴더 (이미지+라벨 혼합) |
| `--output` | (필수) | 출력 YOLO 구조 디렉토리 |
| `--val-ratio` | 0.15 | Validation 비율 |
| `--test-ratio` | 0.0 | Test 비율 (0=없음) |
| `--class-names` | `""` | 클래스 이름 (쉼표 구분) |
| `--seed` | 42 | 랜덤 시드 |
| `--symlink` | false | Symlink 사용 |

**변환 결과:**

```
# 입력 (플랫)                # 출력 (YOLO 구조)
data/raw/                    data/prepared/
├── img001.jpg               ├── data.yaml
├── img001.txt               ├── train/
├── img002.jpg               │   ├── images/
├── img002.txt               │   │   ├── img001.jpg
└── ...                      │   │   └── ...
                             │   └── labels/
                             │       ├── img001.txt
                             │       └── ...
                             └── val/
                                 ├── images/
                                 └── labels/
```

### gather — 데이터셋 이동 + 리매핑

소스 데이터셋 파일을 타겟 데이터셋으로 이동하면서 라벨을 자동 리매핑합니다.

```bash
# 소스 → 타겟으로 이동
python -m models.train gather \
    --target A/data.yaml \
    --source B/data.yaml \
    --source C/data.yaml

# 미리보기 (실제 이동 안 함)
python -m models.train gather \
    --target A/data.yaml \
    --source B/data.yaml \
    --dry-run

# 특정 클래스만
python -m models.train gather \
    --target A/data.yaml \
    --source B/data.yaml \
    --include-classes dog,cat
```

### analyze — 클래스 분포 분석

데이터셋의 클래스별 인스턴스 수, 이미지 수, 불균형 비율을 표시합니다.

```bash
# 전체 분석
python -m models.train analyze --data dataset/data.yaml

# 특정 split만
python -m models.train analyze --data dataset/data.yaml --split train
```

### sample — 클래스 균형 샘플링

클래스 불균형을 해결하기 위한 샘플링 도구입니다.

```bash
# Undersample (다수 클래스 줄이기)
python -m models.train sample \
    --data dataset/data.yaml --output balanced \
    --strategy undersample

# 클래스당 최대 500개 이미지
python -m models.train sample \
    --data dataset/data.yaml --output balanced \
    --strategy cap --max-per-class 500

# Oversample (소수 클래스 늘리기)
python -m models.train sample \
    --data dataset/data.yaml --output balanced \
    --strategy oversample --min-per-class 1000
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--data` | (필수) | 소스 data.yaml |
| `--output` | (필수) | 출력 디렉토리 |
| `--strategy` | `undersample` | `undersample` / `oversample` / `cap` |
| `--max-per-class` | None | undersample/cap 최대 이미지 수 |
| `--min-per-class` | None | oversample 최소 이미지 수 |
| `--seed` | 42 | 랜덤 시드 |

## 공통 학습 옵션

finetune, expand, distill 공통:

### 기본

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--epochs` | 100 | 학습 에포크 수 |
| `--imgsz` | 640 | 입력 이미지 크기 |
| `--batch` | 16 | 배치 크기 |
| `--device` | auto | CUDA(`0`) > MPS(`mps`) > CPU(`cpu`) 자동 감지 |
| `--workers` | 8 | 데이터 로딩 워커 수 |
| `--project` | `runs/train` | 결과 저장 디렉토리 |
| `--name` | None | 실험 이름 |
| `--patience` | 50 | Early stopping patience |
| `--optimizer` | `auto` | `SGD` / `Adam` / `AdamW` / `auto` |
| `--resume` | false | 마지막 체크포인트에서 재개 |

### 손실 함수 가중치

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--cls-loss` | 0.5 | Classification loss 가중치 |
| `--box-loss` | 7.5 | Box regression loss 가중치 |
| `--dfl-loss` | 1.5 | Distribution focal loss 가중치 |

### Augmentation

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--mosaic` | 1.0 | Mosaic 확률 |
| `--copy-paste` | 0.3 | Copy-paste 확률 |
| `--mixup` | 0.15 | MixUp 확률 |
| `--erasing` | 0.4 | Random erasing 확률 |
| `--scale` | 0.9 | 크기 변화 ±비율 |
| `--degrees` | 10.0 | 회전 ±도 |
| `--translate` | 0.2 | 이동 ±비율 |
| `--hsv-h` | 0.015 | HSV 색조 변화 |
| `--hsv-s` | 0.7 | HSV 채도 변화 |
| `--hsv-v` | 0.4 | HSV 밝기 변화 |
| `--fliplr` | 0.5 | 좌우 반전 확률 |
| `--flipud` | 0.0 | 상하 반전 확률 |
| `--close-mosaic` | 10 | 마지막 N 에포크 mosaic 비활성화 |

## YAML Config 사용

CLI 옵션 대신 YAML 파일로 설정을 관리할 수 있습니다.

```bash
python -m models.train finetune --config models/train/configs/finetune_default.yaml --data data.yaml
python -m models.train expand --config models/train/configs/expand_bowl_config.yaml --old-model model.pt --combined-data data.yaml
python -m models.train distill --config models/train/configs/distill_default.yaml --teacher teacher.pt --student student.pt --data data.yaml
```

기본 제공 설정 파일:

| 파일 | 용도 |
|------|------|
| `finetune_default.yaml` | 표준 fine-tuning |
| `expand_default.yaml` | 표준 클래스 확장 |
| `expand_bowl_config.yaml` | Bowl 클래스 추가 특화 |
| `expand_320.yaml` | 320×320 엣지 디바이스 타겟 |
| `distill_default.yaml` | 표준 knowledge distillation |

## Python API

```python
from models.train import (
    run_finetune, run_expand, run_distill, run_merge,
    FinetuneConfig, ExpandConfig, DistillConfig, MergeConfig,
    LossConfig, AugmentConfig, load_config, save_config,
)

# Fine-tune
config = FinetuneConfig(
    model="weights/modelv11x.pt",
    data="data.yaml",
    epochs=100,
    freeze_layers=0,
)
result = run_finetune(config)
print(f"Best model: {result['best_model']}")

# Expand
config = ExpandConfig(
    old_model="weights/modelv11x.pt",
    combined_data="data_with_bowl.yaml",
    freeze_layers=10,
    lr0=0.001,
)
result = run_expand(config)

# Distill
config = DistillConfig(
    teacher_model="weights/modelv11x.pt",
    student_model="yolo11n.pt",
    data="data.yaml",
    temperature=4.0,
    alpha=0.5,
)
result = run_distill(config)

# Merge
config = MergeConfig(
    datasets=["ds1/data.yaml", "ds2/data.yaml"],
    output_dir="merged",
    preserve_splits=True,
)
result = run_merge(config)
print(f"Unified classes: {result['unified_classes']}")
print(f"Total images: {result['total_images']}")

# Config 저장/로드
save_config(config, "my_config.yaml")
loaded = load_config("my_config.yaml", FinetuneConfig)
```

## 출력 구조

```
runs/train/{name}/
├── weights/
│   ├── best.pt         # 최고 mAP 모델
│   └── last.pt         # 마지막 에포크 모델
├── results.csv         # 에포크별 메트릭
├── results.png         # 학습 그래프
├── confusion_matrix.png
├── PR_curve.png
├── F1_curve.png
└── args.yaml           # 학습에 사용된 설정
```

## 권장 워크플로우

### 새 데이터로 성능 개선

```bash
# 1. 데이터 준비
python -m models.train prepare --input raw_data/ --output prepared/ \
    --class-names "background,pet,person,bowl"

# 2. 클래스 분포 확인
python -m models.train analyze --data prepared/data.yaml

# 3. 불균형 심하면 샘플링
python -m models.train sample --data prepared/data.yaml --output balanced/ \
    --strategy cap --max-per-class 2000

# 4. Fine-tune
python -m models.train finetune --data balanced/data.yaml --model weights/modelv11x.pt \
    --epochs 100 --patience 30
```

### 새 클래스 추가 (예: bowl)

```bash
# 1. 기존 데이터셋 + 새 클래스 데이터셋 병합
python -m models.train merge \
    --datasets existing/data.yaml \
    --datasets bowl_data/data.yaml \
    --old-model weights/modelv11x.pt \
    --output merged/

# 2. 클래스 분포 확인
python -m models.train analyze --data merged/data.yaml

# 3. Expand 학습
python -m models.train expand \
    --config models/train/configs/expand_bowl_config.yaml \
    --old-model weights/modelv11x.pt \
    --combined-data merged/data.yaml
```

### 경량 모델 생성

```bash
# Teacher(v11x) → Student(v11n) distillation
python -m models.train distill \
    --teacher weights/modelv11x.pt \
    --student yolo11n.pt \
    --data data.yaml \
    --temperature 4.0 --alpha 0.5 \
    --epochs 100
```

## 파일 구조

```
models/train/
├── cli.py             # Click CLI (finetune, expand, distill, merge, prepare, gather, analyze, sample)
├── config.py          # TrainConfig, FinetuneConfig, ExpandConfig, DistillConfig, MergeConfig
├── finetune.py        # run_finetune() — 동일 클래스 fine-tuning
├── expand.py          # run_expand(), ExpandTrainer, expand_detection_head()
├── distill.py         # run_distill(), DistillationTrainer
├── merge.py           # run_merge() — 데이터셋 병합 + 클래스 리매핑
├── losses.py          # ResponseDistillationLoss, FeatureDistillationLoss, DistillationLoss
├── utils.py           # build_train_args(), expand_detection_head(), print_banner()
├── __init__.py        # 공개 API export
└── configs/
    ├── finetune_default.yaml
    ├── expand_default.yaml
    ├── expand_bowl_config.yaml
    ├── expand_320.yaml
    └── distill_default.yaml
```
