# Pet ReID Metric Learning Pipeline

CCTV crop 이미지와 reference 이미지 간의 유사도를 학습하여, 동일 반려동물의 임베딩이 가까워지고 다른 반려동물의 임베딩이 멀어지도록 인코더를 훈련합니다.

## 아키텍처

```
입력 이미지 (224x224)
    │
    ▼
┌──────────────────────┐
│  Backbone (frozen/    │   DINOv2 ViT-S/14 (384D)
│  trainable)           │   또는 MobileNetV3-Small (576D)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Projection Head      │   Linear → BN → ReLU → Linear → BN
│  (항상 학습)           │   backbone_dim → embed_dim (256D)
└──────────┬───────────┘
           │
           ▼
    L2 정규화 임베딩 [256D]
```

## 손실 함수

| 손실 | 설명 |
|------|------|
| **OnlineTripletLoss** | 배치 내 hardest positive/negative 마이닝. 같은 ID는 가깝게, 다른 ID는 멀게 |
| **ArcFaceLoss** | Angular margin 기반 분류. 임베딩 클러스터를 더 타이트하게 |
| **CombinedLoss** (기본) | `triplet_weight * Triplet + arcface_weight * ArcFace` |

## 데이터 준비

### 폴더 구조

```
data/reid/
├── 뽀삐/                    # identity 폴더 (이름 자유)
│   ├── ref_01.jpg           # reference 이미지 (~20%)
│   ├── ref_02.jpg
│   ├── crop_001.jpg         # CCTV crop 이미지 (~80%)
│   ├── crop_002.jpg
│   └── crop_003.jpg
├── 코코/
│   ├── ref_01.jpg
│   ├── crop_001.jpg
│   └── ...
├── 몽이/
│   └── ...
└── ... (최소 8개 identity, P=8 기준)
```

### 데이터 수집

1. **Reference 이미지**: 반려동물 정면/측면 사진 (배경 깔끔, 대상만 포함)
2. **CCTV crop 이미지**: `tools/crop_detections.py`로 추출

```bash
# YOLO tracking으로 CCTV crop 추출 (ID별 폴더 분류)
python tools/crop_detections.py crop \
    --input cctv_video.mp4 --output data/crops/ \
    --track --interval 5

# 여러 RTSP 스트림에서 프레임 캡처
python tools/crop_detections.py capture \
    --input "rtsp://cam1:554/stream" \
    --input "rtsp://cam2:554/stream" \
    --output data/frames/ --seconds 2
```

3. 추출된 crop을 각 반려동물 폴더로 수동 분류
4. **비율**: ref:crop = 2:8 권장
5. **수량**: identity당 최소 2장, 권장 15장 이상, 많을수록 좋음

### 주의사항

- reference 이미지에 다른 반려동물이 포함되면 안 됨
- identity당 최소 2장 필요 (미만은 자동 스킵)
- 최소 P개 identity 필요 (기본 P=8)

## 사용법

### CLI

```bash
# 기본 학습 (DINOv2 backbone, combined loss)
python -m models.train reid --data-root data/reid

# MobileNet backbone, head만 학습 (빠름)
python -m models.train reid \
    --data-root data/reid \
    --backbone mobilenet_v3_small \
    --freeze-backbone

# Triplet loss만, 100 에포크
python -m models.train reid \
    --data-root data/reid \
    --loss triplet \
    --epochs 100

# YAML config 사용
python -m models.train reid \
    --config models/train/configs/reid_default.yaml \
    --data-root data/reid

# GPU 지정, 배치 크기 조절
python -m models.train reid \
    --data-root data/reid \
    --device 0 \
    --p 16 --k 4
```

### Python API

```python
from models.train import run_reid, ReIDConfig

config = ReIDConfig(
    data_root="data/reid",
    backbone="dinov2_vits14",
    embed_dim=256,
    epochs=60,
    loss="combined",
)

result = run_reid(config)
print(f"Best Rank-1: {result['best_metrics']['rank1']:.1f}%")
print(f"Best model: {result['best_model']}")
```

### 학습된 모델 로드

```python
from models.train.reid_model import ReIDModel

model = ReIDModel.load_inference("runs/reid/pet_reid/best.pt", device="cuda")

# 임베딩 추출
import torch
images = torch.randn(4, 3, 224, 224).cuda()
embeddings = model.extract(images)  # [4, 256], L2-normalized
```

## 주요 옵션

### 모델

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--backbone` | `dinov2_vits14` | backbone 모델. `mobilenet_v3_small`도 가능 |
| `--embed-dim` | 256 | 출력 임베딩 차원 |
| `--freeze-backbone` | false | backbone 가중치 고정 (projection head만 학습) |

### 학습

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--epochs` | 60 | 학습 에포크 수 |
| `--p` | 8 | PK sampler: 배치당 identity 수 |
| `--k` | 4 | PK sampler: identity당 이미지 수 |
| `--lr` | 1e-4 | 학습률 (backbone은 lr * 0.1) |
| `--scheduler` | cosine | LR 스케줄러: `cosine` 또는 `step` |
| `--warmup-epochs` | 5 | warmup 에포크 수 |

### 손실 함수

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--loss` | `combined` | `triplet`, `arcface`, `combined` |
| `--triplet-margin` | 0.3 | Triplet loss margin |
| `--arcface-scale` | 30.0 | ArcFace scale (s) |
| `--arcface-margin` | 0.5 | ArcFace angular margin (m) |

### Augmentation

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--imgsz` | 224 | 입력 이미지 크기 |
| `random_crop` | true | Random crop (imgsz+32 → imgsz) |
| `color_jitter` | 0.3 | 밝기/대비/채도 변화 |
| `random_erasing` | 0.5 | Random erasing (occlusion 시뮬레이션) |
| `horizontal_flip` | 0.5 | 좌우 반전 확률 |

## 평가 지표

학습 중 `--save-interval` 에포크마다 query/gallery split으로 평가:

| 지표 | 설명 |
|------|------|
| **Rank-1** | query 이미지의 가장 가까운 gallery 이미지가 같은 identity인 비율 |
| **Rank-5** | 상위 5개 중 같은 identity가 있는 비율 |
| **Rank-10** | 상위 10개 중 같은 identity가 있는 비율 |
| **mAP** | Mean Average Precision — 전체 gallery에서의 검색 정확도 |

## PK Sampler

일반 랜덤 배치 대신 **P identities x K images** 구조로 배치를 구성합니다.

```
배치 (P=4, K=3, batch_size=12):
  [뽀삐_1, 뽀삐_2, 뽀삐_3, 코코_1, 코코_2, 코코_3, 몽이_1, 몽이_2, 몽이_3, 두부_1, 두부_2, 두부_3]
```

이렇게 하면 각 배치에 positive pair (같은 identity)와 negative pair (다른 identity)가 반드시 존재하여, online hard mining이 효과적으로 동작합니다.

## 출력 구조

```
runs/reid/pet_reid/
├── best.pt          # 최고 Rank-1 모델 (inference용)
├── last.pt          # 마지막 에포크 모델
├── epoch_5.pt       # 체크포인트 (학습 재개용)
├── epoch_10.pt
└── ...
```

## 파일 구조

```
models/train/
├── reid_model.py      # ReIDModel: backbone + projection head
├── reid_losses.py     # OnlineTripletLoss, ArcFaceLoss, CombinedReIDLoss
├── reid_dataset.py    # ReIDDataset, PKSampler, augmentation, query/gallery split
├── reid_train.py      # 학습 루프, 평가, run_reid()
├── config.py          # ReIDConfig dataclass
├── cli.py             # `reid` CLI 커맨드
└── configs/
    └── reid_default.yaml  # 기본 설정
```

## 권장 학습 전략

### 1단계: Backbone 고정, Head만 학습 (빠른 수렴)

```bash
python -m models.train reid \
    --data-root data/reid \
    --freeze-backbone \
    --epochs 30 \
    --lr 1e-3
```

### 2단계: 전체 fine-tuning (정확도 향상)

```bash
python -m models.train reid \
    --data-root data/reid \
    --epochs 60 \
    --lr 1e-4
```

### 데이터가 적을 때 (identity당 5장 미만)

```bash
python -m models.train reid \
    --data-root data/reid \
    --freeze-backbone \
    --p 4 --k 2 \
    --loss triplet \
    --epochs 100
```
