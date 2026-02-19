"""
Lightweight Re-ID Feature Extractor - CPU 실시간 처리 최적화 버전

기존 OSNet/EfficientNet 대신 경량 모델 사용:
1. MobileNetV3-Small: 빠르고 정확한 특징 추출
2. Color + Texture Histogram: 딥러닝 없이 빠른 처리
3. 선택적 Re-ID: 필요할 때만 적용하여 연산량 감소
"""

import cv2
import numpy as np
import logging
from collections import deque
from typing import List, Optional, Tuple
import time

# PyTorch (선택적)
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from torchvision import models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch 사용 불가 - 히스토그램 기반 Re-ID만 사용")


class FastHistogramReID:
    """
    색상 + 텍스처 히스토그램 기반 빠른 Re-ID
    CPU에서 ~1ms 이내 처리 가능
    """

    def __init__(self, color_bins: int = 16, texture_bins: int = 8):
        self.color_bins = color_bins
        self.texture_bins = texture_bins

        # 특징 차원: HSV(16*3) + 텍스처(8) = 56
        self.feature_dim = color_bins * 3 + texture_bins

        # 미리 계산된 범위
        self._hsv_ranges = {
            'h': [0, 180],
            's': [0, 256],
            'v': [0, 256]
        }

        logging.info(f"FastHistogramReID 초기화 - feature_dim: {self.feature_dim}")

    def extract_features(self, crop_image: np.ndarray) -> np.ndarray:
        """
        빠른 히스토그램 기반 특징 추출

        Args:
            crop_image: BGR 이미지 (H, W, 3)

        Returns:
            정규화된 특징 벡터 (feature_dim,)
        """
        if crop_image is None or crop_image.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        try:
            # 크기 정규화 (작은 크기로 빠른 처리)
            h, w = crop_image.shape[:2]
            if h > 64 or w > 32:
                crop_resized = cv2.resize(crop_image, (32, 64), interpolation=cv2.INTER_AREA)
            else:
                crop_resized = crop_image

            # HSV 변환 및 색상 히스토그램
            hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)

            features = []

            # H, S, V 각 채널 히스토그램
            for i, (key, range_) in enumerate(self._hsv_ranges.items()):
                hist = cv2.calcHist([hsv], [i], None, [self.color_bins], range_)
                features.append(hist.flatten())

            # 텍스처 특징 (LBP 간소화 버전 - 그래디언트 기반)
            gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = np.sqrt(gx**2 + gy**2)

            # 그래디언트 크기 히스토그램
            texture_hist = cv2.calcHist([magnitude.astype(np.float32)],
                                        [0], None, [self.texture_bins], [0, 256])
            features.append(texture_hist.flatten())

            # 결합 및 정규화
            combined = np.concatenate(features)
            norm = np.linalg.norm(combined)
            if norm > 1e-7:
                combined /= norm

            return combined.astype(np.float32)

        except Exception as e:
            logging.warning(f"FastHistogram 특징 추출 실패: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)

    def extract_features_batch(self, crop_images: List[np.ndarray]) -> np.ndarray:
        """배치 특징 추출"""
        if not crop_images:
            return np.array([])

        features = [self.extract_features(img) for img in crop_images]
        return np.array(features, dtype=np.float32)


class MobileNetReID:
    """
    MobileNetV3-Small 기반 경량 Re-ID
    CPU에서 ~10-15ms 처리 (OSNet의 1/4 수준)
    """

    def __init__(self, use_pretrained: bool = True):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch가 필요합니다")

        self.device = torch.device('cpu')  # CPU 강제
        self.feature_dim = 576  # MobileNetV3-Small 마지막 레이어

        # 전처리 (간소화)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.target_size = (96, 48)  # 작은 입력 크기로 빠른 처리

        # 모델 로드
        self.model = None
        self._load_model(use_pretrained)

        # 성능 통계
        self.inference_times = deque(maxlen=100)

        logging.info(f"MobileNetReID 초기화 - feature_dim: {self.feature_dim}")

    def _load_model(self, use_pretrained: bool):
        """MobileNetV3-Small 모델 로드"""
        try:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if use_pretrained else None
            self.model = models.mobilenet_v3_small(weights=weights)

            # 분류 레이어 제거하고 특징 추출만
            self.model.classifier = nn.Identity()
            self.model.eval()

            # 실제 특징 차원 확인
            with torch.no_grad():
                dummy = torch.randn(1, 3, *self.target_size)
                out = self.model(dummy)
                self.feature_dim = out.shape[1]

            logging.info(f"MobileNetV3-Small 로드 완료 (feature_dim: {self.feature_dim})")

        except Exception as e:
            logging.error(f"MobileNet 로드 실패: {e}")
            self.model = None

    def extract_features(self, crop_image: np.ndarray) -> np.ndarray:
        """특징 추출"""
        if self.model is None or crop_image is None or crop_image.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        try:
            start = time.time()

            # 전처리
            rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, self.target_size, interpolation=cv2.INTER_LINEAR)

            tensor = self.transform(resized).unsqueeze(0)

            # 추론
            with torch.no_grad():
                features = self.model(tensor).squeeze().numpy()

            # 정규화
            norm = np.linalg.norm(features)
            if norm > 1e-7:
                features /= norm

            self.inference_times.append(time.time() - start)

            return features.astype(np.float32)

        except Exception as e:
            logging.warning(f"MobileNet 특징 추출 실패: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)

    def extract_features_batch(self, crop_images: List[np.ndarray]) -> np.ndarray:
        """배치 특징 추출"""
        if self.model is None or not crop_images:
            return np.array([])

        try:
            # 배치 텐서 생성
            batch_tensors = []
            valid_indices = []

            for i, img in enumerate(crop_images):
                if img is not None and img.size > 0:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
                    tensor = self.transform(resized)
                    batch_tensors.append(tensor)
                    valid_indices.append(i)

            if not batch_tensors:
                return np.zeros((len(crop_images), self.feature_dim), dtype=np.float32)

            batch = torch.stack(batch_tensors)

            # 배치 추론
            with torch.no_grad():
                batch_features = self.model(batch).numpy()

            # 결과 조합
            all_features = np.zeros((len(crop_images), self.feature_dim), dtype=np.float32)
            for i, valid_idx in enumerate(valid_indices):
                features = batch_features[i]
                norm = np.linalg.norm(features)
                if norm > 1e-7:
                    features /= norm
                all_features[valid_idx] = features

            return all_features

        except Exception as e:
            logging.warning(f"MobileNet 배치 추출 실패: {e}")
            return np.zeros((len(crop_images), self.feature_dim), dtype=np.float32)

    def get_avg_inference_time(self) -> float:
        """평균 추론 시간"""
        if not self.inference_times:
            return 0.0
        return np.mean(self.inference_times)


class AdaptiveReID:
    """
    적응형 Re-ID - 상황에 따라 경량/정밀 모델 선택

    - 기본: FastHistogram (빠름)
    - ID 스위칭 의심시: MobileNet (정확)
    """

    def __init__(self, use_deep_model: bool = True, switch_detection_threshold: float = 0.3):
        """
        Args:
            use_deep_model: 딥러닝 모델 사용 여부
            switch_detection_threshold: ID 스위칭 감지 임계값
        """
        # 빠른 모델 (항상 사용)
        self.fast_extractor = FastHistogramReID()

        # 정밀 모델 (선택적)
        self.deep_extractor = None
        if use_deep_model and TORCH_AVAILABLE:
            try:
                self.deep_extractor = MobileNetReID()
            except Exception as e:
                logging.warning(f"MobileNet 초기화 실패, 히스토그램만 사용: {e}")

        # 통합 특징 차원
        self.feature_dim = self.fast_extractor.feature_dim
        if self.deep_extractor:
            self.feature_dim += self.deep_extractor.feature_dim

        self.switch_threshold = switch_detection_threshold

        # 트랙별 상태 (ID 스위칭 감지용)
        self.track_states = {}  # track_id -> {'prev_feature', 'stable_count'}

        logging.info(f"AdaptiveReID 초기화 - feature_dim: {self.feature_dim}, deep_model: {self.deep_extractor is not None}")

    def should_use_deep_model(self, track_id: int, fast_feature: np.ndarray) -> bool:
        """
        딥러닝 모델 사용 여부 결정

        ID 스위칭이 의심되면 True 반환
        """
        if self.deep_extractor is None:
            return False

        if track_id not in self.track_states:
            # 새 트랙 - 처음에는 deep model 사용
            self.track_states[track_id] = {
                'prev_feature': fast_feature,
                'stable_count': 0
            }
            return True

        state = self.track_states[track_id]

        # 이전 특징과 비교
        if state['prev_feature'] is not None:
            similarity = np.dot(fast_feature, state['prev_feature'])

            if similarity < self.switch_threshold:
                # 급격한 변화 = ID 스위칭 의심
                state['stable_count'] = 0
                state['prev_feature'] = fast_feature
                return True
            else:
                state['stable_count'] += 1

        state['prev_feature'] = fast_feature

        # 안정적인 트랙은 deep model 스킵 (5프레임 이상 안정)
        return state['stable_count'] < 5

    def extract_features(self, crop_image: np.ndarray, track_id: int = None) -> np.ndarray:
        """
        적응형 특징 추출

        Args:
            crop_image: 이미지
            track_id: 트랙 ID (ID 스위칭 감지용, 선택적)
        """
        if crop_image is None or crop_image.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        # Fast feature 추출 (항상)
        fast_feat = self.fast_extractor.extract_features(crop_image)

        # Deep feature 추출 (조건부)
        if self.deep_extractor:
            if track_id is None or self.should_use_deep_model(track_id, fast_feat):
                deep_feat = self.deep_extractor.extract_features(crop_image)
            else:
                # 이전 deep feature 재사용 또는 0
                deep_feat = np.zeros(self.deep_extractor.feature_dim, dtype=np.float32)

            # 결합
            combined = np.concatenate([fast_feat, deep_feat])
        else:
            combined = fast_feat

        # 정규화
        norm = np.linalg.norm(combined)
        if norm > 1e-7:
            combined /= norm

        return combined

    def extract_features_batch(self, crop_images: List[np.ndarray],
                               track_ids: List[int] = None) -> np.ndarray:
        """배치 특징 추출"""
        if not crop_images:
            return np.array([])

        n = len(crop_images)

        # Fast features
        fast_feats = self.fast_extractor.extract_features_batch(crop_images)

        if self.deep_extractor:
            # 어떤 트랙에 deep model 적용할지 결정
            if track_ids is None:
                # 모든 이미지에 적용
                deep_feats = self.deep_extractor.extract_features_batch(crop_images)
            else:
                # 선택적 적용
                need_deep = []
                need_deep_indices = []

                for i, (img, tid, ff) in enumerate(zip(crop_images, track_ids, fast_feats)):
                    if self.should_use_deep_model(tid, ff):
                        need_deep.append(img)
                        need_deep_indices.append(i)

                deep_feats = np.zeros((n, self.deep_extractor.feature_dim), dtype=np.float32)

                if need_deep:
                    extracted = self.deep_extractor.extract_features_batch(need_deep)
                    for j, idx in enumerate(need_deep_indices):
                        deep_feats[idx] = extracted[j]

            # 결합
            combined = np.concatenate([fast_feats, deep_feats], axis=1)
        else:
            combined = fast_feats

        # 정규화
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        norms[norms < 1e-7] = 1.0
        combined /= norms

        return combined

    def cleanup_tracks(self, active_track_ids: set):
        """비활성 트랙 상태 정리"""
        inactive = [tid for tid in self.track_states if tid not in active_track_ids]
        for tid in inactive:
            del self.track_states[tid]


def create_lightweight_reid(method: str = 'adaptive', **kwargs):
    """
    경량 Re-ID 추출기 팩토리

    Args:
        method: 'histogram', 'mobilenet', 'adaptive' 중 하나
    """
    if method == 'histogram':
        return FastHistogramReID(**kwargs)
    elif method == 'mobilenet':
        return MobileNetReID(**kwargs)
    elif method == 'adaptive':
        return AdaptiveReID(**kwargs)
    else:
        raise ValueError(f"지원하지 않는 방법: {method}")


# 테스트
if __name__ == "__main__":
    import time

    # 테스트 이미지
    test_img = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)

    print("=== CPU Re-ID 성능 테스트 ===\n")

    # 1. FastHistogram
    extractor = FastHistogramReID()
    times = []
    for _ in range(100):
        start = time.time()
        feat = extractor.extract_features(test_img)
        times.append(time.time() - start)
    print(f"FastHistogram: {np.mean(times)*1000:.2f}ms, dim={len(feat)}")

    # 2. MobileNet (if available)
    if TORCH_AVAILABLE:
        extractor = MobileNetReID()
        times = []
        for _ in range(20):
            start = time.time()
            feat = extractor.extract_features(test_img)
            times.append(time.time() - start)
        print(f"MobileNet: {np.mean(times)*1000:.2f}ms, dim={len(feat)}")

        # 3. Adaptive
        extractor = AdaptiveReID()
        times = []
        for i in range(20):
            start = time.time()
            feat = extractor.extract_features(test_img, track_id=1)
            times.append(time.time() - start)
        print(f"Adaptive: {np.mean(times)*1000:.2f}ms, dim={len(feat)}")
