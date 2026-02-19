"""
Appearance Feature Extractors

외형 기반 특징 추출기들 - 색상, 텍스처, 딥러닝 특징
"""

import cv2
import numpy as np
from typing import List, Optional, Dict
from collections import deque
import time
import logging

from .base import FeatureExtractor, FeatureConfig, TrackContext, FeatureOutput

# PyTorch (선택적)
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from torchvision import models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class HistogramFeatureExtractor(FeatureExtractor):
    """
    색상 + 텍스처 히스토그램 기반 빠른 특징 추출
    CPU에서 ~1ms 이내 처리
    """

    def __init__(self, config: FeatureConfig = None,
                 color_bins: int = 16,
                 texture_bins: int = 8,
                 use_spatial_pyramid: bool = False):
        config = config or FeatureConfig(name="histogram")
        super().__init__(config)

        self.color_bins = color_bins
        self.texture_bins = texture_bins
        self.use_spatial_pyramid = use_spatial_pyramid

        # 기본: HSV(16*3) + 텍스처(8) = 56
        # Spatial pyramid: 56 * 5 (1 global + 4 quadrants) = 280
        base_dim = color_bins * 3 + texture_bins
        self._feature_dim = base_dim * 5 if use_spatial_pyramid else base_dim

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def feature_type(self) -> str:
        return "appearance"

    def _extract_histogram(self, image: np.ndarray) -> np.ndarray:
        """단일 영역에서 히스토그램 추출"""
        if image is None or image.size == 0:
            return np.zeros(self.color_bins * 3 + self.texture_bins, dtype=np.float32)

        # HSV 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        features = []
        # H, S, V 각 채널 히스토그램
        ranges = [(0, 180), (0, 256), (0, 256)]
        for i, (low, high) in enumerate(ranges):
            hist = cv2.calcHist([hsv], [i], None, [self.color_bins], [low, high])
            features.append(hist.flatten())

        # 텍스처 (그래디언트 기반)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)

        texture_hist = cv2.calcHist([magnitude.astype(np.float32)],
                                    [0], None, [self.texture_bins], [0, 256])
        features.append(texture_hist.flatten())

        return np.concatenate(features).astype(np.float32)

    def extract(self, image: np.ndarray, context: TrackContext = None) -> FeatureOutput:
        if image is None or image.size == 0:
            return self.get_zero_feature()

        try:
            # 크기 정규화
            h, w = image.shape[:2]
            if h > 64 or w > 32:
                image = cv2.resize(image, (32, 64), interpolation=cv2.INTER_AREA)
                h, w = 64, 32

            if self.use_spatial_pyramid:
                # Spatial Pyramid Pooling
                features = [self._extract_histogram(image)]  # 전체

                # 4분할
                h2, w2 = h // 2, w // 2
                quadrants = [
                    image[:h2, :w2],
                    image[:h2, w2:],
                    image[h2:, :w2],
                    image[h2:, w2:]
                ]
                for quad in quadrants:
                    if quad.size > 0:
                        features.append(self._extract_histogram(quad))
                    else:
                        features.append(np.zeros(self.color_bins * 3 + self.texture_bins))

                combined = np.concatenate(features)
            else:
                combined = self._extract_histogram(image)

            combined = self.normalize(combined)

            return FeatureOutput(
                feature=combined,
                feature_type=self.feature_type,
                confidence=1.0
            )

        except Exception as e:
            self.logger.warning(f"Histogram extraction failed: {e}")
            return self.get_zero_feature()


class MobileNetFeatureExtractor(FeatureExtractor):
    """
    MobileNetV3-Small 기반 딥러닝 특징 추출
    CPU에서 ~10-15ms 처리
    """

    def __init__(self, config: FeatureConfig = None,
                 use_pretrained: bool = True,
                 input_size: tuple = (96, 48)):
        config = config or FeatureConfig(name="mobilenet")
        super().__init__(config)

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for MobileNetFeatureExtractor")

        self.use_pretrained = use_pretrained
        self.input_size = input_size  # (H, W)
        self._feature_dim = 576  # MobileNetV3-Small 출력

        self.device = torch.device('cpu')
        self.model = None
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 성능 통계
        self.inference_times = deque(maxlen=100)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def feature_type(self) -> str:
        return "appearance"

    def initialize(self) -> bool:
        try:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if self.use_pretrained else None
            self.model = models.mobilenet_v3_small(weights=weights)
            self.model.classifier = nn.Identity()
            self.model.eval()

            # 실제 특징 차원 확인
            with torch.no_grad():
                dummy = torch.randn(1, 3, *self.input_size)
                out = self.model(dummy)
                self._feature_dim = out.shape[1]

            self._initialized = True
            self.logger.info(f"MobileNet initialized (feature_dim: {self._feature_dim})")
            return True

        except Exception as e:
            self.logger.error(f"MobileNet initialization failed: {e}")
            return False

    def extract(self, image: np.ndarray, context: TrackContext = None) -> FeatureOutput:
        if self.model is None or image is None or image.size == 0:
            return self.get_zero_feature()

        try:
            start = time.time()

            # 전처리
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (self.input_size[1], self.input_size[0]),
                                 interpolation=cv2.INTER_LINEAR)
            tensor = self.transform(resized).unsqueeze(0)

            # 추론
            with torch.no_grad():
                features = self.model(tensor).squeeze().numpy()

            features = self.normalize(features)
            self.inference_times.append(time.time() - start)

            return FeatureOutput(
                feature=features.astype(np.float32),
                feature_type=self.feature_type,
                confidence=1.0,
                metadata={'inference_time': self.inference_times[-1]}
            )

        except Exception as e:
            self.logger.warning(f"MobileNet extraction failed: {e}")
            return self.get_zero_feature()

    def extract_batch(self, images: List[np.ndarray],
                     contexts: List[TrackContext] = None) -> List[FeatureOutput]:
        if self.model is None or not images:
            return [self.get_zero_feature() for _ in range(len(images) if images else 0)]

        try:
            batch_tensors = []
            valid_indices = []

            for i, img in enumerate(images):
                if img is not None and img.size > 0:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(rgb, (self.input_size[1], self.input_size[0]),
                                         interpolation=cv2.INTER_LINEAR)
                    tensor = self.transform(resized)
                    batch_tensors.append(tensor)
                    valid_indices.append(i)

            if not batch_tensors:
                return [self.get_zero_feature() for _ in images]

            batch = torch.stack(batch_tensors)

            with torch.no_grad():
                batch_features = self.model(batch).numpy()

            # 결과 조합
            outputs = [self.get_zero_feature() for _ in images]
            for i, valid_idx in enumerate(valid_indices):
                features = self.normalize(batch_features[i])
                outputs[valid_idx] = FeatureOutput(
                    feature=features.astype(np.float32),
                    feature_type=self.feature_type,
                    confidence=1.0
                )

            return outputs

        except Exception as e:
            self.logger.warning(f"MobileNet batch extraction failed: {e}")
            return [self.get_zero_feature() for _ in images]


class AdaptiveAppearanceExtractor(FeatureExtractor):
    """
    적응형 외형 특징 추출 - 상황에 따라 경량/정밀 모델 선택

    - 기본: Histogram (빠름)
    - ID 스위칭 의심시/새 트랙: MobileNet (정확)
    """

    def __init__(self, config: FeatureConfig = None,
                 use_deep_model: bool = True,
                 switch_threshold: float = 0.3,
                 stable_frames: int = 5):
        config = config or FeatureConfig(name="adaptive_appearance")
        super().__init__(config)

        self.use_deep_model = use_deep_model and TORCH_AVAILABLE
        self.switch_threshold = switch_threshold
        self.stable_frames = stable_frames

        # 추출기 초기화
        self.fast_extractor = HistogramFeatureExtractor()
        self.deep_extractor = None

        if self.use_deep_model:
            try:
                self.deep_extractor = MobileNetFeatureExtractor()
                self.deep_extractor.initialize()
            except Exception as e:
                self.logger.warning(f"Deep model init failed: {e}")
                self.deep_extractor = None

        # 특징 차원 계산
        self._feature_dim = self.fast_extractor.feature_dim
        if self.deep_extractor:
            self._feature_dim += self.deep_extractor.feature_dim

        # 트랙별 상태
        self._track_states: Dict[int, Dict] = {}

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def feature_type(self) -> str:
        return "appearance"

    def _should_use_deep(self, track_id: int, fast_feature: np.ndarray) -> bool:
        """딥러닝 모델 사용 여부 결정"""
        if self.deep_extractor is None:
            return False

        if track_id not in self._track_states:
            self._track_states[track_id] = {
                'prev_feature': fast_feature,
                'stable_count': 0
            }
            return True  # 새 트랙은 deep model 사용

        state = self._track_states[track_id]

        if state['prev_feature'] is not None:
            similarity = float(np.dot(fast_feature, state['prev_feature']))

            if similarity < self.switch_threshold:
                # 급격한 변화 = ID 스위칭 의심
                state['stable_count'] = 0
                state['prev_feature'] = fast_feature
                return True
            else:
                state['stable_count'] += 1

        state['prev_feature'] = fast_feature
        return state['stable_count'] < self.stable_frames

    def extract(self, image: np.ndarray, context: TrackContext = None) -> FeatureOutput:
        if image is None or image.size == 0:
            return self.get_zero_feature()

        track_id = context.track_id if context else None

        # Fast feature (항상)
        fast_output = self.fast_extractor.extract(image, context)
        fast_feat = fast_output.feature

        # Deep feature (조건부)
        if self.deep_extractor:
            if track_id is None or self._should_use_deep(track_id, fast_feat):
                deep_output = self.deep_extractor.extract(image, context)
                deep_feat = deep_output.feature
                used_deep = True
            else:
                deep_feat = np.zeros(self.deep_extractor.feature_dim, dtype=np.float32)
                used_deep = False

            combined = np.concatenate([fast_feat, deep_feat])
        else:
            combined = fast_feat
            used_deep = False

        combined = self.normalize(combined)

        return FeatureOutput(
            feature=combined,
            feature_type=self.feature_type,
            confidence=1.0 if used_deep else 0.8,
            metadata={'used_deep_model': used_deep}
        )

    def reset_state(self, track_ids: set = None):
        if track_ids is None:
            self._track_states.clear()
        else:
            active_ids = set(self._track_states.keys())
            for tid in active_ids - track_ids:
                del self._track_states[tid]


class ColorLayoutExtractor(FeatureExtractor):
    """
    Color Layout Descriptor (CLD) - MPEG-7 표준

    공간적 색상 분포를 DCT 계수로 표현
    """

    def __init__(self, config: FeatureConfig = None,
                 grid_size: int = 8,
                 num_coefficients: int = 12):
        config = config or FeatureConfig(name="color_layout")
        super().__init__(config)

        self.grid_size = grid_size
        self.num_coefficients = num_coefficients
        # Y, Cb, Cr 각각 num_coefficients개 = 3 * 12 = 36
        self._feature_dim = 3 * num_coefficients

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def feature_type(self) -> str:
        return "appearance"

    def extract(self, image: np.ndarray, context: TrackContext = None) -> FeatureOutput:
        if image is None or image.size == 0:
            return self.get_zero_feature()

        try:
            # YCrCb 변환
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

            # 그리드로 분할하고 평균 계산
            h, w = image.shape[:2]
            cell_h, cell_w = h // self.grid_size, w // self.grid_size

            if cell_h == 0 or cell_w == 0:
                return self.get_zero_feature()

            features = []
            for channel in range(3):
                grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        cell = ycrcb[i * cell_h:(i + 1) * cell_h,
                                    j * cell_w:(j + 1) * cell_w, channel]
                        grid[i, j] = np.mean(cell)

                # DCT 변환
                dct = cv2.dct(grid)

                # 지그재그 스캔으로 저주파 계수 추출
                coeffs = self._zigzag_scan(dct, self.num_coefficients)
                features.extend(coeffs)

            combined = np.array(features, dtype=np.float32)
            combined = self.normalize(combined)

            return FeatureOutput(
                feature=combined,
                feature_type=self.feature_type,
                confidence=1.0
            )

        except Exception as e:
            self.logger.warning(f"CLD extraction failed: {e}")
            return self.get_zero_feature()

    def _zigzag_scan(self, matrix: np.ndarray, n: int) -> List[float]:
        """DCT 계수의 지그재그 스캔"""
        rows, cols = matrix.shape
        result = []

        for s in range(rows + cols - 1):
            if s < rows:
                i, j = s, 0
            else:
                i, j = rows - 1, s - rows + 1

            while i >= 0 and j < cols and len(result) < n:
                result.append(float(matrix[i, j]))
                i -= 1
                j += 1

            if len(result) >= n:
                break

        return result[:n]
