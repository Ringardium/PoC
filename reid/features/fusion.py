"""
Feature Fusion Engine

다양한 특징을 결합하여 통합 특징 벡터 생성
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from abc import ABC, abstractmethod

from .base import FeatureExtractor, FeatureConfig, FeatureOutput, TrackContext


@dataclass
class FusionConfig:
    """Fusion 설정"""
    strategy: str = 'weighted_concat'  # 'weighted_concat', 'attention', 'learned'
    normalize_output: bool = True
    fallback_on_missing: bool = True  # 일부 특징 실패 시 fallback
    min_confidence: float = 0.3  # 최소 신뢰도
    feature_weights: Dict[str, float] = field(default_factory=dict)


class FusionStrategy(ABC):
    """Fusion 전략 인터페이스"""

    @abstractmethod
    def fuse(self, features: Dict[str, FeatureOutput],
             weights: Dict[str, float] = None) -> FeatureOutput:
        """
        여러 특징을 하나로 융합

        Args:
            features: 추출기 이름 -> FeatureOutput 딕셔너리
            weights: 추출기별 가중치

        Returns:
            융합된 FeatureOutput
        """
        pass


class WeightedConcatFusion(FusionStrategy):
    """가중 연결 융합 - 각 특징에 가중치 적용 후 연결"""

    def fuse(self, features: Dict[str, FeatureOutput],
             weights: Dict[str, float] = None) -> FeatureOutput:
        if not features:
            return FeatureOutput(
                feature=np.array([], dtype=np.float32),
                feature_type='fused',
                confidence=0.0
            )

        weights = weights or {}
        weighted_features = []
        total_confidence = 0.0
        total_weight = 0.0
        metadata = {}

        for name, output in features.items():
            if not output.is_valid:
                continue

            weight = weights.get(name, 1.0)
            weighted_feat = output.feature * weight
            weighted_features.append(weighted_feat)

            total_confidence += output.confidence * weight
            total_weight += weight
            metadata[name] = {'dim': output.dim, 'confidence': output.confidence}

        if not weighted_features:
            return FeatureOutput(
                feature=np.array([], dtype=np.float32),
                feature_type='fused',
                confidence=0.0
            )

        combined = np.concatenate(weighted_features)

        # L2 정규화
        norm = np.linalg.norm(combined)
        if norm > 1e-7:
            combined = combined / norm

        avg_confidence = total_confidence / total_weight if total_weight > 0 else 0.0

        return FeatureOutput(
            feature=combined.astype(np.float32),
            feature_type='fused',
            confidence=avg_confidence,
            metadata={'components': metadata, 'fusion_type': 'weighted_concat'}
        )


class AttentionFusion(FusionStrategy):
    """
    Attention 기반 융합

    각 특징의 신뢰도와 상황에 따라 동적으로 가중치 조절
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def fuse(self, features: Dict[str, FeatureOutput],
             weights: Dict[str, float] = None) -> FeatureOutput:
        if not features:
            return FeatureOutput(
                feature=np.array([], dtype=np.float32),
                feature_type='fused',
                confidence=0.0
            )

        weights = weights or {}

        # Attention score 계산 (신뢰도 기반)
        valid_features = {k: v for k, v in features.items() if v.is_valid}

        if not valid_features:
            return FeatureOutput(
                feature=np.array([], dtype=np.float32),
                feature_type='fused',
                confidence=0.0
            )

        # Softmax attention
        scores = {}
        for name, output in valid_features.items():
            base_weight = weights.get(name, 1.0)
            score = output.confidence * base_weight
            scores[name] = score

        # Temperature-scaled softmax
        score_values = np.array(list(scores.values()))
        exp_scores = np.exp(score_values / self.temperature)
        attention_weights = exp_scores / exp_scores.sum()

        weighted_features = []
        metadata = {}

        for i, (name, output) in enumerate(valid_features.items()):
            attn_weight = attention_weights[i]
            weighted_feat = output.feature * attn_weight
            weighted_features.append(weighted_feat)
            metadata[name] = {
                'dim': output.dim,
                'confidence': output.confidence,
                'attention': float(attn_weight)
            }

        combined = np.concatenate(weighted_features)
        norm = np.linalg.norm(combined)
        if norm > 1e-7:
            combined = combined / norm

        avg_confidence = sum(v.confidence * attention_weights[i]
                             for i, v in enumerate(valid_features.values()))

        return FeatureOutput(
            feature=combined.astype(np.float32),
            feature_type='fused',
            confidence=avg_confidence,
            metadata={'components': metadata, 'fusion_type': 'attention'}
        )


class AdaptiveFusion(FusionStrategy):
    """
    적응형 융합 - 상황에 따라 융합 전략 변경

    - 외형 특징이 불안정하면 모션 가중치 증가
    - 정지 상태면 외형 가중치 증가
    """

    def __init__(self, appearance_types: List[str] = None,
                 motion_types: List[str] = None):
        self.appearance_types = appearance_types or ['appearance']
        self.motion_types = motion_types or ['motion']

    def fuse(self, features: Dict[str, FeatureOutput],
             weights: Dict[str, float] = None) -> FeatureOutput:
        if not features:
            return FeatureOutput(
                feature=np.array([], dtype=np.float32),
                feature_type='fused',
                confidence=0.0
            )

        weights = dict(weights) if weights else {}

        # 외형/모션 특징 분류
        appearance_features = {k: v for k, v in features.items()
                               if v.feature_type in self.appearance_types}
        motion_features = {k: v for k, v in features.items()
                           if v.feature_type in self.motion_types}

        # 외형 특징 신뢰도 평가
        appearance_confidence = np.mean([v.confidence for v in appearance_features.values()]) \
            if appearance_features else 0.0

        # 모션 특징 신뢰도 평가
        motion_confidence = np.mean([v.confidence for v in motion_features.values()]) \
            if motion_features else 0.0

        # 적응형 가중치 조절
        if appearance_confidence < 0.5 and motion_confidence > 0.5:
            # 외형 불안정, 모션 안정 -> 모션 가중치 증가
            for name in motion_features:
                weights[name] = weights.get(name, 1.0) * 1.5
            for name in appearance_features:
                weights[name] = weights.get(name, 1.0) * 0.7

        elif motion_confidence < 0.3:
            # 정지 상태 -> 외형 가중치 증가
            for name in appearance_features:
                weights[name] = weights.get(name, 1.0) * 1.3

        # Weighted concat 사용
        concat_fusion = WeightedConcatFusion()
        result = concat_fusion.fuse(features, weights)
        result.metadata['fusion_type'] = 'adaptive'
        result.metadata['adapted_weights'] = weights

        return result


class FeatureFusionEngine:
    """
    특징 융합 엔진

    여러 특징 추출기를 관리하고 결과를 융합
    """

    def __init__(self, config: FusionConfig = None):
        self.config = config or FusionConfig()
        self.extractors: Dict[str, FeatureExtractor] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # Fusion 전략 선택
        self.fusion_strategy = self._create_fusion_strategy()

        # 통계
        self._stats = defaultdict(lambda: {'calls': 0, 'failures': 0, 'avg_time': 0.0})

    def _create_fusion_strategy(self) -> FusionStrategy:
        """Fusion 전략 생성"""
        strategy = self.config.strategy

        if strategy == 'weighted_concat':
            return WeightedConcatFusion()
        elif strategy == 'attention':
            return AttentionFusion()
        elif strategy == 'adaptive':
            return AdaptiveFusion()
        else:
            self.logger.warning(f"Unknown strategy '{strategy}', using weighted_concat")
            return WeightedConcatFusion()

    def register_extractor(self, name: str, extractor: FeatureExtractor,
                          weight: float = 1.0):
        """특징 추출기 등록"""
        self.extractors[name] = extractor
        self.config.feature_weights[name] = weight
        self.logger.info(f"Registered extractor: {name} (dim={extractor.feature_dim}, weight={weight})")

    def remove_extractor(self, name: str):
        """특징 추출기 제거"""
        if name in self.extractors:
            del self.extractors[name]
            if name in self.config.feature_weights:
                del self.config.feature_weights[name]

    @property
    def total_feature_dim(self) -> int:
        """전체 특징 차원"""
        return sum(e.feature_dim for e in self.extractors.values()
                   if e.config.enabled)

    @property
    def enabled_extractors(self) -> Dict[str, FeatureExtractor]:
        """활성화된 추출기들"""
        return {name: e for name, e in self.extractors.items()
                if e.config.enabled}

    def initialize(self) -> bool:
        """모든 추출기 초기화"""
        success = True
        for name, extractor in self.extractors.items():
            if not extractor.initialize():
                self.logger.warning(f"Failed to initialize {name}")
                success = False
        return success

    def extract(self, image: np.ndarray, context: TrackContext = None) -> FeatureOutput:
        """
        모든 활성 추출기로 특징 추출 후 융합

        Args:
            image: 입력 이미지
            context: 트랙 컨텍스트

        Returns:
            융합된 FeatureOutput
        """
        features = {}

        for name, extractor in self.enabled_extractors.items():
            try:
                output = extractor.extract(image, context)

                if output.is_valid or self.config.fallback_on_missing:
                    if output.confidence >= self.config.min_confidence or self.config.fallback_on_missing:
                        features[name] = output

                self._stats[name]['calls'] += 1

            except Exception as e:
                self.logger.warning(f"Extractor {name} failed: {e}")
                self._stats[name]['failures'] += 1

                if self.config.fallback_on_missing:
                    features[name] = extractor.get_zero_feature()

        result = self.fusion_strategy.fuse(features, self.config.feature_weights)

        if self.config.normalize_output and result.is_valid:
            result.feature = FeatureExtractor.normalize(result.feature)

        return result

    def extract_batch(self, images: List[np.ndarray],
                      contexts: List[TrackContext] = None) -> List[FeatureOutput]:
        """배치 특징 추출 및 융합"""
        if contexts is None:
            contexts = [None] * len(images)

        # 각 추출기별로 배치 처리
        all_features: Dict[str, List[FeatureOutput]] = {}

        for name, extractor in self.enabled_extractors.items():
            try:
                outputs = extractor.extract_batch(images, contexts)
                all_features[name] = outputs
            except Exception as e:
                self.logger.warning(f"Batch extraction failed for {name}: {e}")
                all_features[name] = [extractor.get_zero_feature() for _ in images]

        # 이미지별로 융합
        results = []
        for i in range(len(images)):
            features = {name: outputs[i] for name, outputs in all_features.items()}
            fused = self.fusion_strategy.fuse(features, self.config.feature_weights)

            if self.config.normalize_output and fused.is_valid:
                fused.feature = FeatureExtractor.normalize(fused.feature)

            results.append(fused)

        return results

    def reset_state(self, track_ids: set = None):
        """모든 추출기 상태 초기화"""
        for extractor in self.extractors.values():
            extractor.reset_state(track_ids)

    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        return {
            'extractors': dict(self._stats),
            'total_dim': self.total_feature_dim,
            'enabled_count': len(self.enabled_extractors)
        }

    def cleanup(self):
        """리소스 정리"""
        for extractor in self.extractors.values():
            extractor.cleanup()


def create_default_fusion_engine(include_motion: bool = True,
                                 include_deep: bool = True) -> FeatureFusionEngine:
    """
    기본 설정의 Fusion 엔진 생성

    Args:
        include_motion: 모션 특징 포함 여부
        include_deep: 딥러닝 특징 포함 여부

    Returns:
        설정된 FeatureFusionEngine
    """
    from .appearance import HistogramFeatureExtractor, AdaptiveAppearanceExtractor
    from .motion import MotionFeatureExtractor, TrajectoryFeatureExtractor

    config = FusionConfig(
        strategy='adaptive',
        normalize_output=True
    )

    engine = FeatureFusionEngine(config)

    # 외형 특징
    if include_deep:
        engine.register_extractor('appearance',
                                  AdaptiveAppearanceExtractor(),
                                  weight=1.0)
    else:
        engine.register_extractor('histogram',
                                  HistogramFeatureExtractor(),
                                  weight=1.0)

    # 모션 특징
    if include_motion:
        engine.register_extractor('motion',
                                  MotionFeatureExtractor(),
                                  weight=0.5)
        engine.register_extractor('trajectory',
                                  TrajectoryFeatureExtractor(),
                                  weight=0.3)

    engine.initialize()

    return engine
