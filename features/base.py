"""
Feature Extractor Base Classes

확장 가능한 특징 추출기의 기본 인터페이스 정의
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import logging


@dataclass
class FeatureConfig:
    """특징 추출기 설정"""
    name: str
    enabled: bool = True
    weight: float = 1.0  # Fusion 시 가중치
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError("weight must be non-negative")


@dataclass
class TrackContext:
    """트랙 컨텍스트 정보 - 특징 추출에 필요한 추가 정보"""
    track_id: int
    frame_idx: int = 0
    bbox: Tuple[float, float, float, float] = None  # (x, y, w, h) or (x1, y1, x2, y2)
    confidence: float = 1.0
    class_id: int = -1
    prev_bboxes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    prev_features: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureOutput:
    """특징 추출 결과"""
    feature: np.ndarray
    feature_type: str  # 'appearance', 'motion', 'behavior', etc.
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return len(self.feature) if self.feature is not None else 0

    @property
    def is_valid(self) -> bool:
        return self.feature is not None and len(self.feature) > 0


class FeatureExtractor(ABC):
    """
    특징 추출기 기본 인터페이스

    모든 특징 추출기(외형, 모션, 행동 등)는 이 클래스를 상속해야 함
    """

    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig(name=self.__class__.__name__)
        self._initialized = False
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """특징 벡터 차원"""
        pass

    @property
    @abstractmethod
    def feature_type(self) -> str:
        """특징 타입 (appearance, motion, behavior, etc.)"""
        pass

    @property
    def name(self) -> str:
        """추출기 이름"""
        return self.config.name

    @property
    def weight(self) -> float:
        """Fusion 가중치"""
        return self.config.weight

    @abstractmethod
    def extract(self, image: np.ndarray, context: TrackContext = None) -> FeatureOutput:
        """
        단일 이미지에서 특징 추출

        Args:
            image: 크롭된 이미지 (BGR)
            context: 트랙 컨텍스트 (선택적)

        Returns:
            FeatureOutput 객체
        """
        pass

    def extract_batch(self, images: List[np.ndarray],
                     contexts: List[TrackContext] = None) -> List[FeatureOutput]:
        """
        배치 특징 추출 (기본 구현 - 개별 호출)

        서브클래스에서 최적화된 배치 처리 구현 가능
        """
        if contexts is None:
            contexts = [None] * len(images)

        return [self.extract(img, ctx) for img, ctx in zip(images, contexts)]

    def initialize(self) -> bool:
        """
        추출기 초기화 (모델 로드 등)

        Returns:
            초기화 성공 여부
        """
        self._initialized = True
        return True

    def cleanup(self):
        """리소스 정리"""
        self._initialized = False

    def reset_state(self, track_ids: set = None):
        """
        상태 초기화 (특정 트랙 또는 전체)

        일부 추출기는 트랙별 상태를 유지할 수 있음 (예: 모션 히스토리)
        """
        pass

    def update_track_state(self, track_id: int, feature: FeatureOutput):
        """트랙별 상태 업데이트 (선택적)"""
        pass

    def get_zero_feature(self) -> FeatureOutput:
        """빈/실패 시 반환할 제로 특징"""
        return FeatureOutput(
            feature=np.zeros(self.feature_dim, dtype=np.float32),
            feature_type=self.feature_type,
            confidence=0.0
        )

    @staticmethod
    def normalize(feature: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """L2 정규화"""
        if feature is None:
            return None
        norm = np.linalg.norm(feature)
        if norm > eps:
            return feature / norm
        return feature

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.feature_dim}, type={self.feature_type})"


class CompositeFeatureExtractor(FeatureExtractor):
    """
    복합 특징 추출기 - 여러 추출기를 조합

    단일 추출기처럼 사용하되 내부적으로 여러 특징을 결합
    """

    def __init__(self, extractors: List[FeatureExtractor],
                 config: FeatureConfig = None):
        super().__init__(config)
        self.extractors = extractors
        self._feature_dim = sum(e.feature_dim for e in extractors)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def feature_type(self) -> str:
        return "composite"

    def extract(self, image: np.ndarray, context: TrackContext = None) -> FeatureOutput:
        features = []
        total_confidence = 0.0
        total_weight = 0.0

        for extractor in self.extractors:
            if extractor.config.enabled:
                output = extractor.extract(image, context)
                if output.is_valid:
                    features.append(output.feature * extractor.weight)
                    total_confidence += output.confidence * extractor.weight
                    total_weight += extractor.weight
                else:
                    features.append(np.zeros(extractor.feature_dim, dtype=np.float32))

        if not features:
            return self.get_zero_feature()

        combined = np.concatenate(features)
        combined = self.normalize(combined)

        avg_confidence = total_confidence / total_weight if total_weight > 0 else 0.0

        return FeatureOutput(
            feature=combined,
            feature_type=self.feature_type,
            confidence=avg_confidence,
            metadata={'extractors': [e.name for e in self.extractors]}
        )

    def initialize(self) -> bool:
        success = True
        for extractor in self.extractors:
            if not extractor.initialize():
                success = False
                self.logger.warning(f"Failed to initialize {extractor.name}")
        self._initialized = success
        return success

    def cleanup(self):
        for extractor in self.extractors:
            extractor.cleanup()
        super().cleanup()
