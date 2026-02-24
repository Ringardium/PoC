"""
Extensible Feature Extraction Framework

다양한 특징(외형, 행동, 모션 등)을 플러그인 방식으로 추가할 수 있는 확장 가능한 프레임워크

Architecture:
    FeatureExtractor (base)
        ├── Appearance: HistogramFeatureExtractor, MobileNetFeatureExtractor, DINOv2FeatureExtractor, AdaptiveAppearanceExtractor
        ├── Motion: MotionFeatureExtractor, OpticalFlowFeatureExtractor, TrajectoryFeatureExtractor
        └── (extensible: Behavior, Audio, etc.)

    FeatureFusionEngine
        - 여러 추출기를 조합하여 통합 특징 생성
        - 다양한 Fusion 전략 지원 (weighted_concat, attention, adaptive)

    MatchingStrategy
        - 특징 기반 매칭 전략
        - CosineSimilarity, Euclidean, Cascade, Greedy 등

    TrackEventBus
        - 트랙 상태 변화 이벤트 시스템
        - Observer 패턴으로 확장 가능
"""

from .base import (
    FeatureExtractor,
    FeatureConfig,
    TrackContext,
    FeatureOutput,
    CompositeFeatureExtractor
)

from .appearance import (
    HistogramFeatureExtractor,
    MobileNetFeatureExtractor,
    DINOv2FeatureExtractor,
    AdaptiveAppearanceExtractor,
    ColorLayoutExtractor
)

from .motion import (
    MotionFeatureExtractor,
    OpticalFlowFeatureExtractor,
    TrajectoryFeatureExtractor
)

from .behavior import (
    ActivityLevelExtractor,
    PostureFeatureExtractor,
    InteractionFeatureExtractor,
    BehaviorPatternExtractor
)

from .fusion import (
    FeatureFusionEngine,
    FusionConfig,
    FusionStrategy,
    WeightedConcatFusion,
    AttentionFusion,
    AdaptiveFusion,
    create_default_fusion_engine
)

from .matching import (
    MatchingStrategy,
    MatchResult,
    CosineSimilarityMatching,
    EuclideanDistanceMatching,
    CascadeMatching,
    WeightedFeatureMatching,
    GreedyMatching,
    create_matching_strategy
)

from .events import (
    TrackEventType,
    TrackEvent,
    TrackEventHandler,
    TrackEventBus,
    TrackStateManager,
    LoggingEventHandler,
    IDSwitchHandler,
    OcclusionHandler,
    BehaviorEventHandler
)

__all__ = [
    # Base
    'FeatureExtractor',
    'FeatureConfig',
    'TrackContext',
    'FeatureOutput',
    'CompositeFeatureExtractor',

    # Appearance
    'HistogramFeatureExtractor',
    'MobileNetFeatureExtractor',
    'DINOv2FeatureExtractor',
    'AdaptiveAppearanceExtractor',
    'ColorLayoutExtractor',

    # Motion
    'MotionFeatureExtractor',
    'OpticalFlowFeatureExtractor',
    'TrajectoryFeatureExtractor',

    # Behavior
    'ActivityLevelExtractor',
    'PostureFeatureExtractor',
    'InteractionFeatureExtractor',
    'BehaviorPatternExtractor',

    # Fusion
    'FeatureFusionEngine',
    'FusionConfig',
    'FusionStrategy',
    'WeightedConcatFusion',
    'AttentionFusion',
    'AdaptiveFusion',
    'create_default_fusion_engine',

    # Matching
    'MatchingStrategy',
    'MatchResult',
    'CosineSimilarityMatching',
    'EuclideanDistanceMatching',
    'CascadeMatching',
    'WeightedFeatureMatching',
    'GreedyMatching',
    'create_matching_strategy',

    # Events
    'TrackEventType',
    'TrackEvent',
    'TrackEventHandler',
    'TrackEventBus',
    'TrackStateManager',
    'LoggingEventHandler',
    'IDSwitchHandler',
    'OcclusionHandler',
    'BehaviorEventHandler',
]
