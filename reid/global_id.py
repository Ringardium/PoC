"""
Global ID Manager - 멀티채널/단일채널 환경에서 객체의 전역 ID 관리
확장 가능한 다중 특징 지원 버전

주요 기능:
1. 다중 특징 타입 지원 (외형, 모션, 행동 등)
2. 특징 타입별 가중치 적용
3. 이벤트 시스템 통합 (선택적)
4. 갤러리 관리 전략 플러그인화
5. 기존 API 하위 호환성 유지
"""

import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from abc import ABC, abstractmethod
import time
import logging
import threading


# === 갤러리 클래스 ===

@dataclass
class ObjectGallery:
    """
    객체별 특징 갤러리 (하위 호환용)

    단일 특징 타입만 사용하는 기존 코드와 호환
    """
    global_id: int
    features: deque = field(default_factory=lambda: deque(maxlen=10))
    representative_feature: np.ndarray = None
    last_seen: float = 0.0
    total_appearances: int = 0
    channels_seen: set = field(default_factory=set)

    def update(self, feature: np.ndarray, channel_id: str):
        """특징 업데이트 및 대표 특징 갱신"""
        if feature is None:
            return

        self.features.append(feature)
        self.last_seen = time.time()
        self.total_appearances += 1
        self.channels_seen.add(channel_id)

        # EMA 기반 대표 특징 업데이트 (alpha=0.3)
        if self.representative_feature is None:
            self.representative_feature = feature.copy()
        else:
            alpha = 0.3
            self.representative_feature = alpha * feature + (1 - alpha) * self.representative_feature
            norm = np.linalg.norm(self.representative_feature)
            if norm > 1e-7:
                self.representative_feature /= norm


@dataclass
class MultiFeatureGallery:
    """
    다중 특징 타입을 지원하는 객체 갤러리

    각 특징 타입별로 별도 저장 및 대표 특징 관리
    """
    global_id: int
    features: Dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=10)))
    representatives: Dict[str, np.ndarray] = field(default_factory=dict)
    feature_weights: Dict[str, float] = field(default_factory=dict)

    last_seen: float = 0.0
    total_appearances: int = 0
    channels_seen: set = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 하위 호환용 프로퍼티
    @property
    def representative_feature(self) -> Optional[np.ndarray]:
        """기존 코드 호환 - appearance 특징 반환"""
        return self.representatives.get('appearance')

    def update(self, feature_type: str, feature: np.ndarray,
               channel_id: str, alpha: float = 0.3, weight: float = 1.0):
        """특정 타입의 특징 업데이트"""
        if feature is None or len(feature) == 0:
            return

        self.features[feature_type].append(feature)
        self.feature_weights[feature_type] = weight
        self.last_seen = time.time()
        self.total_appearances += 1
        self.channels_seen.add(channel_id)

        # EMA 대표 특징 업데이트
        if feature_type not in self.representatives:
            self.representatives[feature_type] = feature.copy()
        else:
            self.representatives[feature_type] = (
                alpha * feature + (1 - alpha) * self.representatives[feature_type]
            )
            norm = np.linalg.norm(self.representatives[feature_type])
            if norm > 1e-7:
                self.representatives[feature_type] /= norm

    def update_single(self, feature: np.ndarray, channel_id: str):
        """단일 특징 업데이트 (하위 호환)"""
        self.update('appearance', feature, channel_id)

    def get_combined_representative(self) -> Optional[np.ndarray]:
        """모든 대표 특징을 가중 결합"""
        if not self.representatives:
            return None

        weighted_feats = []
        total_weight = 0.0

        for feat_type, rep in self.representatives.items():
            weight = self.feature_weights.get(feat_type, 1.0)
            weighted_feats.append(rep * weight)
            total_weight += weight

        if total_weight == 0:
            return None

        combined = np.concatenate(weighted_feats)
        norm = np.linalg.norm(combined)
        if norm > 1e-7:
            combined /= norm

        return combined


# === 매칭 전략 ===

class GalleryMatchingStrategy(ABC):
    """갤러리 매칭 전략 인터페이스"""

    @abstractmethod
    def match(self, query_features: Dict[str, np.ndarray],
              galleries: Dict[int, MultiFeatureGallery],
              threshold: float,
              exclude_ids: set = None) -> Tuple[Optional[int], float]:
        pass


class WeightedMultiFeatureMatching(GalleryMatchingStrategy):
    """가중 다중 특징 매칭"""

    def __init__(self, feature_weights: Dict[str, float] = None):
        self.feature_weights = feature_weights or {}

    def match(self, query_features: Dict[str, np.ndarray],
              galleries: Dict[int, MultiFeatureGallery],
              threshold: float,
              exclude_ids: set = None) -> Tuple[Optional[int], float]:

        exclude_ids = exclude_ids or set()
        best_match = None
        best_sim = 0.0

        for gid, gallery in galleries.items():
            if gid in exclude_ids:
                continue

            if not gallery.representatives:
                continue

            total_sim = 0.0
            total_weight = 0.0

            for feat_type, query_feat in query_features.items():
                if feat_type not in gallery.representatives:
                    continue

                gallery_feat = gallery.representatives[feat_type]
                sim = float(np.dot(query_feat, gallery_feat))

                weight = self.feature_weights.get(feat_type, 1.0)
                total_sim += sim * weight
                total_weight += weight

            if total_weight > 0:
                avg_sim = total_sim / total_weight

                if avg_sim > best_sim:
                    best_sim = avg_sim
                    best_match = gid

        if best_sim >= threshold:
            return best_match, best_sim

        return None, 0.0


class SingleFeatureMatching(GalleryMatchingStrategy):
    """단일 특징 매칭 (하위 호환)"""

    def match(self, query_features: Dict[str, np.ndarray],
              galleries: Dict[int, MultiFeatureGallery],
              threshold: float,
              exclude_ids: set = None) -> Tuple[Optional[int], float]:

        exclude_ids = exclude_ids or set()

        # appearance 특징 사용
        query_feat = query_features.get('appearance')
        if query_feat is None:
            # 첫 번째 특징 사용
            if query_features:
                query_feat = list(query_features.values())[0]
            else:
                return None, 0.0

        best_match = None
        best_sim = 0.0

        for gid, gallery in galleries.items():
            if gid in exclude_ids:
                continue

            gallery_feat = gallery.representative_feature
            if gallery_feat is None:
                continue

            sim = float(np.dot(query_feat, gallery_feat))

            if sim > best_sim:
                best_sim = sim
                best_match = gid

        if best_sim >= threshold:
            return best_match, best_sim

        return None, 0.0


# === 설정 ===

@dataclass
class GlobalIDManagerConfig:
    """Global ID Manager 설정"""
    similarity_threshold: float = 0.6
    max_gallery_size: int = 100
    inactive_timeout: float = 300.0
    ema_alpha: float = 0.3
    matching_strategy: str = 'weighted'  # 'weighted', 'single'
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        'appearance': 1.0,
        'motion': 0.5,
        'behavior': 0.3
    })
    enable_events: bool = False


# === 메인 클래스 ===

class GlobalIDManager:
    """
    전역 ID 관리자 - Local ID를 Global ID로 매핑

    Features:
    - 다중 특징 타입 지원 (외형, 모션, 행동 등)
    - 특징 타입별 가중치
    - 플러그인 가능한 매칭 전략
    - 채널 간 객체 추적 (멀티채널)
    - 기존 API 하위 호환성 유지

    Usage (기존 방식 - 하위 호환):
        manager = GlobalIDManager(feature_dim=128)
        gid = manager.get_global_id("cam1", 1, feature_vector)

    Usage (새로운 방식 - 다중 특징):
        manager = GlobalIDManager()
        gid = manager.get_global_id_multi("cam1", 1, {
            'appearance': appearance_feat,
            'motion': motion_feat
        })
    """

    def __init__(self, feature_dim: int = 96, similarity_threshold: float = 0.6,
                 max_gallery_size: int = 100, inactive_timeout: float = 300.0,
                 config: GlobalIDManagerConfig = None):
        """
        Args:
            feature_dim: 특징 벡터 차원 (하위 호환)
            similarity_threshold: 동일 객체 판단 임계값
            max_gallery_size: 최대 갤러리 크기
            inactive_timeout: 비활성 객체 제거 시간 (초)
            config: 상세 설정 (제공시 다른 인자 무시)
        """
        if config:
            self.config = config
        else:
            self.config = GlobalIDManagerConfig(
                similarity_threshold=similarity_threshold,
                max_gallery_size=max_gallery_size,
                inactive_timeout=inactive_timeout
            )

        self.feature_dim = feature_dim
        self.similarity_threshold = self.config.similarity_threshold
        self.max_gallery_size = self.config.max_gallery_size
        self.inactive_timeout = self.config.inactive_timeout

        self.logger = logging.getLogger(self.__class__.__name__)

        # Global ID 카운터
        self._next_global_id = 1
        self._lock = threading.Lock()

        # 객체 갤러리: global_id -> MultiFeatureGallery
        self.galleries: Dict[int, MultiFeatureGallery] = {}

        # 로컬 -> 글로벌 매핑: (channel_id, local_id) -> global_id
        self.local_to_global: Dict[Tuple[str, int], int] = {}

        # 글로벌 -> 로컬 역매핑
        self.global_to_local: Dict[int, List[Tuple[str, int]]] = defaultdict(list)

        # 매칭 전략
        self.matching_strategy = self._create_matching_strategy()

        # 이벤트 버스 (선택적)
        self.event_bus = None

        # 통계
        self.stats = {
            'total_matches': 0,
            'new_objects': 0,
            'cross_channel_matches': 0
        }

        self.logger.info(f"GlobalIDManager 초기화 - threshold: {self.similarity_threshold}")

    def _create_matching_strategy(self) -> GalleryMatchingStrategy:
        """매칭 전략 생성"""
        if self.config.matching_strategy == 'weighted':
            return WeightedMultiFeatureMatching(self.config.feature_weights)
        else:
            return SingleFeatureMatching()

    def set_matching_strategy(self, strategy: GalleryMatchingStrategy):
        """매칭 전략 변경"""
        self.matching_strategy = strategy

    def set_event_bus(self, event_bus):
        """이벤트 버스 설정"""
        self.event_bus = event_bus

    def _get_next_global_id(self) -> int:
        """스레드 안전한 글로벌 ID 생성"""
        with self._lock:
            gid = self._next_global_id
            self._next_global_id += 1
            return gid

    # === 하위 호환 API ===

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        if feat1 is None or feat2 is None:
            return 0.0
        return float(np.dot(feat1, feat2))

    def compute_similarity_batch(self, query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
        """배치 유사도 계산"""
        if query is None or gallery is None or len(gallery) == 0:
            return np.array([])
        return np.dot(gallery, query)

    def find_matching_gallery(self, feature: np.ndarray,
                              exclude_global_id: int = None) -> Tuple[Optional[int], float]:
        """갤러리에서 매칭되는 객체 찾기 (하위 호환)"""
        if feature is None or len(self.galleries) == 0:
            return None, 0.0

        features = {'appearance': feature}
        exclude_ids = {exclude_global_id} if exclude_global_id else set()

        # 활성 갤러리만 대상
        current_time = time.time()
        active_galleries = {
            gid: gallery for gid, gallery in self.galleries.items()
            if (current_time - gallery.last_seen) < self.inactive_timeout
        }

        return self.matching_strategy.match(
            features, active_galleries, self.similarity_threshold, exclude_ids
        )

    def get_global_id(self, channel_id: str, local_id: int,
                      feature: np.ndarray, box: tuple = None,
                      exclude_gids: set = None) -> int:
        """
        로컬 ID에 대한 글로벌 ID 반환 (하위 호환)

        Args:
            channel_id: 채널 식별자
            local_id: 로컬 트랙 ID
            feature: 특징 벡터
            box: 바운딩 박스 (선택)
            exclude_gids: 이번 프레임에서 이미 할당된 global_id (중복 방지)

        Returns:
            global_id: 전역 ID
        """
        features = {'appearance': feature} if feature is not None else {}
        return self._get_global_id_internal(channel_id, local_id, features, box, exclude_gids)

    # === 확장 API ===

    def get_global_id_multi(self, channel_id: str, local_id: int,
                            features: Dict[str, np.ndarray],
                            box: tuple = None) -> int:
        """
        다중 특징으로 글로벌 ID 획득

        Args:
            channel_id: 채널 식별자
            local_id: 로컬 트랙 ID
            features: 특징 타입 -> 특징 벡터 딕셔너리
            box: 바운딩 박스 (선택)

        Returns:
            global_id
        """
        return self._get_global_id_internal(channel_id, local_id, features, box)

    def _get_global_id_internal(self, channel_id: str, local_id: int,
                                 features: Dict[str, np.ndarray],
                                 box: tuple = None,
                                 exclude_gids: set = None) -> int:
        """내부 글로벌 ID 획득 로직"""
        key = (channel_id, local_id)

        # 이미 매핑된 경우
        if key in self.local_to_global:
            global_id = self.local_to_global[key]

            # 같은 프레임에서 다른 객체가 이미 이 global_id를 사용 중이면 재매핑
            if exclude_gids and global_id in exclude_gids:
                del self.local_to_global[key]
                self.logger.info(
                    f"Global ID 충돌: 채널 {channel_id} 로컬 {local_id} -> G:{global_id} 재매핑"
                )
            else:
                if global_id in self.galleries:
                    self._update_gallery(global_id, features, channel_id)
                return global_id

        # 새로운 로컬 ID - 갤러리에서 매칭 시도
        matched_gid, similarity = self._find_matching_gallery_multi(features, exclude_gids)

        if matched_gid is not None:
            global_id = matched_gid
            self.stats['total_matches'] += 1

            # 크로스 채널 매칭 확인
            existing_channels = self.galleries[global_id].channels_seen
            if channel_id not in existing_channels:
                self.stats['cross_channel_matches'] += 1
                self.logger.info(
                    f"크로스 채널 매칭: 채널 {channel_id}의 로컬 ID {local_id} -> "
                    f"글로벌 ID {global_id} (유사도: {similarity:.3f})"
                )

                if self.event_bus:
                    self._emit_event('TRACK_RECOVERED', global_id, {
                        'channel_id': channel_id,
                        'local_id': local_id,
                        'similarity': similarity,
                        'cross_channel': True
                    })
        else:
            # 새로운 객체
            global_id = self._get_next_global_id()
            self.galleries[global_id] = MultiFeatureGallery(global_id=global_id)
            self.stats['new_objects'] += 1

            if self.event_bus:
                self._emit_event('TRACK_CREATED', global_id, {
                    'channel_id': channel_id,
                    'local_id': local_id
                })

        # 매핑 저장
        self.local_to_global[key] = global_id
        self.global_to_local[global_id].append(key)

        # 갤러리 업데이트
        self._update_gallery(global_id, features, channel_id)

        return global_id

    def _find_matching_gallery_multi(self, features: Dict[str, np.ndarray],
                                      exclude_gids: set = None) -> Tuple[Optional[int], float]:
        """다중 특징으로 갤러리 매칭"""
        if not features:
            return None, 0.0

        current_time = time.time()
        active_galleries = {
            gid: gallery for gid, gallery in self.galleries.items()
            if (current_time - gallery.last_seen) < self.inactive_timeout
            and (exclude_gids is None or gid not in exclude_gids)
        }

        if not active_galleries:
            return None, 0.0

        return self.matching_strategy.match(
            features, active_galleries, self.similarity_threshold
        )

    def _update_gallery(self, global_id: int, features: Dict[str, np.ndarray],
                        channel_id: str):
        """갤러리 업데이트"""
        if global_id not in self.galleries:
            return

        gallery = self.galleries[global_id]

        for feat_type, feature in features.items():
            if feature is not None and len(feature) > 0:
                weight = self.config.feature_weights.get(feat_type, 1.0)
                gallery.update(feat_type, feature, channel_id,
                              alpha=self.config.ema_alpha, weight=weight)

    def _emit_event(self, event_type: str, track_id: int, data: dict):
        """이벤트 발행 (이벤트 버스 설정시)"""
        if self.event_bus:
            try:
                from features import TrackEvent, TrackEventType
                event = TrackEvent(
                    event_type=getattr(TrackEventType, event_type),
                    track_id=track_id,
                    data=data
                )
                self.event_bus.publish(event)
            except:
                pass

    # === 관리 메서드 ===

    def remove_local_id(self, channel_id: str, local_id: int):
        """로컬 ID 매핑 제거"""
        key = (channel_id, local_id)

        if key in self.local_to_global:
            global_id = self.local_to_global[key]
            del self.local_to_global[key]

            if global_id in self.global_to_local:
                self.global_to_local[global_id] = [
                    k for k in self.global_to_local[global_id] if k != key
                ]

    def cleanup_old_data(self):
        """오래된 데이터 정리"""
        current_time = time.time()

        # 비활성 갤러리 제거
        inactive_gids = [
            gid for gid, gallery in self.galleries.items()
            if (current_time - gallery.last_seen) > self.inactive_timeout
        ]

        for gid in inactive_gids:
            if gid in self.global_to_local:
                for key in self.global_to_local[gid]:
                    if key in self.local_to_global:
                        del self.local_to_global[key]
                del self.global_to_local[gid]

            del self.galleries[gid]

            if self.event_bus:
                self._emit_event('TRACK_DELETED', gid, {})

        # 갤러리 크기 제한
        if len(self.galleries) > self.max_gallery_size:
            sorted_gids = sorted(
                self.galleries.keys(),
                key=lambda gid: self.galleries[gid].last_seen
            )

            to_remove = sorted_gids[:len(self.galleries) - self.max_gallery_size]
            for gid in to_remove:
                if gid in self.global_to_local:
                    for key in self.global_to_local[gid]:
                        if key in self.local_to_global:
                            del self.local_to_global[key]
                    del self.global_to_local[gid]
                del self.galleries[gid]

        if inactive_gids:
            self.logger.debug(f"갤러리 정리: {len(inactive_gids)}개 제거, 현재 {len(self.galleries)}개")

    # === 조회 메서드 ===

    def get_stats(self) -> dict:
        """통계 반환"""
        return {
            **self.stats,
            'active_galleries': len(self.galleries),
            'total_mappings': len(self.local_to_global)
        }

    def get_cross_channel_objects(self) -> List[int]:
        """여러 채널에서 관찰된 객체 ID 반환"""
        return [
            gid for gid, gallery in self.galleries.items()
            if len(gallery.channels_seen) > 1
        ]

    def get_object_info(self, global_id: int) -> Optional[dict]:
        """특정 객체 정보 반환"""
        if global_id not in self.galleries:
            return None

        gallery = self.galleries[global_id]
        return {
            'global_id': global_id,
            'channels_seen': list(gallery.channels_seen),
            'total_appearances': gallery.total_appearances,
            'last_seen': gallery.last_seen,
            'feature_types': list(gallery.representatives.keys()),
            'feature_counts': {k: len(v) for k, v in gallery.features.items()}
        }


# === 테스트 ===

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== GlobalIDManager Test ===\n")

    # 기존 방식 테스트 (하위 호환)
    print("1. Legacy API Test:")
    manager = GlobalIDManager(feature_dim=128)

    feat1 = np.random.randn(128).astype(np.float32)
    feat1 /= np.linalg.norm(feat1)

    gid1 = manager.get_global_id("cam1", 1, feat1)
    print(f"   Channel 1, Local 1 -> Global {gid1}")

    # 유사한 특징으로 다른 채널에서 등록
    feat2 = feat1 + np.random.randn(128).astype(np.float32) * 0.1
    feat2 /= np.linalg.norm(feat2)

    gid2 = manager.get_global_id("cam2", 5, feat2)
    print(f"   Channel 2, Local 5 -> Global {gid2}")

    # 새로운 방식 테스트 (다중 특징)
    print("\n2. Multi-Feature API Test:")
    manager2 = GlobalIDManager(config=GlobalIDManagerConfig(
        feature_weights={'appearance': 1.0, 'motion': 0.5}
    ))

    appearance = np.random.randn(128).astype(np.float32)
    appearance /= np.linalg.norm(appearance)

    motion = np.random.randn(32).astype(np.float32)
    motion /= np.linalg.norm(motion)

    gid3 = manager2.get_global_id_multi("cam1", 1, {
        'appearance': appearance,
        'motion': motion
    })
    print(f"   Multi-feature -> Global {gid3}")

    print(f"\n3. Stats: {manager.get_stats()}")
    print(f"   Cross-channel objects: {manager.get_cross_channel_objects()}")
