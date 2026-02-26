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
    last_bbox: Optional[tuple] = None  # (x_center, y_center, w, h)
    is_registered: bool = False  # True for pet profile galleries — never expire
    # 등록 프로필 원본 특징 (드리프트 방지용 anchor)
    _anchor_features: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Size profile — accumulated bbox geometry for size-based matching
    size_history: deque = field(default_factory=lambda: deque(maxlen=50))
    # Cached running stats (updated on each update_size_profile call)
    _size_stats: Dict[str, float] = field(default_factory=dict)

    # 하위 호환용 프로퍼티
    @property
    def representative_feature(self) -> Optional[np.ndarray]:
        """기존 코드 호환 - appearance 특징 반환"""
        return self.representatives.get('appearance')

    def update(self, feature_type: str, feature: np.ndarray,
               channel_id: str, alpha: float = 0.3, weight: float = 1.0):
        """특정 타입의 특징 업데이트

        등록된 프로필(is_registered=True)은 representative를 고정하고
        deque에만 런타임 특징을 추가하여 EMA 드리프트를 방지.
        """
        if feature is None or len(feature) == 0:
            return

        self.features[feature_type].append(feature)
        self.feature_weights[feature_type] = weight
        self.last_seen = time.time()
        self.total_appearances += 1
        self.channels_seen.add(channel_id)

        # 등록된 프로필: representative 고정 (EMA 드리프트 방지)
        if self.is_registered:
            if feature_type not in self.representatives:
                self.representatives[feature_type] = feature.copy()
            # else: representative를 업데이트하지 않음 — 원본 ref 유지
            return

        # 비등록 객체: EMA 대표 특징 업데이트
        if feature_type not in self.representatives:
            self.representatives[feature_type] = feature.copy()
        else:
            self.representatives[feature_type] = (
                alpha * feature + (1 - alpha) * self.representatives[feature_type]
            )
            norm = np.linalg.norm(self.representatives[feature_type])
            if norm > 1e-7:
                self.representatives[feature_type] /= norm

    def update_size_profile(self, bbox: tuple):
        """bbox로부터 크기 프로필 업데이트.

        Args:
            bbox: (x_center, y_center, w, h) or (x1, y1, x2, y2)
        """
        if bbox is None or len(bbox) < 4:
            return

        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        # Detect format: if x2 > x1, treat as (x1,y1,x2,y2)
        if x2 > x1 and y2 > y1:
            w = x2 - x1
            h = y2 - y1
            y_center = (y1 + y2) / 2
        else:
            # (cx, cy, w, h) format
            w = x2 if x2 > 0 else 1
            h = y2 if y2 > 0 else 1
            y_center = y1

        aspect_ratio = w / max(h, 1.0)
        area = w * h

        self.size_history.append({
            'aspect_ratio': aspect_ratio,
            'area': area,
            'y_center': y_center,
        })

        # Recompute running stats
        if len(self.size_history) >= 3:
            ars = [s['aspect_ratio'] for s in self.size_history]
            areas = [s['area'] for s in self.size_history]
            self._size_stats = {
                'ar_mean': float(np.mean(ars)),
                'ar_std': float(np.std(ars)),
                'area_mean': float(np.mean(areas)),
                'area_std': float(np.std(areas)),
            }

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
    """가중 다중 특징 Top-K 매칭.

    갤러리의 저장된 특징 전부(최대 K개)와 비교하여 최고 유사도 사용.
    EMA 대표 특징만 쓰는 것보다 드리프트에 강건함.
    """

    def __init__(self, feature_weights: Dict[str, float] = None):
        self.feature_weights = feature_weights or {}

    def _gallery_similarity(self, query_feat: np.ndarray,
                            gallery: MultiFeatureGallery,
                            feat_type: str) -> float:
        """query와 갤러리 저장 특징 전부 비교, 최고 유사도 반환.

        등록 프로필의 anchor 특징도 비교 대상에 포함하여
        deque에서 원본이 밀려나도 안정적으로 매칭.
        """
        best = 0.0

        # anchor 특징 (등록 프로필 원본 — 드리프트 없음)
        anchor = gallery._anchor_features.get(feat_type)
        if anchor is not None:
            sim = float(np.dot(query_feat, anchor))
            if sim > best:
                best = sim

        # deque 저장 특징 (런타임 CCTV crop)
        stored = gallery.features.get(feat_type)
        if stored:
            for feat in stored:
                sim = float(np.dot(query_feat, feat))
                if sim > best:
                    best = sim

        # fallback to representative
        if best == 0.0:
            rep = gallery.representatives.get(feat_type)
            if rep is not None:
                best = float(np.dot(query_feat, rep))

        return best

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

            if not gallery.features and not gallery.representatives:
                continue

            total_sim = 0.0
            total_weight = 0.0

            for feat_type, query_feat in query_features.items():
                sim = self._gallery_similarity(query_feat, gallery, feat_type)
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
    similarity_threshold: float = 0.7
    max_gallery_size: int = 100
    inactive_timeout: float = 300.0
    ema_alpha: float = 0.1
    matching_strategy: str = 'weighted'  # 'weighted', 'single'
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        'appearance': 1.0,
        'motion': 0.5,
        'behavior': 0.3
    })
    enable_events: bool = False
    freeze_registered: bool = True  # True: 등록 프로필 특징 고정 (드리프트 방지), False: EMA 업데이트 허용


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

    def register_pet_profiles(self, profile_dir: str, feature_extractor=None):
        """PetProfileStore의 레퍼런스 이미지로 갤러리를 사전 등록.

        등록된 펫은 pet profile의 global_id를 그대로 사용하므로,
        런타임 매칭 시 pet_name이 올바르게 표시됩니다.

        Args:
            profile_dir: PetProfileStore 기본 디렉토리 (e.g. "references")
            feature_extractor: 특징 추출 함수 (crop_image -> np.ndarray).
                None이면 ColorHistogramReID를 사용.
        """
        try:
            from tools.pet_profiles import PetProfileStore
            store = PetProfileStore(profile_dir)
            profiles = store.list_pets()
            if not profiles:
                self.logger.info("등록된 펫 프로필 없음")
                return

            # 기본 특징 추출기
            if feature_extractor is None:
                from reid.extractor import ColorHistogramReID
                extractor = ColorHistogramReID()
                feature_extractor = extractor.extract_features

            import cv2
            from pathlib import Path
            base = Path(profile_dir)

            registered = 0
            for profile in profiles:
                gid = profile.global_id
                features_list = []
                for img_rel in profile.reference_images:
                    img_path = base / img_rel
                    if not img_path.exists():
                        continue
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    feat = feature_extractor(img)
                    if feat is not None and len(feat) > 0:
                        feat = feat / (np.linalg.norm(feat) + 1e-8)
                        features_list.append(feat)

                if not features_list:
                    continue

                # 대표 특징 = 평균
                rep_feature = np.mean(features_list, axis=0).astype(np.float32)
                rep_feature = rep_feature / (np.linalg.norm(rep_feature) + 1e-8)

                # 갤러리에 등록 — is_registered=True로 만료 방지
                gallery = MultiFeatureGallery(global_id=gid, is_registered=True)
                gallery.update('appearance', rep_feature, '__registered__')
                # 원본 ref 특징을 anchor로 고정 (EMA 드리프트 방지)
                gallery._anchor_features['appearance'] = rep_feature.copy()
                self.galleries[gid] = gallery
                registered += 1
                self.logger.info(f"펫 프로필 등록: {profile.name} -> G:{gid}")

            # _next_global_id를 등록된 ID 이후로 설정
            if self.galleries:
                max_registered = max(self.galleries.keys())
                self._next_global_id = max(self._next_global_id, max_registered + 1)

            self.logger.info(f"펫 프로필 {registered}개 사전 등록 완료 (next_id={self._next_global_id})")

        except Exception as e:
            self.logger.warning(f"펫 프로필 로드 실패: {e}")

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

        # 활성 갤러리만 대상 (등록된 펫은 항상 활성)
        current_time = time.time()
        active_galleries = {
            gid: gallery for gid, gallery in self.galleries.items()
            if gallery.is_registered or (current_time - gallery.last_seen) < self.inactive_timeout
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
                            box: tuple = None,
                            exclude_gids: set = None) -> int:
        """
        다중 특징으로 글로벌 ID 획득

        Args:
            channel_id: 채널 식별자
            local_id: 로컬 트랙 ID
            features: 특징 타입 -> 특징 벡터 딕셔너리
            box: 바운딩 박스 (선택)
            exclude_gids: 이번 프레임에서 이미 할당된 global_id (중복 방지)

        Returns:
            global_id
        """
        return self._get_global_id_internal(channel_id, local_id, features, box, exclude_gids)

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
                    self._update_gallery(global_id, features, channel_id, box)
                return global_id

        # 새로운 로컬 ID - 갤러리에서 매칭 시도
        matched_gid, similarity = self._find_matching_gallery_multi(
            features, exclude_gids, query_bbox=box
        )

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
        self._update_gallery(global_id, features, channel_id, box)

        return global_id

    @staticmethod
    def _bbox_distance(box1, box2) -> float:
        """Center distance between two bboxes."""
        if box1 is None or box2 is None:
            return 0.0
        return ((float(box1[0]) - float(box2[0])) ** 2 +
                (float(box1[1]) - float(box2[1])) ** 2) ** 0.5

    @staticmethod
    def _distance_penalty(dist: float, max_dist: float = 300.0) -> float:
        """Gaussian distance penalty in [0, 1]. 0 = close, 1 = far."""
        if dist <= 0:
            return 0.0
        sigma = max_dist * 0.6
        return 1.0 - np.exp(-(dist ** 2) / (2 * sigma ** 2))

    @staticmethod
    def _size_similarity(query_bbox: tuple, gallery: 'MultiFeatureGallery') -> float:
        """Query bbox와 갤러리 크기 프로필 간 유사도 (0~1).

        aspect ratio는 깊이에 불변하므로 주력으로, 면적은 보조로 사용.
        갤러리에 통계가 충분하지 않으면 1.0 반환 (패널티 없음).
        """
        if not gallery._size_stats or query_bbox is None or len(query_bbox) < 4:
            return 1.0  # 데이터 부족 → 중립

        x1, y1, x2, y2 = float(query_bbox[0]), float(query_bbox[1]), float(query_bbox[2]), float(query_bbox[3])
        if x2 > x1 and y2 > y1:
            w = x2 - x1
            h = y2 - y1
        else:
            w = x2 if x2 > 0 else 1
            h = y2 if y2 > 0 else 1

        query_ar = w / max(h, 1.0)
        query_area = w * h

        stats = gallery._size_stats
        ar_mean = stats['ar_mean']
        ar_std = max(stats['ar_std'], 0.05)  # 최소 std 설정
        area_mean = stats['area_mean']
        area_std = max(stats['area_std'], area_mean * 0.1)  # 최소 10% CV

        # Aspect ratio 유사도 (Gaussian, 주력 — 깊이 불변)
        ar_diff = abs(query_ar - ar_mean)
        ar_sim = float(np.exp(-(ar_diff ** 2) / (2 * ar_std ** 2)))

        # 면적 유사도 (Gaussian, 보조 — 깊이 의존적이므로 느슨하게)
        area_diff = abs(query_area - area_mean)
        # 면적은 3x std 허용 (카메라 깊이 변동 수용)
        area_sim = float(np.exp(-(area_diff ** 2) / (2 * (area_std * 3) ** 2)))

        # 가중 결합: aspect_ratio 70%, area 30%
        combined = 0.7 * ar_sim + 0.3 * area_sim
        return combined

    def _find_matching_gallery_multi(self, features: Dict[str, np.ndarray],
                                      exclude_gids: set = None,
                                      query_bbox: tuple = None) -> Tuple[Optional[int], float]:
        """다중 특징으로 갤러리 매칭 (위치 거리 패널티 포함)"""
        if not features:
            return None, 0.0

        current_time = time.time()
        active_galleries = {
            gid: gallery for gid, gallery in self.galleries.items()
            if (gallery.is_registered or (current_time - gallery.last_seen) < self.inactive_timeout)
            and (exclude_gids is None or gid not in exclude_gids)
        }

        if not active_galleries:
            return None, 0.0

        matched_gid, raw_sim = self.matching_strategy.match(
            features, active_galleries, 0.0  # get raw score, apply threshold after penalty
        )

        if matched_gid is None:
            return None, 0.0

        # Apply distance penalty + size similarity
        final_sim = raw_sim
        gallery = active_galleries.get(matched_gid)
        if query_bbox is not None and gallery:
            if gallery.last_bbox is not None:
                dist = self._bbox_distance(query_bbox, gallery.last_bbox)
                penalty = self._distance_penalty(dist)
                final_sim = raw_sim * (1.0 - 0.5 * penalty)
            # Size similarity: penalize mismatched body proportions
            size_sim = self._size_similarity(query_bbox, gallery)
            final_sim = final_sim * (0.7 + 0.3 * size_sim)  # 최대 30% 감소

        if final_sim >= self.similarity_threshold:
            return matched_gid, final_sim

        # If best match failed after penalty, try others with distance + size awareness
        best_gid = None
        best_score = 0.0
        for gid, gal in active_galleries.items():
            if gid == matched_gid:
                continue
            # Compute similarity per gallery
            sim_result = self.matching_strategy.match(
                features, {gid: gal}, 0.0
            )
            if sim_result[0] is None:
                continue
            score = sim_result[1]
            if query_bbox is not None:
                if gal.last_bbox is not None:
                    dist = self._bbox_distance(query_bbox, gal.last_bbox)
                    penalty = self._distance_penalty(dist)
                    score = score * (1.0 - 0.5 * penalty)
                size_sim = self._size_similarity(query_bbox, gal)
                score = score * (0.7 + 0.3 * size_sim)
            if score > best_score:
                best_score = score
                best_gid = gid

        if best_gid is not None and best_score >= self.similarity_threshold:
            return best_gid, best_score

        return None, 0.0

    def _update_gallery(self, global_id: int, features: Dict[str, np.ndarray],
                        channel_id: str, box: tuple = None):
        """갤러리 업데이트

        등록 프로필(is_registered)은 특징을 업데이트하지 않고 bbox만 갱신.
        anchor + representative가 원본 ref 그대로 유지되어 드리프트 없음.
        """
        if global_id not in self.galleries:
            return

        gallery = self.galleries[global_id]

        if gallery.is_registered and self.config.freeze_registered:
            # 등록 프로필 고정 모드: last_seen, channels_seen만 갱신
            gallery.last_seen = time.time()
            gallery.channels_seen.add(channel_id)
        else:
            # 비등록 객체 또는 freeze_registered=False: 특징 업데이트 (EMA + deque)
            for feat_type, feature in features.items():
                if feature is not None and len(feature) > 0:
                    weight = self.config.feature_weights.get(feat_type, 1.0)
                    gallery.update(feat_type, feature, channel_id,
                                  alpha=self.config.ema_alpha, weight=weight)

        if box is not None:
            gallery.last_bbox = tuple(box[:4])
            gallery.update_size_profile(box)

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

        # 비활성 갤러리 제거 (등록된 펫 프로필은 보존)
        inactive_gids = [
            gid for gid, gallery in self.galleries.items()
            if not gallery.is_registered
            and (current_time - gallery.last_seen) > self.inactive_timeout
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

        # 갤러리 크기 제한 (등록된 펫은 제거하지 않음)
        if len(self.galleries) > self.max_gallery_size:
            removable = [
                gid for gid, g in self.galleries.items() if not g.is_registered
            ]
            removable.sort(key=lambda gid: self.galleries[gid].last_seen)

            to_remove = removable[:len(self.galleries) - self.max_gallery_size]
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
