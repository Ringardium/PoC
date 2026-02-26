"""
Re-ID Tracker - 확장 가능한 ID 보정 트래커

기존 tracking.py의 트래커들과 함께 사용하여:
- ID 스위칭 감지 및 보정
- 객체 재식별 (occlusion 후)
- 선택적 글로벌 ID 할당
- 다중 특징 융합 지원 (외형, 모션, 행동 등)

Usage:
    # 기본 사용 (하위 호환)
    tracker = ReIDTracker(reid_method='adaptive')
    result = tracker.process(frame, boxes, track_ids)

    # 확장 사용
    tracker = ReIDTracker.create_default()
    tracker.add_extractor('behavior', BehaviorExtractor())
    result = tracker.process(frame, boxes, track_ids)
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
import logging
import time

# 새로운 features 모듈 사용
try:
    from reid.features import (
        FeatureExtractor,
        FeatureConfig,
        FeatureOutput,
        TrackContext,
        FeatureFusionEngine,
        FusionConfig,
        MatchingStrategy,
        CosineSimilarityMatching,
        create_matching_strategy,
        TrackEventBus,
        TrackEvent,
        TrackEventType,
        TrackStateManager,
        LoggingEventHandler,
        IDSwitchHandler,
        HistogramFeatureExtractor,
        DINOv2FeatureExtractor,
        MotionFeatureExtractor,
        TrajectoryFeatureExtractor,
    )
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False
    logging.warning("features 모듈을 찾을 수 없습니다. 기존 방식으로 동작합니다.")

# 기존 lightweight 모듈 (fallback)
try:
    from reid.lightweight import AdaptiveReID, FastHistogramReID, create_lightweight_reid
    LIGHTWEIGHT_AVAILABLE = True
except ImportError:
    LIGHTWEIGHT_AVAILABLE = False

from reid.global_id import GlobalIDManager


@dataclass
class ReIDTrackerConfig:
    """ReID 트래커 설정"""
    # 특징 추출
    use_appearance: bool = True
    use_motion: bool = True
    use_deep_model: bool = True

    # 매칭
    matching_method: str = 'cosine'
    similarity_threshold: float = 0.5

    # ID 보정
    correction_enabled: bool = True
    correction_confidence_threshold: float = 0.7
    correction_cooldown_frames: int = 30  # 보정 후 N프레임 동안 재보정 금지

    # 갤러리 관리
    gallery_max_size: int = 10
    gallery_ema_alpha: float = 0.1
    max_disappeared_frames: int = 30

    # 글로벌 ID
    global_id_enabled: bool = False

    # 이벤트
    enable_events: bool = True
    log_events: bool = False

    # Fusion
    fusion_strategy: str = 'adaptive'

    # 디버그
    debug: bool = False


@dataclass
class TrackGallery:
    """트랙별 특징 갤러리"""
    track_id: int
    features: deque = field(default_factory=lambda: deque(maxlen=10))
    representative: np.ndarray = None
    last_seen_frame: int = 0
    disappeared_count: int = 0
    last_bbox: Optional[Tuple] = None  # (x_center, y_center, w, h) — last known position
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update(self, feature: np.ndarray, frame_idx: int, alpha: float = 0.3):
        """특징 업데이트 및 EMA 갱신"""
        if feature is None or len(feature) == 0:
            return

        self.features.append(feature)
        self.last_seen_frame = frame_idx
        self.disappeared_count = 0

        if self.representative is None:
            self.representative = feature.copy()
        else:
            self.representative = alpha * feature + (1 - alpha) * self.representative
            norm = np.linalg.norm(self.representative)
            if norm > 1e-7:
                self.representative /= norm


@dataclass
class ProcessResult:
    """처리 결과"""
    corrected_ids: List[int]
    global_ids: Optional[List[int]] = None
    corrections: List[Dict] = field(default_factory=list)
    features: Optional[List[np.ndarray]] = None
    confidences: Optional[List[float]] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReIDTracker:
    """
    확장 가능한 Re-ID 기반 ID 보정 트래커

    특징:
    - 플러그인 방식 특징 추출기 (외형, 모션, 행동 등 확장 가능)
    - 다양한 Fusion 전략 지원
    - 이벤트 기반 트랙 상태 관리
    - 전략 패턴 기반 매칭
    - 기존 API 하위 호환성 유지
    """

    def __init__(self,
                 reid_method: str = 'adaptive',
                 similarity_threshold: float = 0.5,
                 correction_enabled: bool = True,
                 global_id_enabled: bool = False,
                 max_disappeared_frames: int = 30,
                 config: ReIDTrackerConfig = None):
        """
        Args:
            reid_method: 'histogram', 'mobilenet', 'adaptive' 중 선택 (하위 호환)
            similarity_threshold: 동일 객체 판단 임계값
            correction_enabled: ID 보정 활성화
            global_id_enabled: 글로벌 ID 할당 활성화
            max_disappeared_frames: 객체 정보 유지 프레임 수
            config: 상세 설정 (제공시 다른 인자 무시)
        """
        # 설정 초기화
        if config:
            self.config = config
        else:
            self.config = ReIDTrackerConfig(
                use_appearance=True,
                use_motion=False,
                use_deep_model=(reid_method in ['adaptive', 'mobilenet']),
                matching_method='cosine',
                similarity_threshold=similarity_threshold,
                correction_enabled=correction_enabled,
                global_id_enabled=global_id_enabled,
                max_disappeared_frames=max_disappeared_frames,
                enable_events=False
            )

        self.logger = logging.getLogger(self.__class__.__name__)
        self._reid_method = reid_method

        # Fusion 엔진 또는 기존 추출기
        self.fusion_engine: Optional[FeatureFusionEngine] = None
        self._legacy_extractor = None
        self._setup_extractors()

        # 매칭 전략
        if FEATURES_AVAILABLE:
            self.matching_strategy: MatchingStrategy = create_matching_strategy(
                self.config.matching_method,
                threshold=self.config.similarity_threshold
            )
        else:
            self.matching_strategy = None

        # 이벤트 시스템
        self.event_bus = None
        self.state_manager = None
        if FEATURES_AVAILABLE and self.config.enable_events:
            self.event_bus = TrackEventBus()
            self.state_manager = TrackStateManager(self.event_bus)
            if self.config.log_events:
                self.event_bus.subscribe(LoggingEventHandler())
                self.event_bus.subscribe(IDSwitchHandler())

        # 트랙별 갤러리
        self.galleries: Dict[int, TrackGallery] = {}
        # 하위 호환용 별칭
        self.track_galleries = self.galleries
        self.track_features: Dict[int, np.ndarray] = {}
        self.disappeared: Dict[int, int] = defaultdict(int)

        # ID 매핑
        self.id_corrections: Dict[int, List[int]] = defaultdict(list)

        # 글로벌 ID 매니저
        self.global_id_manager = None
        if self.config.global_id_enabled:
            self.global_id_manager = GlobalIDManager(
                feature_dim=self.feature_dim,
                similarity_threshold=self.config.similarity_threshold
            )

        # 보정 쿨다운: track_id -> 남은 쿨다운 프레임 수
        self._correction_cooldown: Dict[int, int] = {}

        # 통계
        self.stats = {
            'total_corrections': 0,
            'frame_count': 0
        }
        self._processing_times = deque(maxlen=100)
        self._frame_idx = 0

        self.logger.info(f"ReIDTracker 초기화 - method: {self._reid_method}, "
                        f"correction: {self.config.correction_enabled}, "
                        f"global_id: {self.config.global_id_enabled}")

    def _setup_extractors(self):
        """특징 추출기 설정"""
        if FEATURES_AVAILABLE:
            # 새로운 Fusion 엔진 사용
            fusion_config = FusionConfig(
                strategy=self.config.fusion_strategy,
                normalize_output=True
            )
            self.fusion_engine = FeatureFusionEngine(fusion_config)

            # 외형 특징
            if self.config.use_appearance:
                if self.config.use_deep_model:
                    try:
                        dino = DINOv2FeatureExtractor()
                        self.fusion_engine.register_extractor(
                            'appearance', dino, weight=1.0
                        )
                    except Exception as e:
                        self.logger.warning(f"DINOv2 init failed, falling back to Histogram: {e}")
                        self.fusion_engine.register_extractor(
                            'histogram',
                            HistogramFeatureExtractor(),
                            weight=1.0
                        )
                else:
                    self.fusion_engine.register_extractor(
                        'histogram',
                        HistogramFeatureExtractor(),
                        weight=1.0
                    )

            # 모션 특징
            if self.config.use_motion:
                self.fusion_engine.register_extractor(
                    'motion',
                    MotionFeatureExtractor(),
                    weight=0.5
                )
                self.fusion_engine.register_extractor(
                    'trajectory',
                    TrajectoryFeatureExtractor(),
                    weight=0.3
                )

            self.fusion_engine.initialize()

        elif LIGHTWEIGHT_AVAILABLE:
            # 기존 lightweight 추출기 사용
            self._legacy_extractor = create_lightweight_reid(self._reid_method)
        else:
            raise RuntimeError("특징 추출 모듈을 사용할 수 없습니다.")

    @property
    def feature_dim(self) -> int:
        """전체 특징 차원"""
        if self.fusion_engine:
            return self.fusion_engine.total_feature_dim
        elif self._legacy_extractor:
            return self._legacy_extractor.feature_dim
        return 0

    # === 플러그인 API ===

    def add_extractor(self, name: str, extractor, weight: float = 1.0):
        """특징 추출기 추가 (플러그인)"""
        if self.fusion_engine:
            self.fusion_engine.register_extractor(name, extractor, weight)
            self.logger.info(f"Added extractor: {name}")
        else:
            self.logger.warning("Fusion engine not available, cannot add extractor")

    def remove_extractor(self, name: str):
        """특징 추출기 제거"""
        if self.fusion_engine:
            self.fusion_engine.remove_extractor(name)

    def set_matching_strategy(self, strategy):
        """매칭 전략 변경"""
        self.matching_strategy = strategy

    def subscribe_event(self, handler):
        """이벤트 핸들러 등록"""
        if self.event_bus:
            self.event_bus.subscribe(handler)

    def on_event(self, event_type, callback: Callable):
        """특정 이벤트에 콜백 등록"""
        if self.event_bus:
            self.event_bus.subscribe_callback(event_type, callback)

    def register_pet_profiles(self, profile_dir: str):
        """Pet profile 레퍼런스 이미지를 자체 특징 추출기로 추출하여 GlobalIDManager에 등록.

        Fusion engine 사용 시 requires_history=False인 extractor만 사용하여
        feature type별로 개별 등록. 정지 이미지에서 motion/trajectory 등
        시계열 feature가 0으로 채워지는 문제를 방지.
        """
        if not self.global_id_manager:
            return

        try:
            import cv2
            from pathlib import Path
            from tools.pet_profiles import PetProfileStore
            from reid.global_id import MultiFeatureGallery

            store = PetProfileStore(profile_dir)
            profiles = store.list_pets()
            if not profiles:
                self.logger.info("등록된 펫 프로필 없음")
                return

            base = Path(profile_dir)
            registered = 0

            for profile in profiles:
                gid = profile.global_id
                # type별 feature 누적: {feat_type: [feat1, feat2, ...]}
                type_features = {}

                for img_rel in profile.reference_images:
                    img_path = base / img_rel
                    if not img_path.exists():
                        continue
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue

                    if self.fusion_engine:
                        from reid.features.base import TrackContext
                        ctx = TrackContext(track_id=gid, bbox=(0, 0, img.shape[1], img.shape[0]))
                        # extract_static: requires_history=False만, type별 개별 반환
                        static_outputs = self.fusion_engine.extract_static(img, ctx)
                        for feat_type, output in static_outputs.items():
                            if output.is_valid:
                                feat = output.feature
                                norm = np.linalg.norm(feat)
                                if norm > 1e-8:
                                    feat = feat / norm
                                type_features.setdefault(feat_type, []).append(feat)
                    elif self._legacy_extractor:
                        feats = self._legacy_extractor.extract_features_batch([img], [gid])
                        if feats is not None and len(feats) > 0 and feats[0] is not None and len(feats[0]) > 0:
                            feat = feats[0]
                            norm = np.linalg.norm(feat)
                            if norm > 1e-8:
                                feat = feat / norm
                            type_features.setdefault('appearance', []).append(feat)

                if not type_features:
                    continue

                # type별 대표 특징 계산 후 갤러리에 등록
                gallery = MultiFeatureGallery(global_id=gid, is_registered=True)
                dims = []
                for feat_type, feat_list in type_features.items():
                    rep = np.mean(feat_list, axis=0).astype(np.float32)
                    rep = rep / (np.linalg.norm(rep) + 1e-8)
                    gallery.update(feat_type, rep, '__registered__')
                    # 원본 ref 특징을 anchor로 고정 (EMA 드리프트 방지)
                    gallery._anchor_features[feat_type] = rep.copy()
                    dims.append(f"{feat_type}={len(rep)}")

                self.global_id_manager.galleries[gid] = gallery
                registered += 1
                self.logger.info(f"펫 프로필 등록: {profile.name} -> G:{gid} ({', '.join(dims)})")

            if self.global_id_manager.galleries:
                max_gid = max(self.global_id_manager.galleries.keys())
                self.global_id_manager._next_global_id = max(
                    self.global_id_manager._next_global_id, max_gid + 1
                )

            self.logger.info(f"펫 프로필 {registered}개 등록 완료")

        except Exception as e:
            self.logger.warning(f"펫 프로필 등록 실패: {e}", exc_info=True)

    # === 메인 처리 ===

    def process(self, frame: np.ndarray, boxes: List, track_ids: List[int],
                channel_id: str = "default",
                proximity_locked: set = None) -> Dict:
        """
        Re-ID 처리 메인 함수

        Args:
            frame: 입력 프레임
            boxes: 바운딩 박스 리스트 [(x_center, y_center, w, h), ...]
            track_ids: 기존 트래커의 트랙 ID 리스트
            channel_id: 채널 ID (멀티채널용)
            proximity_locked: 근접 잠금 트랙 ID 집합 — 갤러리 업데이트 동결

        Returns:
            {
                'corrected_ids': 보정된 ID 리스트,
                'global_ids': 글로벌 ID 리스트 (활성화시),
                'corrections': 보정 정보,
                'features': 추출된 특징,
                'processing_time': 처리 시간
            }
        """
        start_time = time.time()
        self._frame_idx += 1
        self.stats['frame_count'] += 1

        result = {
            'corrected_ids': list(track_ids),
            'global_ids': None,
            'corrections': [],
            'features': None,
            'processing_time': 0
        }

        if len(boxes) == 0:
            self._update_disappeared_tracks(set())
            result['processing_time'] = time.time() - start_time
            return result

        # 1. 크롭 및 특징 추출
        crops = self._extract_crops(frame, boxes)

        # Global ID용 static feature (appearance만, type별 분리)
        self._last_static_features = None

        if self.fusion_engine:
            contexts = self._create_contexts(boxes, track_ids)
            feature_outputs = self.fusion_engine.extract_batch(crops, contexts)
            features = [out.feature for out in feature_outputs]
            confidences = [out.confidence for out in feature_outputs]
            # Global ID 매칭용: requires_history=False인 feature만 추출
            if self.config.global_id_enabled and self.global_id_manager:
                static_features_list = []
                for crop, ctx in zip(crops, contexts):
                    static_features_list.append(
                        self.fusion_engine.extract_static(crop, ctx)
                    )
                self._last_static_features = static_features_list
        else:
            features = self._legacy_extractor.extract_features_batch(crops, track_ids)
            confidences = [1.0] * len(features)

        result['features'] = features

        # 2. ID 보정
        if self.config.correction_enabled:
            corrected_ids, corrections = self._correct_ids(
                boxes, track_ids, features, confidences
            )
            result['corrected_ids'] = corrected_ids
            result['corrections'] = corrections
        else:
            corrected_ids = track_ids

        # 3. 갤러리 업데이트
        # ONLY update gallery for tracks where no correction was applied.
        # Corrected tracks' galleries would get contaminated with wrong features
        # (stream_processor may also reject the correction via stabilization buffer).
        # Proximity-locked tracks are also frozen to prevent contamination from
        # BotSORT ID swaps during close encounters.
        current_ids = set(corrected_ids)
        _prox = proximity_locked or set()
        corrected_set = {c['new_id'] for c in corrections} if corrections else set()
        for orig_id, corr_id, feature, conf, box in zip(track_ids, corrected_ids, features, confidences, boxes):
            if feature is None or len(feature) == 0:
                continue
            if conf < self.config.correction_confidence_threshold:
                continue
            if orig_id != corr_id:
                # Correction proposed — update bbox only, NOT features
                if corr_id in self.galleries and box is not None:
                    self.galleries[corr_id].last_bbox = tuple(box[:4])
                continue
            if corr_id in corrected_set:
                # Another track was corrected TO this id — skip to avoid contamination
                continue
            if orig_id in _prox:
                # Proximity locked — freeze gallery features, only update bbox
                if corr_id in self.galleries and box is not None:
                    self.galleries[corr_id].last_bbox = tuple(box[:4])
                continue
            self._update_gallery(corr_id, feature, bbox=box)

        # 4. 사라진 트랙 처리
        self._update_disappeared_tracks(current_ids)

        # 5. 글로벌 ID (선택적)
        if self.config.global_id_enabled and self.global_id_manager:
            global_ids = []
            used_gids = set()  # Prevent same global_id assigned to multiple objects in one frame
            for i, (track_id, feature, box) in enumerate(zip(corrected_ids, features, boxes)):
                if track_id in _prox:
                    # Proximity locked — skip feature update, keep existing mapping
                    key = (channel_id, track_id)
                    existing = self.global_id_manager.local_to_global.get(key)
                    if existing is not None:
                        global_ids.append(existing)
                        used_gids.add(existing)
                        continue
                # Static features (appearance만) 사용 — 프로필과 동일 차원
                if self._last_static_features and i < len(self._last_static_features):
                    static_feats = self._last_static_features[i]
                    # type별 features dict 구성
                    feat_dict = {ft: out.feature for ft, out in static_feats.items()
                                 if out.is_valid}
                    gid = self.global_id_manager.get_global_id_multi(
                        channel_id, track_id, feat_dict, box,
                        exclude_gids=used_gids,
                    )
                else:
                    gid = self.global_id_manager.get_global_id(
                        channel_id, track_id, feature, box,
                        exclude_gids=used_gids,
                    )
                global_ids.append(gid)
                used_gids.add(gid)
            result['global_ids'] = global_ids

        result['processing_time'] = time.time() - start_time
        self._processing_times.append(result['processing_time'])

        return result

    def _extract_crops(self, frame: np.ndarray, boxes: List) -> List[np.ndarray]:
        """바운딩 박스에서 크롭 추출"""
        crops = []
        h, w = frame.shape[:2]

        for box in boxes:
            x1, y1, x2, y2 = self._normalize_box(box, w, h)

            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                crops.append(crop)
            else:
                crops.append(np.zeros((10, 10, 3), dtype=np.uint8))

        return crops

    def _normalize_box(self, box, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """박스 형식 정규화 -> (x1, y1, x2, y2)"""
        if len(box) >= 4:
            if box[2] > box[0] and box[3] > box[1]:
                x1, y1, x2, y2 = box[:4]
            else:
                x_c, y_c, bw, bh = box[:4]
                x1 = x_c - bw / 2
                y1 = y_c - bh / 2
                x2 = x_c + bw / 2
                y2 = y_c + bh / 2
        else:
            return 0, 0, 10, 10

        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(img_w, int(x2))
        y2 = min(img_h, int(y2))

        return x1, y1, x2, y2

    def _create_contexts(self, boxes: List, track_ids: List[int]) -> List:
        """트랙 컨텍스트 생성"""
        contexts = []

        for box, tid in zip(boxes, track_ids):
            prev_bboxes = []
            if tid in self.galleries:
                prev_bboxes = self.galleries[tid].metadata.get('prev_bboxes', [])

            context = TrackContext(
                track_id=tid,
                frame_idx=self._frame_idx,
                bbox=tuple(box[:4]) if len(box) >= 4 else None,
                prev_bboxes=prev_bboxes
            )
            contexts.append(context)

            if tid in self.galleries:
                if 'prev_bboxes' not in self.galleries[tid].metadata:
                    self.galleries[tid].metadata['prev_bboxes'] = deque(maxlen=30)
                self.galleries[tid].metadata['prev_bboxes'].append(tuple(box[:4]))

        return contexts

    def _update_gallery(self, track_id: int, feature: np.ndarray,
                        bbox: Optional[Tuple] = None):
        """갤러리 업데이트"""
        if feature is None or len(feature) == 0:
            return

        if track_id not in self.galleries:
            self.galleries[track_id] = TrackGallery(track_id=track_id)

            if self.event_bus:
                self.event_bus.publish(TrackEvent(
                    event_type=TrackEventType.TRACK_CREATED,
                    track_id=track_id,
                    frame_idx=self._frame_idx
                ))

        self.galleries[track_id].update(
            feature, self._frame_idx, self.config.gallery_ema_alpha
        )
        if bbox is not None:
            self.galleries[track_id].last_bbox = tuple(bbox[:4])

        # 하위 호환용
        self.track_features[track_id] = self.galleries[track_id].representative
        self.disappeared[track_id] = 0

    def _correct_ids(self, boxes: List, track_ids: List[int],
                     features: List[np.ndarray],
                     confidences: List[float]) -> Tuple[List[int], List[Dict]]:
        """ID 스위칭 감지 및 보정"""
        if len(track_ids) == 0:
            return [], []

        # Tick cooldown counters
        for tid in list(self._correction_cooldown):
            self._correction_cooldown[tid] -= 1
            if self._correction_cooldown[tid] <= 0:
                del self._correction_cooldown[tid]

        corrected_ids = list(track_ids)
        corrections = []
        # Track both original and already-corrected IDs to prevent duplicates
        used_ids = set(track_ids)

        for i, (track_id, feature, conf, box) in enumerate(zip(track_ids, features, confidences, boxes)):
            if feature is None or len(feature) == 0:
                continue

            if conf < self.config.correction_confidence_threshold:
                continue

            # Skip if this track is in cooldown (recently corrected)
            if track_id in self._correction_cooldown:
                continue

            current_bbox = tuple(box[:4]) if box is not None and len(box) >= 4 else None

            # 기존 트랙과 비교
            if track_id in self.galleries:
                existing = self.galleries[track_id].representative
                if existing is not None:
                    similarity = self._compute_similarity(feature, existing)

                    if similarity < self.config.similarity_threshold:
                        # Cross-match check: if the feature matches another ACTIVE
                        # gallery better than its own, this is likely crop contamination
                        # (overlapping bboxes near each other), not a real ID switch.
                        best_active_sim = similarity
                        for other_tid in track_ids:
                            if other_tid == track_id:
                                continue
                            og = self.galleries.get(other_tid)
                            if og and og.representative is not None:
                                s = self._compute_similarity(feature, og.representative)
                                if s > best_active_sim:
                                    best_active_sim = s
                        if best_active_sim > similarity:
                            # Feature resembles a nearby active track more — crop contamination
                            continue

                        correction = self._try_match_disappeared(
                            i, track_id, feature, used_ids,
                            current_bbox=current_bbox,
                        )
                        if correction:
                            corrected_ids[i] = correction['new_id']
                            used_ids.add(correction['new_id'])
                            self._correction_cooldown[correction['new_id']] = self.config.correction_cooldown_frames
                            corrections.append(correction)
            else:
                correction = self._try_reidentify(
                    i, track_id, feature, used_ids,
                    current_bbox=current_bbox,
                )
                if correction:
                    corrected_ids[i] = correction['new_id']
                    used_ids.add(correction['new_id'])
                    self._correction_cooldown[correction['new_id']] = self.config.correction_cooldown_frames
                    corrections.append(correction)

        return corrected_ids, corrections

    @staticmethod
    def _bbox_center_distance(box1, box2) -> float:
        """Compute Euclidean distance between bbox centers."""
        if box1 is None or box2 is None:
            return 0.0
        cx1, cy1 = float(box1[0]), float(box1[1])
        cx2, cy2 = float(box2[0]), float(box2[1])
        return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

    @staticmethod
    def _distance_penalty(dist: float, max_dist: float = 300.0) -> float:
        """Convert distance to a [0, 1] penalty. 0 = no penalty, 1 = full penalty.

        Uses sigmoid-like ramp: penalty is small when close, large when far.
        ``max_dist`` is the distance at which penalty ≈ 0.7.
        """
        if dist <= 0:
            return 0.0
        # Smooth ramp: penalty = 1 - exp(-dist^2 / (2 * sigma^2))
        sigma = max_dist * 0.6
        return 1.0 - np.exp(-(dist ** 2) / (2 * sigma ** 2))

    def _try_match_disappeared(self, idx: int, track_id: int,
                                feature: np.ndarray,
                                current_ids: set,
                                current_bbox=None) -> Optional[Dict]:
        """사라진 트랙에서 매칭 시도 (위치 거리 패널티 포함)"""
        disappeared_galleries = {
            tid: g for tid, g in self.galleries.items()
            if tid not in current_ids and
               g.disappeared_count <= self.config.max_disappeared_frames and
               g.representative is not None
        }

        if not disappeared_galleries:
            return None

        best_match = None
        best_score = 0.0

        for tid, gallery in disappeared_galleries.items():
            sim = self._compute_similarity(feature, gallery.representative)
            # Apply distance penalty: reduce similarity for far-away tracks
            if current_bbox is not None and gallery.last_bbox is not None:
                dist = self._bbox_center_distance(current_bbox, gallery.last_bbox)
                penalty = self._distance_penalty(dist)
                score = sim * (1.0 - 0.5 * penalty)  # max 50% penalty from distance
            else:
                score = sim
            if score > best_score:
                best_score = score
                best_match = tid

        if best_match is not None and best_score >= self.config.similarity_threshold:
            correction = {
                'index': idx,
                'old_id': track_id,
                'new_id': best_match,
                'similarity': best_score,
                'reason': 'switch_detected'
            }

            self.stats['total_corrections'] += 1
            self.id_corrections[best_match].append(track_id)

            if self.event_bus:
                self.event_bus.publish(TrackEvent(
                    event_type=TrackEventType.ID_CORRECTED,
                    track_id=best_match,
                    frame_idx=self._frame_idx,
                    data=correction
                ))

            self.logger.info(f"ID 보정: {track_id} -> {best_match} (유사도: {best_score:.3f})")
            return correction

        return None

    def _try_reidentify(self, idx: int, track_id: int,
                        feature: np.ndarray,
                        current_ids: set,
                        current_bbox=None) -> Optional[Dict]:
        """재식별 시도 (위치 거리 패널티 포함)"""
        disappeared = {
            tid: g for tid, g in self.galleries.items()
            if tid not in current_ids and
               g.disappeared_count > 0 and
               g.disappeared_count <= self.config.max_disappeared_frames and
               g.representative is not None
        }

        if not disappeared:
            return None

        best_match = None
        best_score = 0.0

        for tid, gallery in disappeared.items():
            sim = self._compute_similarity(feature, gallery.representative)
            # Apply distance penalty
            if current_bbox is not None and gallery.last_bbox is not None:
                dist = self._bbox_center_distance(current_bbox, gallery.last_bbox)
                penalty = self._distance_penalty(dist)
                score = sim * (1.0 - 0.5 * penalty)
            else:
                score = sim
            if score > best_score:
                best_score = score
                best_match = tid

        if best_match is not None and best_score >= self.config.similarity_threshold:
            correction = {
                'index': idx,
                'old_id': track_id,
                'new_id': best_match,
                'similarity': best_score,
                'reason': 're_identified'
            }

            self.stats['total_corrections'] += 1

            if self.event_bus:
                self.event_bus.publish(TrackEvent(
                    event_type=TrackEventType.TRACK_RECOVERED,
                    track_id=best_match,
                    frame_idx=self._frame_idx,
                    data=correction
                ))

            self.logger.info(f"재식별: {track_id} -> {best_match} (유사도: {best_score:.3f})")
            return correction

        return None

    def _compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """특징 유사도 계산"""
        if self.matching_strategy:
            return self.matching_strategy.compute_similarity(feat1, feat2)

        # Fallback: 직접 계산
        if feat1 is None or feat2 is None:
            return 0.0
        return float(np.dot(feat1, feat2))

    def _update_disappeared_tracks(self, current_ids: set):
        """사라진 트랙 카운터 업데이트"""
        to_remove = []

        for track_id, gallery in self.galleries.items():
            if track_id not in current_ids:
                gallery.disappeared_count += 1
                self.disappeared[track_id] = gallery.disappeared_count

                if gallery.disappeared_count > self.config.max_disappeared_frames:
                    to_remove.append(track_id)

                    if self.event_bus:
                        self.event_bus.publish(TrackEvent(
                            event_type=TrackEventType.TRACK_DELETED,
                            track_id=track_id,
                            frame_idx=self._frame_idx
                        ))

        for track_id in to_remove:
            del self.galleries[track_id]
            if track_id in self.track_features:
                del self.track_features[track_id]
            if track_id in self.disappeared:
                del self.disappeared[track_id]

        # Fusion 엔진 상태 정리
        if self.fusion_engine:
            active_ids = set(self.galleries.keys())
            self.fusion_engine.reset_state(active_ids)

        # Legacy 추출기 상태 정리
        if self._legacy_extractor and hasattr(self._legacy_extractor, 'cleanup_tracks'):
            active_ids = set(self.galleries.keys())
            self._legacy_extractor.cleanup_tracks(active_ids)

    # === 하위 호환 API ===

    def correct_ids(self, boxes: List, track_ids: List[int],
                    features: List[np.ndarray]) -> Tuple[List[int], List[dict]]:
        """ID 스위칭 감지 및 보정 (하위 호환)"""
        confidences = [1.0] * len(features)
        return self._correct_ids(boxes, track_ids, features, confidences)

    def _find_best_match(self, feature: np.ndarray,
                         exclude_ids: set = None) -> Tuple[Optional[int], float]:
        """갤러리에서 가장 유사한 트랙 찾기 (하위 호환)"""
        if feature is None or not self.track_features:
            return None, 0.0

        exclude_ids = exclude_ids or set()
        best_id = None
        best_sim = 0.0

        for track_id, rep_feature in self.track_features.items():
            if track_id in exclude_ids:
                continue

            sim = self._compute_similarity(feature, rep_feature)
            if sim > best_sim:
                best_sim = sim
                best_id = track_id

        return best_id, best_sim

    def _cleanup_old_tracks(self):
        """오래된 트랙 정보 정리 (하위 호환)"""
        self._update_disappeared_tracks(set())

    # === 통계 및 유틸리티 ===

    def get_stats(self) -> Dict:
        """통계 반환"""
        stats = {
            **self.stats,
            'active_tracks': len(self.galleries),
            'feature_dim': self.feature_dim,
            'avg_processing_time': np.mean(self._processing_times) if self._processing_times else 0,
        }

        if self.fusion_engine:
            stats['extractors'] = list(self.fusion_engine.extractors.keys())

        if self.event_bus:
            stats['events'] = self.event_bus.get_stats()

        if self.global_id_manager:
            stats['global_id_stats'] = self.global_id_manager.get_stats()

        return stats

    def visualize(self, frame: np.ndarray, boxes: List, track_ids: List[int],
                  global_ids: List[int] = None, corrections: List[dict] = None) -> np.ndarray:
        """결과 시각화"""
        vis_frame = frame.copy()

        for i, (box, tid) in enumerate(zip(boxes, track_ids)):
            x1, y1, x2, y2 = self._normalize_box(box, frame.shape[1], frame.shape[0])

            is_corrected = corrections and any(c['index'] == i for c in corrections)
            color = (0, 255, 255) if is_corrected else (0, 255, 0)

            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

            if global_ids:
                id_text = f"L:{tid}/G:{global_ids[i]}"
            else:
                id_text = f"ID:{tid}"

            cv2.putText(vis_frame, id_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if corrections:
            y_offset = 30
            for corr in corrections[:3]:
                text = f"보정: {corr['old_id']} -> {corr['new_id']} ({corr['similarity']:.2f})"
                cv2.putText(vis_frame, text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_offset += 20

        return vis_frame

    def reset(self):
        """상태 초기화"""
        self.galleries.clear()
        self.track_features.clear()
        self.id_corrections.clear()
        self.disappeared.clear()
        self.stats = {'total_corrections': 0, 'frame_count': 0}
        self._processing_times.clear()
        self._frame_idx = 0

        if self.fusion_engine:
            self.fusion_engine.reset_state()

        if self.event_bus:
            self.event_bus.clear_history()

    # === 팩토리 메서드 ===

    @classmethod
    def create_default(cls, use_deep: bool = True, use_motion: bool = True) -> 'ReIDTracker':
        """기본 설정 트래커 생성"""
        config = ReIDTrackerConfig(
            use_appearance=True,
            use_motion=use_motion,
            use_deep_model=use_deep,
            correction_enabled=True,
            enable_events=True
        )
        return cls(config=config)

    @classmethod
    def create_lightweight(cls) -> 'ReIDTracker':
        """경량 트래커 생성"""
        config = ReIDTrackerConfig(
            use_appearance=True,
            use_motion=False,
            use_deep_model=False,
            correction_enabled=True,
            enable_events=False
        )
        return cls(config=config)

    @classmethod
    def create_accurate(cls) -> 'ReIDTracker':
        """정확도 우선 트래커 생성"""
        config = ReIDTrackerConfig(
            use_appearance=True,
            use_motion=True,
            use_deep_model=True,
            correction_enabled=True,
            correction_confidence_threshold=0.8,
            similarity_threshold=0.6,
            fusion_strategy='attention',
            enable_events=True,
            log_events=True
        )
        return cls(config=config)


# 테스트
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== ReIDTracker Test ===\n")

    # 기본 트래커 (하위 호환 테스트)
    tracker = ReIDTracker(reid_method='adaptive', correction_enabled=True)
    print(f"Feature dim: {tracker.feature_dim}")

    # 테스트 프레임 및 박스
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_boxes = [(100, 100, 50, 100), (300, 150, 50, 100)]
    test_ids = [1, 2]

    # 처리
    for i in range(5):
        result = tracker.process(test_frame, test_boxes, test_ids)
        print(f"Frame {i + 1}: corrected_ids={result['corrected_ids']}, "
              f"time={result['processing_time'] * 1000:.2f}ms")

    print(f"\nStats: {tracker.get_stats()}")

    # 팩토리 메서드 테스트
    print("\n=== Factory Methods Test ===")
    light_tracker = ReIDTracker.create_lightweight()
    print(f"Lightweight: dim={light_tracker.feature_dim}")
