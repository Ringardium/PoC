"""
Behavior Feature Extractors

행동 패턴 기반 특징 추출기 - 활동 수준, 상호작용 패턴 등

이 모듈은 확장 예시로, 실제 행동 분석 알고리즘을 추가할 때 참고용
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from collections import deque
import logging

from .base import FeatureExtractor, FeatureConfig, TrackContext, FeatureOutput


class ActivityLevelExtractor(FeatureExtractor):
    """
    활동 수준 특징 추출기

    이동량, 속도 변화, 방향 변화 등을 분석하여 활동 수준 특징 추출
    """

    def __init__(self, config: FeatureConfig = None,
                 history_length: int = 30,
                 expected_fps: float = 30.0,
                 gap_threshold_multiplier: float = 2.5):
        config = config or FeatureConfig(name="activity_level")
        super().__init__(config)

        self.history_length = history_length
        self._feature_dim = 8  # 활동 수준 관련 특징들

        # 프레임 갭 감지 설정
        self.gap_threshold = (1.0 / max(expected_fps, 1.0)) * gap_threshold_multiplier

        # 트랙별 히스토리
        self._track_histories: Dict[int, deque] = {}

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def feature_type(self) -> str:
        return "behavior"

    def extract(self, image: np.ndarray, context: TrackContext = None) -> FeatureOutput:
        if context is None or context.bbox is None:
            return self.get_zero_feature()

        track_id = context.track_id

        # 중심점 및 크기 계산
        bbox = context.bbox
        if len(bbox) >= 4:
            if bbox[2] > bbox[0]:  # (x1, y1, x2, y2)
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            else:  # (x, y, w, h)
                cx = bbox[0] + bbox[2] / 2
                cy = bbox[1] + bbox[3] / 2
                area = bbox[2] * bbox[3]
        else:
            return self.get_zero_feature()

        # 히스토리 초기화
        if track_id not in self._track_histories:
            self._track_histories[track_id] = deque(maxlen=self.history_length)

        history = self._track_histories[track_id]
        history.append({
            'cx': cx, 'cy': cy, 'area': area,
            'frame_idx': context.frame_idx,
            'timestamp': time.monotonic()
        })

        if len(history) < 5:
            return FeatureOutput(
                feature=np.zeros(self.feature_dim, dtype=np.float32),
                feature_type=self.feature_type,
                confidence=0.2
            )

        # 활동 수준 특징 계산
        features = self._compute_activity_features(history)

        # 갭 비율 기반 confidence 감소
        timestamps = [h.get('timestamp', 0) for h in history]
        gap_count = sum(
            1 for i in range(1, len(timestamps))
            if timestamps[i] - timestamps[i - 1] > self.gap_threshold
        )
        gap_ratio = gap_count / max(len(timestamps) - 1, 1)
        base_confidence = min(len(history) / self.history_length, 1.0)
        confidence = base_confidence * (1.0 - 0.7 * gap_ratio)

        return FeatureOutput(
            feature=features,
            feature_type=self.feature_type,
            confidence=confidence,
            metadata={'gap_count': gap_count, 'gap_ratio': gap_ratio}
        )

    def _compute_activity_features(self, history: deque) -> np.ndarray:
        """활동 수준 특징 계산 (프레임 갭 걸친 쌍 스킵)"""
        positions = [(h['cx'], h['cy']) for h in history]
        areas = [h['area'] for h in history]
        timestamps = [h.get('timestamp', 0) for h in history]

        # 이동 거리들 (갭 걸친 쌍 스킵)
        displacements = []
        for i in range(1, len(positions)):
            dt_sec = timestamps[i] - timestamps[i - 1]
            if dt_sec > self.gap_threshold:
                continue
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            displacements.append(np.sqrt(dx ** 2 + dy ** 2))

        # 특징 계산
        features = [
            np.mean(displacements) if displacements else 0,  # 평균 이동 거리
            np.std(displacements) if len(displacements) > 1 else 0,  # 이동 거리 변동성
            np.max(displacements) if displacements else 0,  # 최대 이동 거리
            sum(d > 10 for d in displacements) / max(len(displacements), 1),  # 활발한 이동 비율
            np.mean(areas) / 10000,  # 평균 크기 (정규화)
            np.std(areas) / 1000 if len(areas) > 1 else 0,  # 크기 변동성
            self._compute_direction_entropy(positions, timestamps),  # 방향 엔트로피
            self._compute_territory_size(positions)  # 활동 영역 크기
        ]

        feature_vec = np.array(features, dtype=np.float32)
        feature_vec = np.nan_to_num(feature_vec, nan=0.0, posinf=1.0, neginf=0.0)

        return self.normalize(feature_vec)

    def _compute_direction_entropy(self, positions: List[Tuple],
                                    timestamps: List[float] = None) -> float:
        """이동 방향 엔트로피 (불규칙할수록 높음, 프레임 갭 걸친 쌍 스킵)"""
        if len(positions) < 3:
            return 0.0

        # 방향 각도 계산
        angles = []
        for i in range(1, len(positions)):
            if timestamps and len(timestamps) > i:
                dt_sec = timestamps[i] - timestamps[i - 1]
                if dt_sec > self.gap_threshold:
                    continue
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                angle = np.arctan2(dy, dx)
                # 8방향으로 양자화
                bin_idx = int((angle + np.pi) / (np.pi / 4)) % 8
                angles.append(bin_idx)

        if not angles:
            return 0.0

        # 히스토그램 엔트로피
        hist = np.bincount(angles, minlength=8) / len(angles)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0

        return entropy / 3.0  # 최대 엔트로피(log2(8)=3)로 정규화

    def _compute_territory_size(self, positions: List[Tuple]) -> float:
        """활동 영역 크기 (convex hull 면적)"""
        if len(positions) < 3:
            return 0.0

        points = np.array(positions, dtype=np.float32)

        try:
            hull = cv2.convexHull(points)
            area = cv2.contourArea(hull)
            return min(area / 100000, 1.0)  # 정규화
        except:
            return 0.0

    def reset_state(self, track_ids: set = None):
        if track_ids is None:
            self._track_histories.clear()
        else:
            active_ids = set(self._track_histories.keys())
            for tid in active_ids - track_ids:
                del self._track_histories[tid]


class PostureFeatureExtractor(FeatureExtractor):
    """
    자세 특징 추출기

    바운딩 박스의 형태 변화를 통해 대략적인 자세 추정
    (실제로는 pose estimation 모델 사용 권장)
    """

    def __init__(self, config: FeatureConfig = None,
                 history_length: int = 10):
        config = config or FeatureConfig(name="posture")
        super().__init__(config)

        self.history_length = history_length
        self._feature_dim = 6

        self._track_histories: Dict[int, deque] = {}

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def feature_type(self) -> str:
        return "behavior"

    def extract(self, image: np.ndarray, context: TrackContext = None) -> FeatureOutput:
        if context is None or context.bbox is None:
            return self.get_zero_feature()

        track_id = context.track_id
        bbox = context.bbox

        # 종횡비 및 크기 계산
        if len(bbox) >= 4:
            if bbox[2] > bbox[0]:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
            else:
                w, h = bbox[2], bbox[3]
        else:
            return self.get_zero_feature()

        aspect_ratio = w / max(h, 1)
        size = w * h

        # 히스토리 관리
        if track_id not in self._track_histories:
            self._track_histories[track_id] = deque(maxlen=self.history_length)

        history = self._track_histories[track_id]
        history.append({'aspect_ratio': aspect_ratio, 'size': size, 'timestamp': time.monotonic()})

        # 자세 특징 계산
        features = self._compute_posture_features(history)

        return FeatureOutput(
            feature=features,
            feature_type=self.feature_type,
            confidence=min(len(history) / self.history_length, 1.0)
        )

    def _compute_posture_features(self, history: deque) -> np.ndarray:
        """자세 관련 특징"""
        aspect_ratios = [h['aspect_ratio'] for h in history]
        sizes = [h['size'] for h in history]

        current_ar = aspect_ratios[-1]
        current_size = sizes[-1]

        features = [
            current_ar,  # 현재 종횡비
            np.mean(aspect_ratios),  # 평균 종횡비
            np.std(aspect_ratios) if len(aspect_ratios) > 1 else 0,  # 종횡비 변동성
            1.0 if current_ar > 1.5 else (0.5 if current_ar > 0.7 else 0.0),  # 대략적 자세 상태
            current_size / 10000,  # 현재 크기 (정규화)
            np.mean(sizes) / 10000  # 평균 크기
        ]

        feature_vec = np.array(features, dtype=np.float32)
        return self.normalize(feature_vec)

    def reset_state(self, track_ids: set = None):
        if track_ids is None:
            self._track_histories.clear()
        else:
            active_ids = set(self._track_histories.keys())
            for tid in active_ids - track_ids:
                del self._track_histories[tid]


class InteractionFeatureExtractor(FeatureExtractor):
    """
    상호작용 특징 추출기

    다른 객체와의 거리, 접근 패턴 등 분석
    """

    def __init__(self, config: FeatureConfig = None,
                 interaction_threshold: float = 100.0):
        config = config or FeatureConfig(name="interaction")
        super().__init__(config)

        self.interaction_threshold = interaction_threshold
        self._feature_dim = 6

        # 모든 트랙의 현재 위치 저장
        self._all_positions: Dict[int, Tuple[float, float]] = {}
        self._interaction_history: Dict[int, deque] = {}

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def feature_type(self) -> str:
        return "behavior"

    def extract(self, image: np.ndarray, context: TrackContext = None) -> FeatureOutput:
        if context is None or context.bbox is None:
            return self.get_zero_feature()

        track_id = context.track_id
        bbox = context.bbox

        # 현재 위치 계산
        if len(bbox) >= 4:
            if bbox[2] > bbox[0]:
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
            else:
                cx = bbox[0] + bbox[2] / 2
                cy = bbox[1] + bbox[3] / 2
        else:
            return self.get_zero_feature()

        # 위치 업데이트
        self._all_positions[track_id] = (cx, cy)

        # 상호작용 히스토리 초기화
        if track_id not in self._interaction_history:
            self._interaction_history[track_id] = deque(maxlen=30)

        # 다른 객체와의 거리 계산
        distances_to_others = []
        for other_id, other_pos in self._all_positions.items():
            if other_id != track_id:
                dist = np.sqrt((cx - other_pos[0]) ** 2 + (cy - other_pos[1]) ** 2)
                distances_to_others.append(dist)

        # 상호작용 정보 기록
        interaction_info = {
            'distances': distances_to_others,
            'closest': min(distances_to_others) if distances_to_others else float('inf'),
            'num_nearby': sum(1 for d in distances_to_others if d < self.interaction_threshold)
        }
        self._interaction_history[track_id].append(interaction_info)

        # 특징 추출
        features = self._compute_interaction_features(
            self._interaction_history[track_id]
        )

        return FeatureOutput(
            feature=features,
            feature_type=self.feature_type,
            confidence=1.0 if distances_to_others else 0.5
        )

    def _compute_interaction_features(self, history: deque) -> np.ndarray:
        """상호작용 특징 계산"""
        if not history:
            return np.zeros(self.feature_dim, dtype=np.float32)

        closest_distances = [h['closest'] for h in history if h['closest'] < float('inf')]
        num_nearby_history = [h['num_nearby'] for h in history]

        features = [
            min(closest_distances) / 200 if closest_distances else 1.0,  # 최소 거리 (정규화)
            np.mean(closest_distances) / 200 if closest_distances else 1.0,  # 평균 최소 거리
            np.max(num_nearby_history) if num_nearby_history else 0,  # 최대 근접 객체 수
            np.mean(num_nearby_history) if num_nearby_history else 0,  # 평균 근접 객체 수
            sum(1 for d in closest_distances if d < self.interaction_threshold) / max(len(history), 1),  # 상호작용 빈도
            np.std(closest_distances) / 100 if len(closest_distances) > 1 else 0  # 거리 변동성
        ]

        feature_vec = np.array(features, dtype=np.float32)
        feature_vec = np.nan_to_num(feature_vec, nan=0.0, posinf=1.0, neginf=0.0)

        return self.normalize(feature_vec)

    def reset_state(self, track_ids: set = None):
        if track_ids is None:
            self._all_positions.clear()
            self._interaction_history.clear()
        else:
            for tid in list(self._all_positions.keys()):
                if tid not in track_ids:
                    del self._all_positions[tid]

            for tid in list(self._interaction_history.keys()):
                if tid not in track_ids:
                    del self._interaction_history[tid]


class BehaviorPatternExtractor(FeatureExtractor):
    """
    종합 행동 패턴 추출기

    ActivityLevel, Posture, Interaction을 결합
    """

    def __init__(self, config: FeatureConfig = None):
        config = config or FeatureConfig(name="behavior_pattern")
        super().__init__(config)

        self.activity_extractor = ActivityLevelExtractor()
        self.posture_extractor = PostureFeatureExtractor()
        self.interaction_extractor = InteractionFeatureExtractor()

        self._feature_dim = (
            self.activity_extractor.feature_dim +
            self.posture_extractor.feature_dim +
            self.interaction_extractor.feature_dim
        )

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def feature_type(self) -> str:
        return "behavior"

    def extract(self, image: np.ndarray, context: TrackContext = None) -> FeatureOutput:
        activity_out = self.activity_extractor.extract(image, context)
        posture_out = self.posture_extractor.extract(image, context)
        interaction_out = self.interaction_extractor.extract(image, context)

        # 결합
        combined = np.concatenate([
            activity_out.feature,
            posture_out.feature,
            interaction_out.feature
        ])

        avg_confidence = (
            activity_out.confidence +
            posture_out.confidence +
            interaction_out.confidence
        ) / 3

        combined = self.normalize(combined)

        return FeatureOutput(
            feature=combined,
            feature_type=self.feature_type,
            confidence=avg_confidence,
            metadata={
                'activity_confidence': activity_out.confidence,
                'posture_confidence': posture_out.confidence,
                'interaction_confidence': interaction_out.confidence
            }
        )

    def reset_state(self, track_ids: set = None):
        self.activity_extractor.reset_state(track_ids)
        self.posture_extractor.reset_state(track_ids)
        self.interaction_extractor.reset_state(track_ids)
