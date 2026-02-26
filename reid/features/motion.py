"""
Motion Feature Extractors

모션/동작 기반 특징 추출기들 - 이동 패턴, 속도, 방향 등
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from collections import deque
import logging

from .base import FeatureExtractor, FeatureConfig, TrackContext, FeatureOutput


class MotionFeatureExtractor(FeatureExtractor):
    """
    모션 특징 추출기

    이동 패턴, 속도, 방향 등 시간적 특징 추출
    """

    def __init__(self, config: FeatureConfig = None,
                 history_length: int = 10,
                 include_velocity: bool = True,
                 include_acceleration: bool = True,
                 include_direction: bool = True,
                 include_shape_change: bool = True,
                 expected_fps: float = 30.0,
                 gap_threshold_multiplier: float = 2.5):
        config = config or FeatureConfig(name="motion")
        super().__init__(config)

        self.history_length = history_length
        self.include_velocity = include_velocity
        self.include_acceleration = include_acceleration
        self.include_direction = include_direction
        self.include_shape_change = include_shape_change

        # 프레임 갭 감지 설정
        self.expected_interval = 1.0 / max(expected_fps, 1.0)
        self.gap_threshold = self.expected_interval * gap_threshold_multiplier

        # 특징 차원 계산
        dim = 0
        if include_velocity:
            dim += 4  # vx, vy, speed, speed_std
        if include_acceleration:
            dim += 3  # ax, ay, magnitude
        if include_direction:
            dim += 4  # direction_mean, direction_std, direction_change, consistency
        if include_shape_change:
            dim += 4  # aspect_ratio_change, size_change, mean_aspect, mean_size

        self._feature_dim = max(dim, 1)

        # 트랙별 히스토리
        self._track_histories: Dict[int, deque] = {}

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def feature_type(self) -> str:
        return "motion"

    @property
    def requires_history(self) -> bool:
        return True

    def _get_center_and_size(self, bbox: Tuple) -> Tuple[float, float, float, float]:
        """바운딩 박스에서 중심점과 크기 추출"""
        if len(bbox) == 4:
            # (x, y, w, h) 또는 (x1, y1, x2, y2) 형식 판단
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                # (x1, y1, x2, y2)
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
            else:
                # (x, y, w, h)
                x, y, w, h = bbox
                cx = x + w / 2
                cy = y + h / 2
        else:
            cx, cy, w, h = 0, 0, 1, 1

        return cx, cy, max(w, 1), max(h, 1)

    def extract(self, image: np.ndarray, context: TrackContext = None) -> FeatureOutput:
        if context is None or context.bbox is None:
            return self.get_zero_feature()

        track_id = context.track_id
        current_bbox = context.bbox

        # 히스토리 초기화
        if track_id not in self._track_histories:
            self._track_histories[track_id] = deque(maxlen=self.history_length)

        history = self._track_histories[track_id]

        # 현재 상태 추출
        cx, cy, w, h = self._get_center_and_size(current_bbox)
        current_state = {
            'cx': cx, 'cy': cy, 'w': w, 'h': h,
            'aspect_ratio': w / h,
            'size': w * h,
            'frame_idx': context.frame_idx,
            'timestamp': time.monotonic()
        }

        history.append(current_state)

        # 히스토리가 충분하지 않으면 제로 특징 반환
        if len(history) < 2:
            return FeatureOutput(
                feature=np.zeros(self.feature_dim, dtype=np.float32),
                feature_type=self.feature_type,
                confidence=0.3,
                metadata={'history_length': len(history)}
            )

        features = []

        # 속도 특징
        if self.include_velocity:
            velocities = self._compute_velocities(history)
            features.extend(velocities)

        # 가속도 특징
        if self.include_acceleration:
            accelerations = self._compute_accelerations(history)
            features.extend(accelerations)

        # 방향 특징
        if self.include_direction:
            directions = self._compute_directions(history)
            features.extend(directions)

        # 형태 변화 특징
        if self.include_shape_change:
            shape_changes = self._compute_shape_changes(history)
            features.extend(shape_changes)

        feature_vec = np.array(features, dtype=np.float32)

        # NaN/Inf 처리
        feature_vec = np.nan_to_num(feature_vec, nan=0.0, posinf=1.0, neginf=-1.0)

        # 정규화 (선택적 - 모션 특징은 스케일이 중요할 수 있음)
        feature_vec = self.normalize(feature_vec)

        # 갭 비율 기반 confidence 감소
        gaps = self._detect_gaps(history)
        gap_count = sum(gaps)
        gap_ratio = gap_count / max(len(gaps), 1)
        base_confidence = min(len(history) / self.history_length, 1.0)
        confidence = base_confidence * (1.0 - 0.7 * gap_ratio)

        return FeatureOutput(
            feature=feature_vec,
            feature_type=self.feature_type,
            confidence=confidence,
            metadata={
                'history_length': len(history),
                'gap_count': gap_count,
                'gap_ratio': gap_ratio
            }
        )

    def _detect_gaps(self, history: deque) -> List[bool]:
        """연속 히스토리 엔트리 간 프레임 갭 감지.
        gaps[i] = True → history[i]와 history[i+1] 사이에 프레임 손실 존재."""
        if len(history) < 2:
            return []
        gaps = []
        for i in range(len(history) - 1):
            t0 = history[i].get('timestamp', 0)
            t1 = history[i + 1].get('timestamp', 0)
            dt_sec = max(t1 - t0, 0)
            gaps.append(dt_sec > self.gap_threshold)
        return gaps

    def _compute_velocities(self, history: deque) -> List[float]:
        """속도 관련 특징 계산 (프레임 갭 걸친 쌍은 스킵)"""
        if len(history) < 2:
            return [0.0, 0.0, 0.0, 0.0]

        gaps = self._detect_gaps(history)
        velocities_x = []
        velocities_y = []

        for i in range(1, len(history)):
            if gaps[i - 1]:
                continue
            prev, curr = history[i - 1], history[i]
            dt = max(curr['frame_idx'] - prev['frame_idx'], 1)
            vx = (curr['cx'] - prev['cx']) / dt
            vy = (curr['cy'] - prev['cy']) / dt
            velocities_x.append(vx)
            velocities_y.append(vy)

        if not velocities_x:
            return [0.0, 0.0, 0.0, 0.0]

        speeds = [np.sqrt(vx ** 2 + vy ** 2) for vx, vy in zip(velocities_x, velocities_y)]

        return [
            np.mean(velocities_x),
            np.mean(velocities_y),
            np.mean(speeds),
            np.std(speeds) if len(speeds) > 1 else 0.0
        ]

    def _compute_accelerations(self, history: deque) -> List[float]:
        """가속도 관련 특징 계산 (프레임 갭 걸친 트리플렛 스킵)"""
        if len(history) < 3:
            return [0.0, 0.0, 0.0]

        gaps = self._detect_gaps(history)
        accelerations_x = []
        accelerations_y = []

        for i in range(2, len(history)):
            if gaps[i - 2] or gaps[i - 1]:
                continue
            h0, h1, h2 = history[i - 2], history[i - 1], history[i]
            dt1 = max(h1['frame_idx'] - h0['frame_idx'], 1)
            dt2 = max(h2['frame_idx'] - h1['frame_idx'], 1)

            vx1 = (h1['cx'] - h0['cx']) / dt1
            vy1 = (h1['cy'] - h0['cy']) / dt1
            vx2 = (h2['cx'] - h1['cx']) / dt2
            vy2 = (h2['cy'] - h1['cy']) / dt2

            ax = (vx2 - vx1) / dt2
            ay = (vy2 - vy1) / dt2

            accelerations_x.append(ax)
            accelerations_y.append(ay)

        if not accelerations_x:
            return [0.0, 0.0, 0.0]

        mags = [np.sqrt(ax ** 2 + ay ** 2) for ax, ay in zip(accelerations_x, accelerations_y)]

        return [
            np.mean(accelerations_x),
            np.mean(accelerations_y),
            np.mean(mags)
        ]

    def _compute_directions(self, history: deque) -> List[float]:
        """이동 방향 관련 특징 계산 (프레임 갭 걸친 쌍 스킵)"""
        if len(history) < 2:
            return [0.0, 0.0, 0.0, 0.0]

        gaps = self._detect_gaps(history)
        directions = []

        for i in range(1, len(history)):
            if gaps[i - 1]:
                continue
            prev, curr = history[i - 1], history[i]
            dx = curr['cx'] - prev['cx']
            dy = curr['cy'] - prev['cy']

            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                angle = np.arctan2(dy, dx)
                directions.append(angle)

        if not directions:
            return [0.0, 0.0, 0.0, 1.0]

        # 방향 변화량
        direction_changes = []
        for i in range(1, len(directions)):
            diff = abs(directions[i] - directions[i - 1])
            if diff > np.pi:
                diff = 2 * np.pi - diff
            direction_changes.append(diff)

        # 일관성: 방향 변화가 작을수록 높음
        consistency = 1.0 / (1.0 + np.mean(direction_changes)) if direction_changes else 1.0

        return [
            np.mean(directions),
            np.std(directions) if len(directions) > 1 else 0.0,
            np.mean(direction_changes) if direction_changes else 0.0,
            consistency
        ]

    def _compute_shape_changes(self, history: deque) -> List[float]:
        """형태 변화 특징 계산"""
        if len(history) < 2:
            return [0.0, 0.0, 1.0, 1.0]

        aspect_ratios = [h['aspect_ratio'] for h in history]
        sizes = [h['size'] for h in history]

        # 변화율
        ar_changes = [abs(aspect_ratios[i] - aspect_ratios[i - 1])
                      for i in range(1, len(aspect_ratios))]
        size_changes = [abs(sizes[i] - sizes[i - 1]) / max(sizes[i - 1], 1)
                        for i in range(1, len(sizes))]

        return [
            np.mean(ar_changes),
            np.mean(size_changes),
            np.mean(aspect_ratios),
            np.mean(sizes) / 10000  # 정규화
        ]

    def reset_state(self, track_ids: set = None):
        if track_ids is None:
            self._track_histories.clear()
        else:
            active_ids = set(self._track_histories.keys())
            for tid in active_ids - track_ids:
                del self._track_histories[tid]


class OpticalFlowFeatureExtractor(FeatureExtractor):
    """
    Optical Flow 기반 모션 특징 추출

    이미지 시퀀스에서 optical flow를 계산하여 모션 패턴 추출
    """

    def __init__(self, config: FeatureConfig = None,
                 grid_size: int = 4,
                 flow_bins: int = 8,
                 expected_fps: float = 30.0,
                 gap_threshold_multiplier: float = 2.5):
        config = config or FeatureConfig(name="optical_flow")
        super().__init__(config)

        self.grid_size = grid_size
        self.flow_bins = flow_bins

        # 각 그리드 셀에서 flow magnitude histogram (flow_bins) + dominant direction (1)
        self._feature_dim = grid_size * grid_size * (flow_bins + 1)

        # 이전 프레임 저장
        self._prev_grays: Dict[int, np.ndarray] = {}
        # 프레임 갭 감지용 타임스탬프
        self._prev_timestamps: Dict[int, float] = {}
        self.gap_threshold = (1.0 / max(expected_fps, 1.0)) * gap_threshold_multiplier

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def feature_type(self) -> str:
        return "motion"

    def extract(self, image: np.ndarray, context: TrackContext = None) -> FeatureOutput:
        if image is None or image.size == 0:
            return self.get_zero_feature()

        track_id = context.track_id if context else 0

        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))

        now = time.monotonic()

        if track_id not in self._prev_grays:
            self._prev_grays[track_id] = gray
            self._prev_timestamps[track_id] = now
            return self.get_zero_feature()

        prev_gray = self._prev_grays[track_id]
        prev_ts = self._prev_timestamps.get(track_id, now)
        dt_sec = max(now - prev_ts, 0)

        # 프레임 갭 감지: 비연속 프레임 간 optical flow는 신뢰 불가
        if dt_sec > self.gap_threshold:
            self._prev_grays[track_id] = gray
            self._prev_timestamps[track_id] = now
            return FeatureOutput(
                feature=np.zeros(self.feature_dim, dtype=np.float32),
                feature_type=self.feature_type,
                confidence=0.1,
                metadata={'gap_detected': True, 'dt_sec': dt_sec}
            )

        try:
            # Optical flow 계산 (Farneback)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            # 그리드별 특징 추출
            features = []
            h, w = flow.shape[:2]
            cell_h, cell_w = h // self.grid_size, w // self.grid_size

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    cell_flow = flow[i * cell_h:(i + 1) * cell_h,
                                    j * cell_w:(j + 1) * cell_w]

                    # Magnitude와 방향
                    fx, fy = cell_flow[..., 0], cell_flow[..., 1]
                    magnitude = np.sqrt(fx ** 2 + fy ** 2)
                    angle = np.arctan2(fy, fx)

                    # Magnitude histogram
                    mag_hist, _ = np.histogram(magnitude.flatten(),
                                               bins=self.flow_bins,
                                               range=(0, 10))
                    mag_hist = mag_hist.astype(np.float32)
                    if mag_hist.sum() > 0:
                        mag_hist /= mag_hist.sum()

                    # Dominant direction
                    weighted_angles = angle * magnitude
                    dom_direction = np.mean(weighted_angles) if magnitude.sum() > 0 else 0

                    features.extend(mag_hist)
                    features.append(dom_direction)

            self._prev_grays[track_id] = gray
            self._prev_timestamps[track_id] = now

            feature_vec = np.array(features, dtype=np.float32)
            feature_vec = self.normalize(feature_vec)

            return FeatureOutput(
                feature=feature_vec,
                feature_type=self.feature_type,
                confidence=1.0
            )

        except Exception as e:
            self.logger.warning(f"Optical flow extraction failed: {e}")
            self._prev_grays[track_id] = gray
            self._prev_timestamps[track_id] = now
            return self.get_zero_feature()

    def reset_state(self, track_ids: set = None):
        if track_ids is None:
            self._prev_grays.clear()
            self._prev_timestamps.clear()
        else:
            active_ids = set(self._prev_grays.keys())
            for tid in active_ids - track_ids:
                self._prev_grays.pop(tid, None)
                self._prev_timestamps.pop(tid, None)


class TrajectoryFeatureExtractor(FeatureExtractor):
    """
    궤적 패턴 특징 추출

    이동 궤적의 형태를 분석하여 특징 추출 (직선, 곡선, 정지 등)
    """

    def __init__(self, config: FeatureConfig = None,
                 history_length: int = 30,
                 num_trajectory_points: int = 8,
                 expected_fps: float = 30.0,
                 gap_threshold_multiplier: float = 2.5):
        config = config or FeatureConfig(name="trajectory")
        super().__init__(config)

        self.history_length = history_length
        self.num_trajectory_points = num_trajectory_points

        # 특징: 정규화된 궤적 (x, y) * num_points + curvature + linearity + compactness
        self._feature_dim = num_trajectory_points * 2 + 3

        self._track_positions: Dict[int, deque] = {}

        # 프레임 갭 감지 설정
        self.gap_threshold = (1.0 / max(expected_fps, 1.0)) * gap_threshold_multiplier

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def feature_type(self) -> str:
        return "motion"

    @property
    def requires_history(self) -> bool:
        return True

    def extract(self, image: np.ndarray, context: TrackContext = None) -> FeatureOutput:
        if context is None or context.bbox is None:
            return self.get_zero_feature()

        track_id = context.track_id
        bbox = context.bbox

        # 중심점 계산
        if len(bbox) >= 4:
            if bbox[2] > bbox[0]:  # (x1, y1, x2, y2)
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
            else:  # (x, y, w, h)
                cx = bbox[0] + bbox[2] / 2
                cy = bbox[1] + bbox[3] / 2
        else:
            return self.get_zero_feature()

        if track_id not in self._track_positions:
            self._track_positions[track_id] = deque(maxlen=self.history_length)

        self._track_positions[track_id].append((cx, cy, time.monotonic()))

        raw_positions = list(self._track_positions[track_id])

        if len(raw_positions) < 3:
            return self.get_zero_feature()

        try:
            # 프레임 갭 기준으로 연속 세그먼트 분할
            segments = [[raw_positions[0]]]
            for i in range(1, len(raw_positions)):
                dt = raw_positions[i][2] - raw_positions[i - 1][2]
                if dt > self.gap_threshold:
                    segments.append([raw_positions[i]])
                else:
                    segments[-1].append(raw_positions[i])

            # 가장 긴 연속 세그먼트 사용
            best_segment = max(segments, key=len)

            if len(best_segment) < 3:
                return self.get_zero_feature()

            # (cx, cy)만 추출
            positions = np.array([(p[0], p[1]) for p in best_segment])

            # 중심으로 이동
            center = positions.mean(axis=0)
            normalized = positions - center

            # 스케일 정규화
            scale = np.max(np.abs(normalized)) if np.max(np.abs(normalized)) > 0 else 1.0
            normalized = normalized / scale

            # 리샘플링
            indices = np.linspace(0, len(normalized) - 1, self.num_trajectory_points)
            resampled = []
            for idx in indices:
                lower = int(np.floor(idx))
                upper = min(lower + 1, len(normalized) - 1)
                t = idx - lower
                point = normalized[lower] * (1 - t) + normalized[upper] * t
                resampled.append(point)

            resampled = np.array(resampled).flatten()

            # 곡률 계산
            curvature = self._compute_curvature(positions)

            # 선형성 (직선에 얼마나 가까운지)
            linearity = self._compute_linearity(positions)

            # 컴팩트함 (얼마나 한 곳에 머무는지)
            compactness = self._compute_compactness(positions)

            features = np.concatenate([
                resampled,
                [curvature, linearity, compactness]
            ]).astype(np.float32)

            features = self.normalize(features)

            # 히스토리 커버리지 + 갭 분할 패널티 반영
            coverage = min(len(best_segment) / self.history_length, 1.0)
            frag_penalty = 1.0 - (len(segments) - 1) / max(len(raw_positions) - 1, 1)
            confidence = coverage * max(frag_penalty, 0.3)

            return FeatureOutput(
                feature=features,
                feature_type=self.feature_type,
                confidence=confidence,
                metadata={
                    'segment_count': len(segments),
                    'best_segment_len': len(best_segment)
                }
            )

        except Exception as e:
            self.logger.warning(f"Trajectory feature extraction failed: {e}")
            return self.get_zero_feature()

    def _compute_curvature(self, positions: np.ndarray) -> float:
        """평균 곡률 계산"""
        if len(positions) < 3:
            return 0.0

        curvatures = []
        for i in range(1, len(positions) - 1):
            p0, p1, p2 = positions[i - 1], positions[i], positions[i + 1]

            v1 = p1 - p0
            v2 = p2 - p1

            cross = np.cross(v1, v2)
            dot = np.dot(v1, v2)

            angle = np.arctan2(abs(cross), dot)
            curvatures.append(angle)

        return np.mean(curvatures) if curvatures else 0.0

    def _compute_linearity(self, positions: np.ndarray) -> float:
        """직선성 계산 (0=곡선, 1=직선)"""
        if len(positions) < 2:
            return 1.0

        # 시작점-끝점 직선 거리
        direct_dist = np.linalg.norm(positions[-1] - positions[0])

        # 실제 이동 거리
        total_dist = sum(np.linalg.norm(positions[i + 1] - positions[i])
                         for i in range(len(positions) - 1))

        if total_dist < 1e-6:
            return 1.0

        return direct_dist / total_dist

    def _compute_compactness(self, positions: np.ndarray) -> float:
        """컴팩트함 계산 (분산의 역수)"""
        if len(positions) < 2:
            return 1.0

        variance = np.var(positions)
        return 1.0 / (1.0 + variance)

    def reset_state(self, track_ids: set = None):
        if track_ids is None:
            self._track_positions.clear()
        else:
            active_ids = set(self._track_positions.keys())
            for tid in active_ids - track_ids:
                del self._track_positions[tid]
