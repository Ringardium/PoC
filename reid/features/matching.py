"""
Matching Strategy Module

다양한 매칭 전략 구현 - 특징 기반 유사도 계산 및 매칭
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict
import logging
from scipy.optimize import linear_sum_assignment


@dataclass
class MatchResult:
    """매칭 결과"""
    matched_pairs: List[Tuple[int, int]]  # (query_idx, gallery_idx)
    unmatched_queries: List[int]
    unmatched_galleries: List[int]
    similarities: Dict[Tuple[int, int], float]  # 매칭된 쌍의 유사도
    metadata: Dict[str, Any] = field(default_factory=dict)


class MatchingStrategy(ABC):
    """매칭 전략 인터페이스"""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """두 특징 간 유사도 계산"""
        pass

    @abstractmethod
    def compute_similarity_matrix(self, queries: np.ndarray,
                                   galleries: np.ndarray) -> np.ndarray:
        """쿼리-갤러리 유사도 행렬 계산"""
        pass

    @abstractmethod
    def match(self, queries: np.ndarray, galleries: np.ndarray,
              query_ids: List[int] = None,
              gallery_ids: List[int] = None) -> MatchResult:
        """
        쿼리와 갤러리 매칭

        Args:
            queries: 쿼리 특징 배열 (N, D)
            galleries: 갤러리 특징 배열 (M, D)
            query_ids: 쿼리 ID (선택)
            gallery_ids: 갤러리 ID (선택)

        Returns:
            MatchResult
        """
        pass


class CosineSimilarityMatching(MatchingStrategy):
    """코사인 유사도 기반 매칭"""

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        if feat1 is None or feat2 is None:
            return 0.0

        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)

        if norm1 < 1e-7 or norm2 < 1e-7:
            return 0.0

        return float(np.dot(feat1, feat2) / (norm1 * norm2))

    def compute_similarity_matrix(self, queries: np.ndarray,
                                   galleries: np.ndarray) -> np.ndarray:
        if len(queries) == 0 or len(galleries) == 0:
            return np.array([])

        # 정규화
        q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
        g_norms = np.linalg.norm(galleries, axis=1, keepdims=True)

        q_norms[q_norms < 1e-7] = 1.0
        g_norms[g_norms < 1e-7] = 1.0

        queries_norm = queries / q_norms
        galleries_norm = galleries / g_norms

        # 유사도 행렬
        return np.dot(queries_norm, galleries_norm.T)

    def match(self, queries: np.ndarray, galleries: np.ndarray,
              query_ids: List[int] = None,
              gallery_ids: List[int] = None) -> MatchResult:

        if len(queries) == 0:
            return MatchResult(
                matched_pairs=[],
                unmatched_queries=[],
                unmatched_galleries=list(range(len(galleries))),
                similarities={}
            )

        if len(galleries) == 0:
            return MatchResult(
                matched_pairs=[],
                unmatched_queries=list(range(len(queries))),
                unmatched_galleries=[],
                similarities={}
            )

        # 유사도 행렬 계산
        sim_matrix = self.compute_similarity_matrix(queries, galleries)

        # Hungarian algorithm으로 최적 매칭
        cost_matrix = 1 - sim_matrix  # 유사도 -> 비용
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched_pairs = []
        similarities = {}
        matched_queries = set()
        matched_galleries = set()

        for q_idx, g_idx in zip(row_indices, col_indices):
            sim = sim_matrix[q_idx, g_idx]

            if sim >= self.threshold:
                matched_pairs.append((q_idx, g_idx))
                similarities[(q_idx, g_idx)] = float(sim)
                matched_queries.add(q_idx)
                matched_galleries.add(g_idx)

        unmatched_queries = [i for i in range(len(queries)) if i not in matched_queries]
        unmatched_galleries = [i for i in range(len(galleries)) if i not in matched_galleries]

        return MatchResult(
            matched_pairs=matched_pairs,
            unmatched_queries=unmatched_queries,
            unmatched_galleries=unmatched_galleries,
            similarities=similarities,
            metadata={'method': 'cosine_hungarian'}
        )


class EuclideanDistanceMatching(MatchingStrategy):
    """유클리드 거리 기반 매칭"""

    def __init__(self, threshold: float = 0.5, max_distance: float = 2.0):
        super().__init__(threshold)
        self.max_distance = max_distance

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        if feat1 is None or feat2 is None:
            return 0.0

        distance = np.linalg.norm(feat1 - feat2)
        # 거리를 유사도로 변환 (0 ~ 1)
        similarity = max(0, 1 - distance / self.max_distance)
        return float(similarity)

    def compute_similarity_matrix(self, queries: np.ndarray,
                                   galleries: np.ndarray) -> np.ndarray:
        if len(queries) == 0 or len(galleries) == 0:
            return np.array([])

        # 거리 행렬 계산
        # (q - g)^2 = q^2 + g^2 - 2*q*g
        q_sq = np.sum(queries ** 2, axis=1, keepdims=True)
        g_sq = np.sum(galleries ** 2, axis=1, keepdims=True)
        cross = np.dot(queries, galleries.T)

        dist_sq = q_sq + g_sq.T - 2 * cross
        dist_sq = np.maximum(dist_sq, 0)  # 수치 안정성
        distances = np.sqrt(dist_sq)

        # 유사도로 변환
        similarities = np.maximum(0, 1 - distances / self.max_distance)
        return similarities

    def match(self, queries: np.ndarray, galleries: np.ndarray,
              query_ids: List[int] = None,
              gallery_ids: List[int] = None) -> MatchResult:
        # Cosine과 동일한 로직 사용
        return CosineSimilarityMatching.match(self, queries, galleries,
                                               query_ids, gallery_ids)


class CascadeMatching(MatchingStrategy):
    """
    Cascade 매칭 - 여러 전략을 순차적으로 적용

    첫 번째 전략에서 매칭되지 않은 것들을 다음 전략으로 시도
    """

    def __init__(self, strategies: List[MatchingStrategy],
                 threshold: float = 0.5):
        super().__init__(threshold)
        self.strategies = strategies

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        # 첫 번째 전략 사용
        if self.strategies:
            return self.strategies[0].compute_similarity(feat1, feat2)
        return 0.0

    def compute_similarity_matrix(self, queries: np.ndarray,
                                   galleries: np.ndarray) -> np.ndarray:
        if self.strategies:
            return self.strategies[0].compute_similarity_matrix(queries, galleries)
        return np.array([])

    def match(self, queries: np.ndarray, galleries: np.ndarray,
              query_ids: List[int] = None,
              gallery_ids: List[int] = None) -> MatchResult:

        all_matched_pairs = []
        all_similarities = {}

        remaining_queries = list(range(len(queries)))
        remaining_galleries = list(range(len(galleries)))

        for i, strategy in enumerate(self.strategies):
            if not remaining_queries or not remaining_galleries:
                break

            # 남은 쿼리와 갤러리로 매칭
            sub_queries = queries[remaining_queries]
            sub_galleries = galleries[remaining_galleries]

            result = strategy.match(sub_queries, sub_galleries)

            # 원래 인덱스로 변환
            for q_sub, g_sub in result.matched_pairs:
                q_orig = remaining_queries[q_sub]
                g_orig = remaining_galleries[g_sub]
                all_matched_pairs.append((q_orig, g_orig))
                all_similarities[(q_orig, g_orig)] = result.similarities.get(
                    (q_sub, g_sub), 0.0)

            # 매칭된 것들 제거
            matched_q = {remaining_queries[q] for q, _ in result.matched_pairs}
            matched_g = {remaining_galleries[g] for _, g in result.matched_pairs}

            remaining_queries = [q for q in remaining_queries if q not in matched_q]
            remaining_galleries = [g for g in remaining_galleries if g not in matched_g]

        return MatchResult(
            matched_pairs=all_matched_pairs,
            unmatched_queries=remaining_queries,
            unmatched_galleries=remaining_galleries,
            similarities=all_similarities,
            metadata={'method': 'cascade', 'stages': len(self.strategies)}
        )


class WeightedFeatureMatching(MatchingStrategy):
    """
    가중 특징 매칭 - 특징의 여러 부분에 다른 가중치 적용

    예: appearance(0~100) + motion(100~150)에서
        appearance 가중치 0.7, motion 가중치 0.3
    """

    def __init__(self, feature_ranges: Dict[str, Tuple[int, int]],
                 feature_weights: Dict[str, float],
                 threshold: float = 0.5):
        super().__init__(threshold)
        self.feature_ranges = feature_ranges  # 이름 -> (시작, 끝)
        self.feature_weights = feature_weights  # 이름 -> 가중치

        # 가중치 정규화
        total_weight = sum(feature_weights.values())
        if total_weight > 0:
            self.normalized_weights = {k: v / total_weight
                                       for k, v in feature_weights.items()}
        else:
            self.normalized_weights = feature_weights

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        if feat1 is None or feat2 is None:
            return 0.0

        total_sim = 0.0

        for name, (start, end) in self.feature_ranges.items():
            weight = self.normalized_weights.get(name, 1.0)

            sub_feat1 = feat1[start:end]
            sub_feat2 = feat2[start:end]

            # 각 부분의 코사인 유사도
            norm1 = np.linalg.norm(sub_feat1)
            norm2 = np.linalg.norm(sub_feat2)

            if norm1 > 1e-7 and norm2 > 1e-7:
                sim = np.dot(sub_feat1, sub_feat2) / (norm1 * norm2)
            else:
                sim = 0.0

            total_sim += weight * sim

        return float(total_sim)

    def compute_similarity_matrix(self, queries: np.ndarray,
                                   galleries: np.ndarray) -> np.ndarray:
        if len(queries) == 0 or len(galleries) == 0:
            return np.array([])

        total_sim = np.zeros((len(queries), len(galleries)))

        for name, (start, end) in self.feature_ranges.items():
            weight = self.normalized_weights.get(name, 1.0)

            sub_q = queries[:, start:end]
            sub_g = galleries[:, start:end]

            # 정규화
            q_norms = np.linalg.norm(sub_q, axis=1, keepdims=True)
            g_norms = np.linalg.norm(sub_g, axis=1, keepdims=True)

            q_norms[q_norms < 1e-7] = 1.0
            g_norms[g_norms < 1e-7] = 1.0

            sub_q_norm = sub_q / q_norms
            sub_g_norm = sub_g / g_norms

            sim = np.dot(sub_q_norm, sub_g_norm.T)
            total_sim += weight * sim

        return total_sim

    def match(self, queries: np.ndarray, galleries: np.ndarray,
              query_ids: List[int] = None,
              gallery_ids: List[int] = None) -> MatchResult:
        return CosineSimilarityMatching.match(self, queries, galleries,
                                               query_ids, gallery_ids)


class GreedyMatching(MatchingStrategy):
    """
    Greedy 매칭 - 가장 높은 유사도부터 순차적으로 매칭

    Hungarian보다 빠르지만 최적해 보장 안됨
    """

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        if feat1 is None or feat2 is None:
            return 0.0
        return float(np.dot(feat1, feat2))

    def compute_similarity_matrix(self, queries: np.ndarray,
                                   galleries: np.ndarray) -> np.ndarray:
        if len(queries) == 0 or len(galleries) == 0:
            return np.array([])
        return np.dot(queries, galleries.T)

    def match(self, queries: np.ndarray, galleries: np.ndarray,
              query_ids: List[int] = None,
              gallery_ids: List[int] = None) -> MatchResult:

        if len(queries) == 0 or len(galleries) == 0:
            return MatchResult(
                matched_pairs=[],
                unmatched_queries=list(range(len(queries))),
                unmatched_galleries=list(range(len(galleries))),
                similarities={}
            )

        sim_matrix = self.compute_similarity_matrix(queries, galleries)

        matched_pairs = []
        similarities = {}
        used_queries = set()
        used_galleries = set()

        # 모든 유사도를 정렬
        indices = np.dstack(np.unravel_index(
            np.argsort(sim_matrix.ravel())[::-1], sim_matrix.shape))[0]

        for q_idx, g_idx in indices:
            if q_idx in used_queries or g_idx in used_galleries:
                continue

            sim = sim_matrix[q_idx, g_idx]
            if sim < self.threshold:
                break

            matched_pairs.append((q_idx, g_idx))
            similarities[(q_idx, g_idx)] = float(sim)
            used_queries.add(q_idx)
            used_galleries.add(g_idx)

        unmatched_queries = [i for i in range(len(queries)) if i not in used_queries]
        unmatched_galleries = [i for i in range(len(galleries)) if i not in used_galleries]

        return MatchResult(
            matched_pairs=matched_pairs,
            unmatched_queries=unmatched_queries,
            unmatched_galleries=unmatched_galleries,
            similarities=similarities,
            metadata={'method': 'greedy'}
        )


def create_matching_strategy(method: str = 'cosine',
                             threshold: float = 0.5,
                             **kwargs) -> MatchingStrategy:
    """매칭 전략 팩토리"""
    strategies = {
        'cosine': CosineSimilarityMatching,
        'euclidean': EuclideanDistanceMatching,
        'greedy': GreedyMatching,
    }

    if method not in strategies:
        raise ValueError(f"Unknown matching method: {method}")

    return strategies[method](threshold=threshold, **kwargs)
