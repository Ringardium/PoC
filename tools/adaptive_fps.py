"""
적응형 FPS 컨트롤러

객체 수, 움직임 크기, 이벤트 발생 여부에 따라 분석 FPS를 자동 조절하여
GPU 사용률을 최적화합니다.

처리시간 기반 프레임 skip(ProcessingConfig.enable_adaptive_skip)과 달리,
이 모듈은 *컨텐츠* 기반으로 YOLO 추론 자체를 건너뛸지 결정합니다.
두 레이어가 독립적으로 작동하여 상호 보완합니다.

사용 예:
    controller = AdaptiveFPSController(max_fps=10, min_fps=0.1, idle_fps=1.0)

    while True:
        if controller.should_analyze(last_analysis_time):
            # YOLO + 트래킹 + 행동분석 실행
            last_analysis_time = time.time()
            controller.update(num_objects, avg_displacement, has_event)
"""

import time


class AdaptiveFPSController:
    """
    객체 수/움직임/이벤트에 따라 분석 FPS를 자동 조절

    상황별 동작:
    - 객체 없음 → min_fps (기본 0.1fps = 10초에 1회)
    - 객체 있음 + 움직임 적음 → idle_fps (기본 1.0fps)
    - 객체 있음 + 움직임 큼 → max_fps (기본 10fps)
    - 이벤트 감지 중 (싸움/이탈 등) → max_fps
    """

    def __init__(
        self,
        max_fps: float = 10.0,
        min_fps: float = 0.1,
        idle_fps: float = 1.0,
        displacement_low: float = 5.0,
        displacement_high: float = 50.0,
        no_object_grace_count: int = 5,
    ):
        """
        Args:
            max_fps: 최대 분석 FPS (활발할 때)
            min_fps: 최소 분석 FPS (객체 없을 때, 예: 0.1 = 10초에 1회)
            idle_fps: 유휴 분석 FPS (객체 있지만 조용할 때)
            displacement_low: 저움직임 임계값 (이하면 idle)
            displacement_high: 고움직임 임계값 (이상이면 max)
            no_object_grace_count: 객체 없음이 이 횟수 연속되면 min_fps로 전환
        """
        self.max_fps = max_fps
        self.min_fps = min_fps
        self.idle_fps = idle_fps
        self.displacement_low = displacement_low
        self.displacement_high = displacement_high
        self.no_object_grace_count = no_object_grace_count

        self._no_object_streak = 0
        self._current_fps = max_fps
        self._last_update_time = time.time()

    def update(self, num_objects: int, avg_displacement: float = 0.0, has_event: bool = False):
        """
        분석 결과에 따라 내부 상태를 업데이트

        Args:
            num_objects: 현재 감지된 객체 수
            avg_displacement: 평균 변위 (이전 프레임 대비, px/frame)
            has_event: 이벤트 감지 여부 (싸움, 이탈 등)
        """
        self._last_update_time = time.time()

        if has_event:
            self._current_fps = self.max_fps
            self._no_object_streak = 0
            return

        if num_objects == 0:
            self._no_object_streak += 1
            if self._no_object_streak >= self.no_object_grace_count:
                self._current_fps = self.min_fps
            else:
                # 처음 몇 번은 idle_fps 유지 (객체가 다시 나타날 수 있으므로)
                self._current_fps = self.idle_fps
            return

        self._no_object_streak = 0

        if avg_displacement < self.displacement_low:
            self._current_fps = self.idle_fps
        elif avg_displacement < self.displacement_high:
            # 선형 보간: idle_fps ~ max_fps
            ratio = (avg_displacement - self.displacement_low) / (
                self.displacement_high - self.displacement_low
            )
            self._current_fps = self.idle_fps + ratio * (self.max_fps - self.idle_fps)
        else:
            self._current_fps = self.max_fps

    def get_analysis_interval(self) -> float:
        """
        다음 분석까지 대기해야 할 시간 (초) 반환

        Returns:
            float: 분석 간격 (초). 예: 0.1 = 10fps, 10.0 = 0.1fps
        """
        return 1.0 / max(self._current_fps, 0.01)

    def should_analyze(self, last_analysis_time: float) -> bool:
        """
        현재 시점에서 분석을 실행해야 하는지 판단

        Args:
            last_analysis_time: 마지막 분석 실행 시각 (time.time())

        Returns:
            bool: 분석 실행 여부
        """
        return (time.time() - last_analysis_time) >= self.get_analysis_interval()

    @property
    def current_fps(self) -> float:
        """현재 적용 중인 분석 FPS"""
        return self._current_fps

    @property
    def current_interval(self) -> float:
        """현재 분석 간격 (초)"""
        return self.get_analysis_interval()

    def get_status(self) -> dict:
        """현재 상태 정보 반환 (디버깅/모니터링용)"""
        if self._no_object_streak >= self.no_object_grace_count:
            mode = "sleep"
        elif self._current_fps <= self.idle_fps:
            mode = "idle"
        elif self._current_fps >= self.max_fps:
            mode = "active"
        else:
            mode = "moderate"

        return {
            "mode": mode,
            "current_fps": round(self._current_fps, 2),
            "interval_ms": round(self.get_analysis_interval() * 1000, 1),
            "no_object_streak": self._no_object_streak,
        }
