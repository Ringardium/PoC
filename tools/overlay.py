"""
분석 결과 오버레이 유틸리티

분석/표시 분리 패턴에서, 캐싱된 분석 결과를 원본 프레임 위에 오버레이합니다.
YOLO는 N fps로 분석하되, 모든 프레임에 최신 결과를 그려서 부드러운 화면을 유지합니다.

사용 예:
    cached = None
    while True:
        frame = cap.read()
        if should_analyze:
            boxes, track_ids = track(frame)
            cached = build_overlay_cache(boxes, track_ids, ...)
        display = draw_cached_overlay(frame.copy(), cached)
        queue.put(display)
"""

import cv2
import numpy as np

# 색상 상수 (BGR)
RED     = (0, 0, 255)    # 싸움
GREEN   = (0, 255, 0)    # 비활동
BLUE    = (255, 0, 0)    # 이탈
YELLOW  = (0, 255, 255)  # 식사
CYAN    = (255, 255, 0)  # 밥그릇
ORANGE  = (0, 165, 255)  # 활동
GRAY    = (128, 128, 128) # 수면
PURPLE  = (255, 0, 128)  # 배변
WHITE   = (255, 255, 255)
BLACK   = (0, 0, 0)


def build_overlay_cache(
    boxes,
    track_ids,
    fight_boxes=None,
    escaped_boxes=None,
    inert_boxes=None,
    eating_boxes=None,
    bowl_boxes=None,
    sleep_boxes=None,
    bathroom_boxes=None,
    active_boxes=None,
    state_str: str = "",
    stream_id=None,
    frame_count: int = 0,
    num_objects: int = 0,
    fps_info: dict = None,
) -> dict:
    """
    분석 결과를 캐시 딕셔너리로 구성

    Args:
        boxes: 트래킹된 bbox 리스트 [(x_center, y_center, w, h), ...]
        track_ids: 트래킹 ID 리스트
        fight_boxes: 싸움 감지된 bbox 리스트
        escaped_boxes: 이탈 감지된 bbox 리스트
        inert_boxes: 비활동 감지된 bbox 리스트
        eating_boxes: 식사 감지된 bbox 리스트
        bowl_boxes: 밥그릇 bbox 리스트
        sleep_boxes: 수면 감지된 bbox 리스트
        bathroom_boxes: 배변 감지된 bbox 리스트
        active_boxes: 활동 감지된 bbox 리스트
        state_str: 상태 문자열 (예: "Fight Inert")
        stream_id: 스트림 ID
        frame_count: 프레임 카운트
        num_objects: 감지 객체 수
        fps_info: AdaptiveFPSController.get_status() 결과 딕셔너리

    Returns:
        dict: 캐시 딕셔너리
    """
    return {
        "boxes": boxes if boxes is not None else [],
        "track_ids": track_ids if track_ids is not None else [],
        "fight_boxes": fight_boxes or [],
        "escaped_boxes": escaped_boxes or [],
        "inert_boxes": inert_boxes or [],
        "eating_boxes": eating_boxes or [],
        "bowl_boxes": bowl_boxes or [],
        "sleep_boxes": sleep_boxes or [],
        "bathroom_boxes": bathroom_boxes or [],
        "active_boxes": active_boxes or [],
        "state_str": state_str,
        "stream_id": stream_id,
        "frame_count": frame_count,
        "num_objects": num_objects,
        "fps_info": fps_info,
    }


def draw_cached_overlay(frame: np.ndarray, cached: dict, show_info: bool = True) -> np.ndarray:
    """
    캐싱된 분석 결과를 프레임 위에 오버레이

    Args:
        frame: 원본 프레임 (in-place 수정. copy 필요 시 호출부에서 처리)
        cached: build_overlay_cache로 생성된 캐시 딕셔너리
        show_info: 상단 정보 텍스트 표시 여부

    Returns:
        frame: 오버레이가 적용된 프레임
    """
    if cached is None:
        return frame

    def _draw_rect(frame, box, color, thickness=3):
        x_center, y_center, w, h = box
        pt1 = (int(x_center - w / 2), int(y_center - h / 2))
        pt2 = (int(x_center + w / 2), int(y_center + h / 2))
        cv2.rectangle(frame, pt1, pt2, color, thickness)
        return pt1

    # 밥그릇 (시안, 얇은 테두리)
    for box in cached["bowl_boxes"]:
        _draw_rect(frame, box, CYAN, 2)

    # 트래킹 bbox + 중심점 + ID
    for box, track_id in zip(cached["boxes"], cached["track_ids"]):
        x_center, y_center, w, h = box
        cv2.circle(frame, (int(x_center), int(y_center)), 5, RED, 3, cv2.LINE_AA)
        pt1 = (int(x_center - w / 2), int(y_center - h / 2))
        cv2.putText(
            frame,
            f"id:{track_id}",
            (pt1[0] + 10, pt1[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA,
        )

    # 행동별 bbox (우선순위 순: fight > escape > eat > bathroom > sleep > active > inert)
    _BEHAVIOR_LAYERS = [
        ("fight_boxes",    RED,    "Fight"),
        ("escaped_boxes",  BLUE,   "Escape"),
        ("eating_boxes",   YELLOW, "Eat"),
        ("bathroom_boxes", PURPLE, "Bathroom"),
        ("sleep_boxes",    GRAY,   "Sleep"),
        ("active_boxes",   ORANGE, "Active"),
        ("inert_boxes",    GREEN,  "Inert"),
    ]
    for key, color, label in _BEHAVIOR_LAYERS:
        for box in cached.get(key, []):
            pt1 = _draw_rect(frame, box, color)
            cv2.putText(
                frame, label,
                (pt1[0] + 2, pt1[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )

    # 상단 HUD
    if show_info:
        state_str  = cached.get("state_str", "")
        stream_id  = cached.get("stream_id")
        frame_count = cached.get("frame_count", 0)
        num_objects = cached.get("num_objects", 0)

        prefix = f"Stream {stream_id} | " if stream_id is not None else ""
        line1 = f"{prefix}State: {state_str or 'Normal'}"
        cv2.putText(frame, line1, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)

        line2 = f"Objects: {num_objects} | Frame: {frame_count}"
        fps_info = cached.get("fps_info")
        if fps_info:
            line2 += (
                f" | AnalysisFPS: {fps_info.get('current_fps', '?'):.1f}"
                f" ({fps_info.get('mode', '?')})"
            )
        cv2.putText(frame, line2, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)

    return frame
