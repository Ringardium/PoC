import numpy as np


def detect_sleep(
    sleep_coor,
    sleep_bbox,
    sleep_threshold,
    sleep_frames,
    aspect_ratio_threshold=1.2,
    area_stability_threshold=0.15,
):
    """
    수면 상태를 감지합니다.

    판단 기준:
    1. 이동량(displacement)이 sleep_threshold 이하 (거의 움직이지 않음)
    2. bbox의 장축/단축 비율(aspect ratio)이 aspect_ratio_threshold 이상 (누운 자세, 방향 무관)
    3. bbox 면적 변화율(coefficient of variation)이 area_stability_threshold 이하 (자세가 안정적)

    Args:
        sleep_coor: dict {track_id: deque([(x_center, y_center), ...])}
            각 ID별 중심점 좌표 이력
        sleep_bbox: dict {track_id: deque([(width, height), ...])}
            각 ID별 bbox 크기 이력
        sleep_threshold: float
            이동량 임계값 (이 값 이하면 정지 상태로 판단)
        sleep_frames: int
            최소 필요 프레임 수 (이 프레임 이상 유지되어야 수면 판단)
        aspect_ratio_threshold: float
            bbox 장축/단축 비율 임계값 (이 값 이상이면 누운 자세로 판단, 방향 무관)
        area_stability_threshold: float
            bbox 면적 변동계수(CV) 임계값 (이 값 이하면 면적이 안정적)
            예: 0.15 = 면적 변화가 평균 대비 15% 이내

    Returns:
        sleep_id: list - 수면 상태로 판단된 track_id 리스트
    """
    sleep_id = []

    for id in sleep_coor.keys():
        coor = np.array(sleep_coor[id])
        bbox = np.array(sleep_bbox[id])

        if len(coor) < sleep_frames:
            continue

        # 1. 이동량 체크 (detect_inert와 동일 방식)
        distances = np.sum((coor[1:] - coor[:-1]) ** 2, axis=1)
        displacement = np.sum(distances)

        if displacement >= sleep_threshold:
            continue

        # 2. bbox 장축/단축 비율 체크 (누운 자세 판단)
        #    max(w,h)/min(w,h) 사용 — 카메라 방향에 무관하게 길쭉한 자세 감지
        widths = bbox[:, 0]
        heights = bbox[:, 1]
        valid = (heights > 0) & (widths > 0)
        if not np.any(valid):
            continue

        long_side = np.maximum(widths[valid], heights[valid])
        short_side = np.minimum(widths[valid], heights[valid])
        aspect_ratios = long_side / short_side
        mean_aspect_ratio = np.mean(aspect_ratios)

        if mean_aspect_ratio < aspect_ratio_threshold:
            continue

        # 3. bbox 면적 안정성 체크 (수면 중에는 크기 변화가 적음)
        areas = widths[valid] * heights[valid]
        mean_area = np.mean(areas)
        if mean_area == 0:
            continue

        area_cv = np.std(areas) / mean_area  # 변동계수 (CV)

        if area_cv <= area_stability_threshold:
            sleep_id.append(id)

    return sleep_id
