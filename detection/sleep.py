import time

import numpy as np


def detect_sleep(
    sleep_coor,
    sleep_bbox,
    sleep_threshold_px_sec,
    sleep_seconds,
    aspect_ratio_threshold=1.2,
    area_stability_threshold=0.15,
):
    """수면 상태를 감지합니다.

    FPS-independent: uses timestamps stored in the deques to compute a true
    time window and real px/sec average speed.

    판단 기준:
    1. 평균 이동 속도(px/sec)가 sleep_threshold_px_sec 이하 (거의 움직이지 않음)
    2. bbox의 장축/단축 비율(aspect ratio)이 aspect_ratio_threshold 이상 (누운 자세)
    3. bbox 면적 변화율(coefficient of variation)이 area_stability_threshold 이하

    Args:
        sleep_coor: dict {track_id: deque((timestamp, x_center, y_center))}
        sleep_bbox: dict {track_id: deque((timestamp, width, height))}
        sleep_threshold_px_sec: max average speed (px/sec) to count as sleeping.
            Convert from old total-displacement threshold:
            sqrt(sleep_threshold / sleep_frames) * target_fps
        sleep_seconds: time window in seconds.
            Convert from frames: sleep_frames / target_fps.
        aspect_ratio_threshold: bbox long/short side ratio threshold.
        area_stability_threshold: bbox area coefficient-of-variation threshold.

    Returns:
        sleep_id: list — track_ids judged to be sleeping.
    """
    sleep_id = []
    now = time.time()

    for id in sleep_coor.keys():
        raw_coor = list(sleep_coor[id])
        raw_bbox = list(sleep_bbox.get(id, []))

        if not raw_coor or not raw_bbox:
            continue

        # Filter to time window
        coor_entries = [(t, x, y) for t, x, y in raw_coor if now - t <= sleep_seconds]
        bbox_entries = [(t, w, h) for t, w, h in raw_bbox if now - t <= sleep_seconds]

        if len(coor_entries) < 2 or len(bbox_entries) < 1:
            continue

        # Require at least 50% time coverage
        time_span = coor_entries[-1][0] - coor_entries[0][0]
        if time_span < sleep_seconds * 0.5:
            continue

        # 1. Average speed check (px/sec)
        pts = np.array([(x, y) for _, x, y in coor_entries])
        times = np.array([t for t, _, _ in coor_entries])
        distances = np.sqrt(np.sum((pts[1:] - pts[:-1]) ** 2, axis=1))
        dts = np.maximum(np.diff(times), 1e-6)
        avg_speed = float(np.mean(distances / dts))

        if avg_speed >= sleep_threshold_px_sec:
            continue

        # 2. bbox aspect ratio check (누운 자세)
        widths = np.array([w for _, w, _ in bbox_entries])
        heights = np.array([h for _, _, h in bbox_entries])
        valid = (heights > 0) & (widths > 0)
        if not np.any(valid):
            continue

        long_side = np.maximum(widths[valid], heights[valid])
        short_side = np.minimum(widths[valid], heights[valid])
        mean_aspect_ratio = float(np.mean(long_side / short_side))

        if mean_aspect_ratio < aspect_ratio_threshold:
            continue

        # 3. bbox area stability check
        areas = widths[valid] * heights[valid]
        mean_area = float(np.mean(areas))
        if mean_area == 0:
            continue

        area_cv = float(np.std(areas) / mean_area)

        if area_cv <= area_stability_threshold:
            sleep_id.append(id)

    return sleep_id
