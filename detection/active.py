import numpy as np


def detect_active(
    active_coor,
    active_threshold,
    active_frames,
    speed_threshold=3.0,
):
    """Detect actively moving (running/playing) pets.

    Flags pets whose cumulative displacement over the last ``active_frames``
    exceeds ``active_threshold`` AND whose average speed exceeds
    ``speed_threshold`` (pixels per frame).

    Args:
        active_coor: dict mapping track_id -> deque of [x, y] centres.
        active_threshold: minimum cumulative displacement (px) to be considered active.
        active_frames: minimum number of coordinate history frames required.
        speed_threshold: minimum average speed (px/frame) to be considered active.

    Returns:
        List of track_ids that are currently active.
    """
    active_id = []

    for id in active_coor.keys():
        coor = np.array(active_coor[id])

        if len(coor) < active_frames:
            continue

        distances = np.sqrt(np.sum((coor[1:] - coor[:-1]) ** 2, axis=1))
        displacement = np.sum(distances)
        avg_speed = np.mean(distances)

        if displacement > active_threshold and avg_speed > speed_threshold:
            active_id.append(id)

    return active_id
