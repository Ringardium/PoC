import numpy as np


def detect_active(
    active_coor,
    active_threshold,
    active_frames,
    speed_threshold=5.0,
    max_speed=150.0,
):
    """Detect actively moving (running/playing) pets.

    Uses **average speed** (px/frame) as the primary metric so the result is
    independent of the coordinate history length.  A ``max_speed`` cap filters
    out sudden tracker-ID-swap glitches that produce unrealistically large
    jumps.

    Args:
        active_coor: dict mapping track_id -> deque of [x, y] centres.
        active_threshold: *ignored* (kept for backward-compat signature).
        active_frames: minimum number of coordinate history frames required.
        speed_threshold: minimum average speed (px/frame) to be considered active.
        max_speed: per-frame distance cap — frames exceeding this are excluded
            from the average (likely tracking glitches).

    Returns:
        List of track_ids that are currently active.
    """
    active_id = []

    for id in active_coor.keys():
        coor = np.array(active_coor[id])

        if len(coor) < active_frames:
            continue

        distances = np.sqrt(np.sum((coor[1:] - coor[:-1]) ** 2, axis=1))

        # Filter out tracking glitch frames
        valid = distances[distances < max_speed]
        if len(valid) == 0:
            continue

        avg_speed = float(np.mean(valid))

        if avg_speed > speed_threshold:
            active_id.append(id)

    return active_id
