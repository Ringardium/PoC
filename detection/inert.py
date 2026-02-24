import numpy as np


def detect_inert(
    inert_coor,
    inert_threshold,
    inert_frames,
    max_speed=150.0,
):
    """Detect stationary (inactive) pets.

    Uses **average speed** (px/frame) so the result is independent of the
    coordinate history length.  Frames with unrealistically large jumps
    (> ``max_speed``) are excluded from the average to avoid tracker-ID-swap
    glitches masking truly inert pets.

    Args:
        inert_coor: dict mapping track_id -> deque of [x, y] centres.
        inert_threshold: maximum average speed (px/frame) to be considered inert.
        inert_frames: minimum number of coordinate history frames required.
        max_speed: per-frame distance cap — frames exceeding this are excluded.

    Returns:
        List of track_ids that are currently inert.
    """
    inert_id = []

    for i, id in enumerate(inert_coor.keys()):
        coor = np.array(inert_coor[id])

        if len(coor) < inert_frames:
            continue

        distances = np.sqrt(np.sum((coor[1:] - coor[:-1]) ** 2, axis=1))

        # Filter out tracking glitch frames
        valid = distances[distances < max_speed]
        if len(valid) == 0:
            continue

        avg_speed = float(np.mean(valid))

        if avg_speed < inert_threshold:
            inert_id.append(id)

    return inert_id
