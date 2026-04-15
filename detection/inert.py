import time

import numpy as np


def detect_inert(
    inert_coor,
    inert_threshold_px_sec,
    inert_seconds,
    max_speed_px_sec: float = 4500.0,
):
    """Detect stationary (inactive) pets.

    FPS-independent: uses timestamps stored in ``inert_coor`` to compute a
    true time window and real px/sec speeds.  Frames with unrealistically
    large jumps (> ``max_speed_px_sec``) are excluded to avoid tracker
    ID-swap glitches masking truly inert pets.

    Args:
        inert_coor: dict mapping track_id -> deque of (timestamp, x, y).
        inert_threshold_px_sec: maximum average speed (px/sec) to be
            considered inert.  Convert from px/frame: threshold * target_fps.
        inert_seconds: time window in seconds.
            Convert from frames: inert_frames / target_fps.
        max_speed_px_sec: per-step speed cap (px/sec) — steps exceeding this
            are excluded (likely tracking glitches).
            Convert from px/frame: max_speed * target_fps.

    Returns:
        List of track_ids that are currently inert.
    """
    inert_id = []
    now = time.time()

    for id in inert_coor.keys():
        raw = list(inert_coor[id])
        if not raw:
            continue

        # Filter to time window
        entries = [(t, x, y) for t, x, y in raw if now - t <= inert_seconds]

        if len(entries) < 2:
            continue

        # Require at least 50% time coverage to avoid false positives
        # from short bursts of data
        time_span = entries[-1][0] - entries[0][0]
        if time_span < inert_seconds * 0.5:
            continue

        pts = np.array([(x, y) for _, x, y in entries])
        times = np.array([t for t, _, _ in entries])

        distances = np.sqrt(np.sum((pts[1:] - pts[:-1]) ** 2, axis=1))
        dts = np.diff(times)

        # Compute per-step speed (px/sec) and filter glitches
        safe_dts = np.maximum(dts, 1e-6)
        step_speeds = distances / safe_dts
        valid = step_speeds < max_speed_px_sec

        if not np.any(valid):
            continue

        avg_speed = float(np.mean(step_speeds[valid]))

        if avg_speed < inert_threshold_px_sec:
            inert_id.append(id)

    return inert_id
