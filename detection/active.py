import time

import numpy as np


def detect_active(
    active_coor,
    active_threshold,
    active_seconds,
    speed_threshold_px_sec: float = 150.0,
    max_speed_px_sec: float = 4500.0,
):
    """Detect actively moving (running/playing) pets.

    FPS-independent: uses timestamps stored in ``active_coor`` to compute a
    true time window and real px/sec speeds.  A ``max_speed_px_sec`` cap
    filters out sudden tracker ID-swap glitches.

    Args:
        active_coor: dict mapping track_id -> deque of (timestamp, x, y).
        active_threshold: *ignored* (kept for backward-compat signature).
        active_seconds: time window in seconds.
            Convert from frames: active_frames / target_fps.
        speed_threshold_px_sec: minimum average speed (px/sec) to be
            considered active.  Convert from px/frame: threshold * target_fps.
        max_speed_px_sec: per-step speed cap (px/sec) — steps exceeding this
            are excluded (likely tracking glitches).
            Convert from px/frame: max_speed * target_fps.

    Returns:
        List of track_ids that are currently active.
    """
    active_id = []
    now = time.time()

    for id in active_coor.keys():
        raw = list(active_coor[id])
        if not raw:
            continue

        # Filter to time window
        entries = [(t, x, y) for t, x, y in raw if now - t <= active_seconds]

        if len(entries) < 2:
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

        if avg_speed > speed_threshold_px_sec:
            active_id.append(id)

    return active_id
