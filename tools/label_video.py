"""Generate a labeled video with custom names, behaviors, and time ranges.

Usage:
    # Step 1: Preview — see track IDs
    python tools/label_video.py --input video.mp4 --output preview.mp4

    # Step 2: Create config JSON with names + behaviors
    # Step 3: Generate labeled video
    python tools/label_video.py --input video.mp4 --output labeled.mp4 --config label_config.json

    # With ReID (reduces ID switching)
    python tools/label_video.py --input video.mp4 --output labeled.mp4 --config label_config.json --reid

Config JSON format:
    {
        "names": {"1": "뽀삐", "3": "초코", "5": "루루"},
        "events": [
            {"id": 1, "behavior": "fight",  "start": "00:10", "end": "00:25"},
            {"id": 1, "behavior": "eat",    "start": "01:30", "end": "02:00"},
            {"id": 3, "behavior": "sleep",  "start": "00:30", "end": "03:00"},
            {"id": 5, "behavior": "active", "start": "00:00", "end": "01:00"}
        ],
        "behavior_colors": {
            "fight":    [255, 0, 0],
            "sleep":    [0, 0, 255],
            "eat":      [0, 200, 100],
            "active":   [255, 165, 0],
            "inert":    [128, 128, 128],
            "bathroom": [160, 82, 45],
            "escape":   [255, 0, 255]
        },
        "model": "weights/modelv11x.pt",
        "tracker": "botsort",
        "conf": 0.25,
        "iou": 0.5,
        "classes": [1],
        "imgsz": 640,
        "show_unnamed": true,
        "default_color": [0, 255, 0],
        "font_scale": 0.7,
        "thickness": 2,
        "use_reid": false,
        "reid_method": "adaptive",
        "reid_threshold": 0.5
    }

Time format: "MM:SS" or "HH:MM:SS"
"""

import json
import sys

import click
import cv2
import numpy as np
from ultralytics import YOLO

try:
    from reid import ReIDTracker
    REID_AVAILABLE = True
except ImportError:
    REID_AVAILABLE = False


BEHAVIOR_COLORS = {
    "fight":    (0, 0, 255),
    "sleep":    (255, 0, 0),
    "eat":      (100, 200, 0),
    "active":   (0, 165, 255),
    "inert":    (128, 128, 128),
    "bathroom": (45, 82, 160),
    "escape":   (255, 0, 255),
}

DEFAULT_CONFIG = {
    "names": {},
    "events": [],
    "behavior_colors": BEHAVIOR_COLORS,
    "model": "weights/modelv11x.pt",
    "tracker": "botsort",
    "conf": 0.25,
    "iou": 0.5,
    "classes": [1],
    "imgsz": 640,
    "show_unnamed": True,
    "default_color": [0, 255, 0],
    "font_scale": 0.7,
    "thickness": 2,
    "use_reid": False,
    "reid_method": "adaptive",
    "reid_threshold": 0.5,
}


def parse_time(t: str) -> float:
    """Parse 'MM:SS' or 'HH:MM:SS' to seconds."""
    parts = t.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(t)


def build_event_index(events: list) -> dict:
    """Pre-process events into {track_id: [(start_sec, end_sec, behavior), ...]}."""
    index = {}
    for ev in events:
        tid = int(ev["id"])
        start = parse_time(str(ev["start"]))
        end = parse_time(str(ev["end"]))
        behavior = ev["behavior"]
        index.setdefault(tid, []).append((start, end, behavior))
    # Sort by start time
    for tid in index:
        index[tid].sort(key=lambda x: x[0])
    return index


def get_behavior_at(event_index: dict, tid: int, sec: float) -> str | None:
    """Return behavior name if tid has an event at given time, else None."""
    if tid not in event_index:
        return None
    for start, end, behavior in event_index[tid]:
        if start <= sec <= end:
            return behavior
        if start > sec:
            break
    return None


def draw_label(frame, x1, y1, x2, y2, label, color, font_scale, thickness):
    """Draw bbox and label on frame."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)


def draw_timeline(frame, sec, total_sec, events, behavior_colors, w, h):
    """Draw a small timeline bar at the bottom showing behavior events."""
    bar_h = 16
    bar_y = h - bar_h - 4
    # Background
    cv2.rectangle(frame, (10, bar_y), (w - 10, bar_y + bar_h), (40, 40, 40), -1)
    bar_w = w - 20

    if total_sec <= 0:
        return

    # Draw event spans
    for ev in events:
        start = parse_time(str(ev["start"]))
        end = parse_time(str(ev["end"]))
        behavior = ev["behavior"]
        color = tuple(behavior_colors.get(behavior, [128, 128, 128]))
        sx = int(10 + (start / total_sec) * bar_w)
        ex = int(10 + (end / total_sec) * bar_w)
        cv2.rectangle(frame, (sx, bar_y + 1), (ex, bar_y + bar_h - 1), color, -1)

    # Current position marker
    cx = int(10 + (sec / total_sec) * bar_w)
    cv2.line(frame, (cx, bar_y - 2), (cx, bar_y + bar_h + 2), (255, 255, 255), 2)

    # Timestamp text
    mm, ss = divmod(int(sec), 60)
    hh, mm = divmod(mm, 60)
    ts = f"{hh:02d}:{mm:02d}:{ss:02d}" if hh else f"{mm:02d}:{ss:02d}"
    cv2.putText(frame, ts, (cx + 4, bar_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def xyxy_to_xywh_center(xyxy):
    """Convert xyxy bboxes to (x_center, y_center, w, h) for ReID."""
    result = []
    for x1, y1, x2, y2 in xyxy:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        result.append((cx, cy, w, h))
    return result


@click.command()
@click.option("--input", "input_path", required=True, help="Input video path")
@click.option("--output", "output_path", required=True, help="Output video path")
@click.option("--config", "config_path", default=None, help="JSON config with names + behaviors")
@click.option("--model", default=None, help="YOLO model path (overrides config)")
@click.option("--reid", "use_reid", is_flag=True, default=False, help="Enable ReID for stable IDs")
def main(input_path, output_path, config_path, model, use_reid):
    """Generate labeled video with names, behavior colors, and timeline."""

    # Load config
    cfg = dict(DEFAULT_CONFIG)
    if config_path:
        with open(config_path) as f:
            user_cfg = json.load(f)
        # Merge behavior_colors — user config is RGB, convert to BGR for OpenCV
        if "behavior_colors" in user_cfg:
            merged_colors = dict(BEHAVIOR_COLORS)
            for k, v in user_cfg["behavior_colors"].items():
                r, g, b = v
                merged_colors[k] = (b, g, r)
            user_cfg["behavior_colors"] = merged_colors
        cfg.update(user_cfg)

    if model:
        cfg["model"] = model
    if use_reid:
        cfg["use_reid"] = True

    names = {int(k): v for k, v in cfg["names"].items()}
    events = cfg["events"]
    event_index = build_event_index(events)
    behavior_colors = {k: tuple(v) if isinstance(v, list) else v
                       for k, v in cfg["behavior_colors"].items()}
    default_color = tuple(cfg["default_color"])
    font_scale = cfg["font_scale"]
    thickness = cfg["thickness"]
    tracker_yaml = f"{cfg['tracker']}.yaml"

    # Load model
    click.echo(f"Model: {cfg['model']}")
    click.echo(f"Names: {names if names else '(none — preview mode)'}")
    click.echo(f"Events: {len(events)}")
    yolo = YOLO(cfg["model"])

    # Initialize ReID tracker
    reid_tracker = None
    if cfg["use_reid"]:
        if not REID_AVAILABLE:
            click.echo("[WARN] reid module not available, running without ReID")
        else:
            reid_tracker = ReIDTracker(
                reid_method=cfg["reid_method"],
                similarity_threshold=cfg["reid_threshold"],
                correction_enabled=True,
                global_id_enabled=False,
            )
            click.echo(f"ReID: {cfg['reid_method']} (threshold={cfg['reid_threshold']})")

    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        click.echo(f"[ERROR] Cannot open: {input_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = total / fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    click.echo(f"Input: {input_path} ({w}x{h}, {fps:.1f}fps, {total} frames, {total_sec:.1f}s)")
    click.echo(f"Output: {output_path}")
    click.echo("Processing...")

    reid_corrections = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        sec = frame_idx / fps

        # Track
        results = yolo.track(
            frame, persist=True,
            conf=cfg["conf"], iou=cfg["iou"],
            classes=cfg["classes"], tracker=tracker_yaml,
            imgsz=cfg["imgsz"], verbose=False,
        )

        boxes_data = results[0].boxes
        if boxes_data.id is not None:
            xyxy = boxes_data.xyxy.cpu().numpy().astype(int)
            ids = boxes_data.id.int().cpu().tolist()
            confs = boxes_data.conf.cpu().tolist()

            # Apply ReID correction
            if reid_tracker is not None and len(ids) > 0:
                xywh_center = xyxy_to_xywh_center(xyxy)
                reid_result = reid_tracker.process(frame, xywh_center, ids)
                corrected = reid_result.get("corrected_ids", ids)
                corrections = reid_result.get("corrections", [])
                if corrections:
                    reid_corrections += len(corrections)
                ids = corrected

            for i, tid in enumerate(ids):
                x1, y1, x2, y2 = xyxy[i]
                conf = confs[i]

                # Check behavior at current time
                behavior = get_behavior_at(event_index, tid, sec)
                pet_name = names.get(tid)

                if behavior:
                    color = behavior_colors.get(behavior, default_color)
                    if pet_name:
                        label = f"{pet_name} - {behavior}"
                    else:
                        label = f"ID:{tid} - {behavior}"
                elif pet_name:
                    color = default_color
                    label = pet_name
                elif cfg["show_unnamed"]:
                    color = default_color
                    label = f"ID:{tid}"
                else:
                    # No name, no behavior — bbox only, no label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), default_color, thickness)
                    continue

                draw_label(frame, x1, y1, x2, y2, label, color, font_scale, thickness)

        # Draw timeline bar
        if events:
            draw_timeline(frame, sec, total_sec, events, behavior_colors, w, h)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            pct = frame_idx / total * 100 if total else 0
            click.echo(f"  {frame_idx}/{total} ({pct:.0f}%)")

    cap.release()
    out.release()
    click.echo(f"\nDone! {frame_idx} frames → {output_path}")
    if reid_tracker is not None:
        click.echo(f"ReID corrections: {reid_corrections}")


if __name__ == "__main__":
    main()
