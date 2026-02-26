"""Crop detections and capture frames from video.

Commands:
    crop     - Crop YOLO-detected objects from video
    capture  - Capture raw frames from video at intervals

Usage:
    # Crop all detections
    python tools/crop_detections.py crop --input video.mp4 --output crops/

    # Crop with tracking, best per ID
    python tools/crop_detections.py crop --input video.mp4 --output crops/ --track --best

    # Capture frames every 1 second
    python tools/crop_detections.py capture --input video.mp4 --output frames/

    # Capture every 30th frame, resized to 640px
    python tools/crop_detections.py capture --input video.mp4 --output frames/ --interval 30 --resize 640

    # Capture specific time range (seconds)
    python tools/crop_detections.py capture --input video.mp4 --output frames/ --start 10 --end 60
"""

import sys
from pathlib import Path

import click
import cv2


@click.group()
def cli():
    """Video frame capture & detection crop tool."""
    pass


# ── Crop command ───────────────────────────────────────────────────────


@cli.command()
@click.option("--input", "input_path", required=True, help="Input video path")
@click.option("--output", "output_dir", required=True, help="Output directory for crops")
@click.option("--model", default="weights/modelv11x.pt", help="YOLO model path")
@click.option("--conf", default=0.25, type=float, help="Confidence threshold")
@click.option("--iou", default=0.5, type=float, help="IoU threshold")
@click.option("--classes", default="1", help="Detection classes, comma-separated (e.g. '1' or '1,3')")
@click.option("--imgsz", default=640, type=int, help="Inference image size")
@click.option("--tracker", default="botsort", help="Tracker: botsort, bytetrack")
@click.option("--track", is_flag=True, default=False, help="Enable tracking, group crops by ID")
@click.option("--best", is_flag=True, default=False, help="Save only the best (highest conf) crop per track ID")
@click.option("--interval", default=1, type=int, help="Process every N-th frame (1=all)")
@click.option("--margin", default=10, type=int, help="Padding pixels around crop")
@click.option("--max-crops", default=0, type=int, help="Max total crops to save (0=unlimited)")
def crop(input_path, output_dir, model, conf, iou, classes, imgsz, tracker,
         track, best, interval, margin, max_crops):
    """Crop detected objects from video and save as images.

    \b
    Output structure:
      Without --track:  crops/frame_000100_det0_0.85.jpg
      With --track:     crops/id_001/frame_000100_0.85.jpg
      With --best:      crops/id_001_best_0.91.jpg
    """
    from ultralytics import YOLO

    cls_list = [int(c.strip()) for c in classes.split(",")]
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    click.echo(f"Model: {model}")
    click.echo(f"Classes: {cls_list}, conf={conf}, track={track}, best={best}, interval={interval}")
    yolo = YOLO(model)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        click.echo(f"[ERROR] Cannot open: {input_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    click.echo(f"Input: {input_path} ({w}x{h}, {fps:.1f}fps, {total} frames)")
    click.echo(f"Output: {output_dir}")

    best_crops = {}
    saved_count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval != 0:
            frame_idx += 1
            continue

        if track:
            results = yolo.track(
                frame, persist=True,
                conf=conf, iou=iou, classes=cls_list,
                tracker=f"{tracker}.yaml", imgsz=imgsz, verbose=False,
            )
        else:
            results = yolo.predict(
                frame, conf=conf, iou=iou, classes=cls_list,
                imgsz=imgsz, verbose=False,
            )

        boxes_data = results[0].boxes
        if boxes_data is None or len(boxes_data) == 0:
            frame_idx += 1
            continue

        xyxy = boxes_data.xyxy.cpu().numpy().astype(int)
        confs = boxes_data.conf.cpu().tolist()
        ids = None
        if track and boxes_data.id is not None:
            ids = boxes_data.id.int().cpu().tolist()

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            c = confs[i]

            x1m = max(0, x1 - margin)
            y1m = max(0, y1 - margin)
            x2m = min(w, x2 + margin)
            y2m = min(h, y2 + margin)
            crop_img = frame[y1m:y2m, x1m:x2m]

            if crop_img.size == 0:
                continue

            if track and ids is not None:
                tid = ids[i]

                if best:
                    if tid not in best_crops or c > best_crops[tid][0]:
                        best_crops[tid] = (c, crop_img.copy())
                else:
                    id_dir = out_path / f"id_{tid:03d}"
                    id_dir.mkdir(exist_ok=True)
                    fname = f"frame_{frame_idx:06d}_{c:.2f}.jpg"
                    cv2.imwrite(str(id_dir / fname), crop_img)
                    saved_count += 1
            else:
                fname = f"frame_{frame_idx:06d}_det{i}_{c:.2f}.jpg"
                cv2.imwrite(str(out_path / fname), crop_img)
                saved_count += 1

            if max_crops > 0 and saved_count >= max_crops:
                break

        if max_crops > 0 and saved_count >= max_crops:
            click.echo(f"Reached max crops ({max_crops}), stopping.")
            break

        frame_idx += 1
        if frame_idx % 100 == 0:
            pct = frame_idx / total * 100 if total else 0
            click.echo(f"  {frame_idx}/{total} ({pct:.0f}%) — {saved_count} crops")

    cap.release()

    if best and best_crops:
        for tid, (c, crop_img) in best_crops.items():
            fname = f"id_{tid:03d}_best_{c:.2f}.jpg"
            cv2.imwrite(str(out_path / fname), crop_img)
            saved_count += 1

    click.echo(f"\nDone! {saved_count} crops saved -> {output_dir}")
    if best:
        click.echo(f"  Unique track IDs: {len(best_crops)}")


# ── Capture internals ──────────────────────────────────────────────────


def _capture_single(source: str, out_dir: Path, label: str,
                    frame_interval: int, start: float, end: float,
                    resize: int, encode_params: list, fmt: str,
                    max_frames: int, prefix: str, stop_event=None):
    """Capture frames from a single source. Returns saved count."""
    import time as _time

    is_stream = source.startswith("rtsp://") or source.startswith("http")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        click.echo(f"[{label}] ERROR: Cannot open {source}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_stream else 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Recalculate interval based on this source's fps if using seconds
    if frame_interval <= 0:
        frame_interval = max(1, int(round(fps)))

    src_type = "Stream" if is_stream else "File"
    click.echo(f"[{label}] {src_type}: {source} ({w}x{h}, {fps:.1f}fps)")

    start_frame = int(start * fps) if (start > 0 and not is_stream) else 0
    end_frame = int(end * fps) if (end > 0 and not is_stream) else 0

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    saved_count = 0
    frame_idx = start_frame
    consecutive_failures = 0
    max_failures = 30

    while True:
        if stop_event and stop_event.is_set():
            break

        ret, frame = cap.read()
        if not ret:
            if is_stream:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    click.echo(f"[{label}] Reconnecting...")
                    cap.release()
                    _time.sleep(2)
                    cap = cv2.VideoCapture(source)
                    if not cap.isOpened():
                        click.echo(f"[{label}] Reconnect failed, stopping.")
                        break
                    consecutive_failures = 0
                continue
            else:
                break

        consecutive_failures = 0

        if end_frame > 0 and frame_idx >= end_frame:
            break

        if (frame_idx - start_frame) % frame_interval != 0:
            frame_idx += 1
            continue

        if resize > 0:
            rh, rw = frame.shape[:2]
            if max(rh, rw) > resize:
                scale = resize / max(rh, rw)
                frame = cv2.resize(frame, (int(rw * scale), int(rh * scale)),
                                   interpolation=cv2.INTER_AREA)

        time_sec = frame_idx / fps
        minutes = int(time_sec // 60)
        secs = int(time_sec % 60)
        timestamp = f"{minutes:02d}m{secs:02d}s"

        saved_count += 1
        fname = f"{prefix}_{saved_count:06d}_{timestamp}.{fmt}"
        cv2.imwrite(str(out_dir / fname), frame, encode_params)

        if max_frames > 0 and saved_count >= max_frames:
            click.echo(f"[{label}] Reached max frames ({max_frames}).")
            break

        frame_idx += 1
        if saved_count % 50 == 0:
            if total > 0:
                pct = frame_idx / total * 100
                click.echo(f"[{label}] {saved_count} frames ({pct:.0f}%)")
            else:
                click.echo(f"[{label}] {saved_count} frames")

    cap.release()
    duration = (frame_idx - start_frame) / fps
    click.echo(f"[{label}] Done! {saved_count} frames ({duration:.1f}s)")
    return saved_count


# ── Capture command ────────────────────────────────────────────────────


@cli.command()
@click.option("--input", "input_paths", required=True, multiple=True,
              help="Input video/RTSP URL (repeat for multiple sources)")
@click.option("--output", "output_dir", required=True, help="Output directory for frames")
@click.option("--interval", default=0, type=int, help="Save every N-th frame (0=use --seconds)")
@click.option("--seconds", default=1.0, type=float, help="Save one frame every N seconds (default 1.0)")
@click.option("--start", default=0.0, type=float, help="Start time in seconds (file only)")
@click.option("--end", default=0.0, type=float, help="End time in seconds (0=until end, file only)")
@click.option("--resize", default=0, type=int, help="Resize long side to this px (0=original)")
@click.option("--quality", default=95, type=int, help="JPEG quality 1-100 (default 95)")
@click.option("--format", "fmt", default="jpg", type=click.Choice(["jpg", "png", "bmp"]), help="Image format")
@click.option("--max-frames", default=0, type=int, help="Max frames per source (0=unlimited)")
@click.option("--prefix", default="frame", help="Filename prefix (default: frame)")
def capture(input_paths, output_dir, interval, seconds, start, end,
            resize, quality, fmt, max_frames, prefix):
    """Capture raw frames from one or more video/stream sources.

    \b
    Multiple sources run in parallel threads, each saving to its own subfolder.

    \b
    Examples:
      # Single file
      python tools/crop_detections.py capture --input video.mp4 --output frames/

      # Multiple RTSP streams
      python tools/crop_detections.py capture \\
          --input "rtsp://admin:pass@cam1:554/stream" \\
          --input "rtsp://admin:pass@cam2:554/stream" \\
          --output frames/ --seconds 2 --max-frames 100

      # Multiple files
      python tools/crop_detections.py capture \\
          --input video1.mp4 --input video2.mp4 \\
          --output frames/ --interval 30 --resize 640

    \b
    Output (single source):
      frames/frame_000001_00m05s.jpg

    Output (multiple sources):
      frames/source_0/frame_000001_00m05s.jpg
      frames/source_1/frame_000001_00m03s.jpg
    """
    import threading

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Frame interval (will be recalculated per-source if using seconds)
    if interval > 0:
        frame_interval = interval
    else:
        frame_interval = 0  # signal to use fps * seconds per source

    # Encode params
    encode_params = []
    if fmt == "jpg":
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif fmt == "png":
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, max(0, min(9, (100 - quality) // 10))]

    # Determine interval from seconds if not set by --interval
    effective_interval = interval if interval > 0 else max(1, int(round(30 * seconds)))

    n_sources = len(input_paths)
    click.echo(f"Sources: {n_sources}")
    click.echo(f"Output: {output_dir} (format={fmt}, quality={quality})")
    click.echo(f"Interval: {'every ' + str(interval) + ' frames' if interval > 0 else f'every {seconds}s'}")
    if resize > 0:
        click.echo(f"Resize: long side -> {resize}px")
    if max_frames > 0:
        click.echo(f"Max frames per source: {max_frames}")

    multi = n_sources > 1
    if multi:
        click.echo("Press Ctrl+C to stop all streams.")

    stop_event = threading.Event()
    results = [0] * n_sources

    def _worker(idx, source):
        if multi:
            label = f"src_{idx}"
            out_dir = out_root / label
        else:
            label = "capture"
            out_dir = out_root
        out_dir.mkdir(parents=True, exist_ok=True)

        results[idx] = _capture_single(
            source=source,
            out_dir=out_dir,
            label=label,
            frame_interval=effective_interval,
            start=start,
            end=end,
            resize=resize,
            encode_params=encode_params,
            fmt=fmt,
            max_frames=max_frames,
            prefix=prefix,
            stop_event=stop_event,
        )

    if n_sources == 1:
        try:
            _worker(0, input_paths[0])
        except KeyboardInterrupt:
            click.echo("\nStopped by user.")
    else:
        threads = []
        for i, src in enumerate(input_paths):
            t = threading.Thread(target=_worker, args=(i, src), daemon=True)
            threads.append(t)
            t.start()

        try:
            # Wait for all threads (check every 0.5s for KeyboardInterrupt)
            while any(t.is_alive() for t in threads):
                for t in threads:
                    t.join(timeout=0.5)
        except KeyboardInterrupt:
            click.echo("\nStopping all streams...")
            stop_event.set()
            for t in threads:
                t.join(timeout=5)

    total_saved = sum(results)
    click.echo(f"\nTotal: {total_saved} frames saved -> {output_dir}")
    if multi:
        for i, count in enumerate(results):
            click.echo(f"  src_{i}: {count} frames")


if __name__ == "__main__":
    cli()
