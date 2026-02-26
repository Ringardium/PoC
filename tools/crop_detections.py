"""Crop and save YOLO-detected objects from a video.

Usage:
    # Crop all detections (one image per detection per frame)
    python tools/crop_detections.py --input video.mp4 --output crops/

    # Crop every 10th frame only (faster)
    python tools/crop_detections.py --input video.mp4 --output crops/ --interval 10

    # With tracking — group by track ID
    python tools/crop_detections.py --input video.mp4 --output crops/ --track

    # Best crop per track ID (highest confidence)
    python tools/crop_detections.py --input video.mp4 --output crops/ --track --best

    # Custom model / classes
    python tools/crop_detections.py --input video.mp4 --output crops/ --model weights/best.pt --classes 1 3

Output structure:
    # Without --track:
    crops/frame_000100_det0_0.85.jpg
    crops/frame_000100_det1_0.72.jpg

    # With --track:
    crops/id_001/frame_000100_0.85.jpg
    crops/id_003/frame_000250_0.91.jpg

    # With --track --best:
    crops/id_001_best.jpg
    crops/id_003_best.jpg
"""

import sys
from pathlib import Path

import click
import cv2
from ultralytics import YOLO


@click.command()
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
def main(input_path, output_dir, model, conf, iou, classes, imgsz, tracker,
         track, best, interval, margin, max_crops):
    """Crop detected objects from video and save as images."""

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

    # For --best mode: track_id -> (conf, crop_image)
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

            # Apply margin with bounds check
            x1m = max(0, x1 - margin)
            y1m = max(0, y1 - margin)
            x2m = min(w, x2 + margin)
            y2m = min(h, y2 + margin)
            crop = frame[y1m:y2m, x1m:x2m]

            if crop.size == 0:
                continue

            if track and ids is not None:
                tid = ids[i]

                if best:
                    # Keep only highest confidence crop per ID
                    if tid not in best_crops or c > best_crops[tid][0]:
                        best_crops[tid] = (c, crop.copy())
                else:
                    # Save each crop grouped by ID folder
                    id_dir = out_path / f"id_{tid:03d}"
                    id_dir.mkdir(exist_ok=True)
                    fname = f"frame_{frame_idx:06d}_{c:.2f}.jpg"
                    cv2.imwrite(str(id_dir / fname), crop)
                    saved_count += 1
            else:
                # No tracking — flat output
                fname = f"frame_{frame_idx:06d}_det{i}_{c:.2f}.jpg"
                cv2.imwrite(str(out_path / fname), crop)
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

    # Save best crops
    if best and best_crops:
        for tid, (c, crop) in best_crops.items():
            fname = f"id_{tid:03d}_best_{c:.2f}.jpg"
            cv2.imwrite(str(out_path / fname), crop)
            saved_count += 1

    click.echo(f"\nDone! {saved_count} crops saved → {output_dir}")
    if best:
        click.echo(f"  Unique track IDs: {len(best_crops)}")


if __name__ == "__main__":
    main()
