"""
Privacy Filter for Videos
Detects humans and applies blur or mosaic to protect privacy
"""

import click
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm


def apply_blur(frame, x1, y1, x2, y2, blur_strength=51):
    """
    Apply Gaussian blur to a region

    Args:
        frame: Input frame
        x1, y1, x2, y2: Bounding box coordinates
        blur_strength: Blur kernel size (must be odd number)
    """
    # Ensure blur_strength is odd
    blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1

    # Extract the region
    roi = frame[y1:y2, x1:x2]

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)

    # Replace the region
    frame[y1:y2, x1:x2] = blurred

    return frame


def apply_mosaic(frame, x1, y1, x2, y2, mosaic_size=20):
    """
    Apply mosaic (pixelation) to a region

    Args:
        frame: Input frame
        x1, y1, x2, y2: Bounding box coordinates
        mosaic_size: Size of mosaic blocks
    """
    # Extract the region
    roi = frame[y1:y2, x1:x2]

    h, w = roi.shape[:2]

    if h == 0 or w == 0:
        return frame

    # Resize down
    temp = cv2.resize(roi, (max(1, w // mosaic_size), max(1, h // mosaic_size)),
                      interpolation=cv2.INTER_LINEAR)

    # Resize back up (creates pixelated effect)
    mosaic = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

    # Replace the region
    frame[y1:y2, x1:x2] = mosaic

    return frame


def apply_black_box(frame, x1, y1, x2, y2):
    """
    Apply solid black box to a region

    Args:
        frame: Input frame
        x1, y1, x2, y2: Bounding box coordinates
    """
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    return frame


@click.command()
@click.option("--model", default="yolo11n.pt", help="YOLO model path (default uses pretrained)")
@click.option("--input", required=True, help="Input video path")
@click.option("--output", required=True, help="Output video path")
@click.option("--method", type=click.Choice(["blur", "mosaic", "black"]),
              default="blur", help="Privacy filter method")
@click.option("--blur-strength", default=51, help="Blur strength (kernel size, must be odd)")
@click.option("--mosaic-size", default=20, help="Mosaic block size (pixels)")
@click.option("--conf", default=0.5, help="Confidence threshold for person detection")
@click.option("--iou", default=0.5, help="IoU threshold for NMS")
@click.option("--padding", default=10, help="Padding around detected bbox (pixels)")
@click.option("--show-bbox", is_flag=True, help="Show bounding boxes around detected persons")
@click.option("--device", default="0", help="Device to run on (0, 1, cpu)")
def main(
    model,
    input,
    output,
    method,
    blur_strength,
    mosaic_size,
    conf,
    iou,
    padding,
    show_bbox,
    device,
):
    """
    Privacy filter for videos - detects and anonymizes humans

    Detects persons (YOLO class 0) and applies privacy filters

    Examples:

    # Apply blur to persons
    python privacy_filter.py --input video.mp4 --output output.mp4 --method blur

    # Apply mosaic with custom size
    python privacy_filter.py --input video.mp4 --output output.mp4 --method mosaic --mosaic-size 30

    # Apply strong blur
    python privacy_filter.py --input video.mp4 --output output.mp4 --method blur --blur-strength 99

    # Black boxes
    python privacy_filter.py --input video.mp4 --output output.mp4 --method black

    # Show bounding boxes
    python privacy_filter.py --input video.mp4 --output output.mp4 --show-bbox
    """

    click.echo("="*60)
    click.echo("Privacy Filter for Videos")
    click.echo("="*60)
    click.echo(f"Input: {input}")
    click.echo(f"Output: {output}")
    click.echo(f"Method: {method}")
    click.echo(f"Model: {model}")
    click.echo(f"Confidence threshold: {conf}")

    if method == "blur":
        click.echo(f"Blur strength: {blur_strength}")
    elif method == "mosaic":
        click.echo(f"Mosaic size: {mosaic_size}")

    click.echo("="*60)

    # Load YOLO model
    click.echo("\n[INFO] Loading YOLO model...")
    yolo_model = YOLO(model)

    # Open video
    click.echo(f"[INFO] Opening video: {input}")
    cap = cv2.VideoCapture(input)

    if not cap.isOpened():
        click.echo(f"[ERROR] Could not open video: {input}", err=True)
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    click.echo(f"[INFO] Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))

    if not out.isOpened():
        click.echo(f"[ERROR] Could not create output video: {output}", err=True)
        cap.release()
        return

    click.echo(f"[INFO] Processing video...")
    click.echo(f"[INFO] Detecting persons (class 0) and applying {method} filter\n")

    # Process video
    frame_count = 0
    person_detections = 0

    with tqdm(total=total_frames, desc="Processing", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # Detect persons (class 0 in COCO dataset)
            results = yolo_model(frame, conf=conf, iou=iou, classes=[0], device=device, verbose=False)

            # Process detections
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                for box in boxes:
                    x1, y1, x2, y2 = box

                    # Add padding
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(width, x2 + padding)
                    y2 = min(height, y2 + padding)

                    # Apply privacy filter
                    if method == "blur":
                        frame = apply_blur(frame, x1, y1, x2, y2, blur_strength)
                    elif method == "mosaic":
                        frame = apply_mosaic(frame, x1, y1, x2, y2, mosaic_size)
                    elif method == "black":
                        frame = apply_black_box(frame, x1, y1, x2, y2)

                    # Optionally show bounding box
                    if show_bbox:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            "Person",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

                    person_detections += 1

            # Write frame
            out.write(frame)
            pbar.update(1)

    # Cleanup
    cap.release()
    out.release()

    click.echo("\n" + "="*60)
    click.echo("Processing completed!")
    click.echo(f"Processed frames: {frame_count}")
    click.echo(f"Person detections: {person_detections}")
    click.echo(f"Output saved to: {output}")
    click.echo("="*60)


if __name__ == "__main__":
    main()
