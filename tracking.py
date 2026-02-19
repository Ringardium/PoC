import av
import cv2

from deep_sort.detection import Detection


def track_with_bytetrack(model, frame, bbox_color=(0, 255, 0), bbox_thickness=2,
                         font_scale=0.5, font_color=(255, 255, 255), font_thickness=1,
                         show_id=True, show_conf=False, label_format="ID: {id}"):
    """
    Track objects using ByteTrack algorithm with customizable visualization.

    Args:
        model: YOLO model instance
        frame: Input frame (numpy array)
        bbox_color: Bounding box color in BGR format (default: green)
        bbox_thickness: Bounding box line thickness (default: 2)
        font_scale: Font size scale (default: 0.5)
        font_color: Font color in BGR format (default: white)
        font_thickness: Font line thickness (default: 1)
        show_id: Whether to show track ID (default: True)
        show_conf: Whether to show confidence score (default: False)
        label_format: Label format string. Available placeholders: {id}, {conf}
                     (default: "ID: {id}")

    Returns:
        boxes: List of bounding boxes in xywh format
        track_ids: List of track IDs
        frame: Annotated frame
    """
    results = model.track(
        frame,
        persist=True,
        conf=0.7,
        iou=0.5,
        classes=[1],
        tracker="bytetrack.yaml",
    )

    ids = results[0].boxes.id

    if ids is None:
        boxes = []
        track_ids = []
    else:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()

        # [주석처리] bbox는 main.py에서 supervision으로 그림
        # for box, track_id, conf in zip(boxes, track_ids, confidences):
        #     x_center, y_center, width, height = box.tolist()
        #
        #     # Calculate corner coordinates
        #     x1 = int(x_center - width / 2)
        #     y1 = int(y_center - height / 2)
        #     x2 = int(x_center + width / 2)
        #     y2 = int(y_center + height / 2)
        #
        #     # Draw bounding box
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, bbox_thickness)
        #
        #     # Prepare label text
        #     if show_id or show_conf:
        #         label = label_format.format(id=track_id, conf=f"{conf:.2f}")
        #
        #         # Calculate text size for background
        #         (text_width, text_height), baseline = cv2.getTextSize(
        #             label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        #         )
        #
        #         # Draw background rectangle for text
        #         cv2.rectangle(
        #             frame,
        #             (x1, y1 - text_height - baseline - 5),
        #             (x1 + text_width, y1),
        #             bbox_color,
        #             -1  # Filled rectangle
        #         )
        #
        #         # Draw text
        #         cv2.putText(
        #             frame,
        #             label,
        #             (x1, y1 - baseline - 2),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             font_scale,
        #             font_color,
        #             font_thickness,
        #             cv2.LINE_AA,
        #         )
        pass

    return boxes, track_ids, frame


def track_with_botsort(model, frame, bbox_color=(0, 255, 0), bbox_thickness=2,
                       font_scale=0.5, font_color=(255, 255, 255), font_thickness=1,
                       show_id=True, show_conf=False, label_format="ID: {id}"):
    """
    Track objects using BoT-SORT algorithm with customizable visualization.

    Args:
        model: YOLO model instance
        frame: Input frame (numpy array)
        bbox_color: Bounding box color in BGR format (default: green)
        bbox_thickness: Bounding box line thickness (default: 2)
        font_scale: Font size scale (default: 0.5)
        font_color: Font color in BGR format (default: white)
        font_thickness: Font line thickness (default: 1)
        show_id: Whether to show track ID (default: True)
        show_conf: Whether to show confidence score (default: False)
        label_format: Label format string. Available placeholders: {id}, {conf}
                     (default: "ID: {id}")

    Returns:
        boxes: List of bounding boxes in xywh format
        track_ids: List of track IDs
        frame: Annotated frame
    """
    results = model.track(
        frame, persist=True, conf=0.7, iou=0.5, classes=[1], tracker="botsort.yaml"
    )

    ids = results[0].boxes.id

    if ids is None:
        boxes = []
        track_ids = []
    else:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()

        # [주석처리] bbox는 main.py에서 supervision으로 그림
        # for box, track_id, conf in zip(boxes, track_ids, confidences):
        #     x_center, y_center, width, height = box.tolist()
        #
        #     # Calculate corner coordinates
        #     x1 = int(x_center - width / 2)
        #     y1 = int(y_center - height / 2)
        #     x2 = int(x_center + width / 2)
        #     y2 = int(y_center + height / 2)
        #
        #     # Draw bounding box
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, bbox_thickness)
        #
        #     # Prepare label text
        #     if show_id or show_conf:
        #         label = label_format.format(id=track_id, conf=f"{conf:.2f}")
        #
        #         # Calculate text size for background
        #         (text_width, text_height), baseline = cv2.getTextSize(
        #             label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        #         )
        #
        #         # Draw background rectangle for text
        #         cv2.rectangle(
        #             frame,
        #             (x1, y1 - text_height - baseline - 5),
        #             (x1 + text_width, y1),
        #             bbox_color,
        #             -1  # Filled rectangle
        #         )
        #
        #         # Draw text
        #         cv2.putText(
        #             frame,
        #             label,
        #             (x1, y1 - baseline - 2),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             font_scale,
        #             font_color,
        #             font_thickness,
        #             cv2.LINE_AA,
        #         )
        pass

    return boxes, track_ids, frame


def track_with_deepsort(model, tracker, frame):
    results = model(frame, conf=0.7, iou=0.5, classes=[1])

    boxes = results[0].boxes

    detections = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy.cpu().squeeze(0).numpy().tolist()
        detections.append(
            Detection(
                [x1, y1, x2 - x1, y2 - y1], box.conf, box.data.cpu().squeeze(0).numpy()
            )
        )

    tracker.predict()
    tracker.update(detections)

    boxes = []
    track_ids = []
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        x, y, w, h = track.to_tlwh().tolist()
        x_center = x + w / 2
        y_center = y + h / 2

        boxes.append((x_center, y_center, w, h))
        track_ids.append(track.track_id)

    # [주석처리] bbox는 main.py에서 supervision으로 그림
    # for box, track_id in zip(boxes, track_ids):
    #     x_center, y_center, width, height = box
    #
    #     red = (0, 0, 255)
    #     cv2.rectangle(
    #         frame,
    #         (int(x_center - width / 2), int(y_center - height / 2)),
    #         (int(x_center + width / 2), int(y_center + height / 2)),
    #         red,
    #         2,
    #     )
    #
    #     text_color = (0, 0, 0)
    #     cv2.putText(
    #         frame,
    #         f"id:{track_id}",
    #         (int(x_center - width / 2) + 10, int(y_center - height / 2) - 5),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.5,
    #         text_color,
    #         1,
    #         cv2.LINE_AA,
    #     )

    return boxes, track_ids, frame
