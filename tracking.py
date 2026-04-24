import av
import cv2
import torch
import numpy as np

from deep_sort.detection import Detection


def save_video(f, frames, fps):
    """AV 라이브러리를 사용해 H264 코덱으로 비디오 저장"""
    container = av.open(f, mode="w")
    stream = container.add_stream("h264", rate=fps)
    first_frame = frames[0]
    stream.height = first_frame.shape[0]
    stream.width = first_frame.shape[1]
    stream.pix_fmt = "yuv420p"
    for img in frames:
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return f


def track_with_bytetrack(model, frame, device=None, conf=0.7, iou=0.5,
                         bbox_color=(0, 255, 0), bbox_thickness=2,
                         font_scale=0.5, font_color=(255, 255, 255), font_thickness=1,
                         show_id=True, show_conf=False, label_format="ID: {id}"):
    """
    Track objects using ByteTrack algorithm with customizable visualization.

    Args:
        model: YOLO model instance
        frame: Input frame (numpy array)
        device: torch device string (e.g. "cuda:0"). None = model default.
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
    kwargs = dict(persist=True, conf=conf, iou=iou, classes=[1], tracker="bytetrack.yaml")
    if device is not None:
        kwargs["device"] = device
    results = model.track(frame, **kwargs)

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


def track_with_botsort(model, frame, device=None, conf=0.7, iou=0.5,
                       bbox_color=(0, 255, 0), bbox_thickness=2,
                       font_scale=0.5, font_color=(255, 255, 255), font_thickness=1,
                       show_id=True, show_conf=False, label_format="ID: {id}"):
    """
    Track objects using BoT-SORT algorithm with customizable visualization.

    Args:
        model: YOLO model instance
        frame: Input frame (numpy array)
        device: torch device string (e.g. "cuda:0"). None = model default.
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
    kwargs = dict(persist=True, conf=conf, iou=iou, classes=[1], tracker="botsort.yaml")
    if device is not None:
        kwargs["device"] = device
    results = model.track(frame, **kwargs)

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


def track_with_deepsort(model, tracker, frame, conf=0.7, iou=0.5):
    results = model(frame, conf=conf, iou=iou, classes=[1])

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


# ---------------------------------------------------------------------------
# OCSort tracker (fast-deep-oc-sort)
# ---------------------------------------------------------------------------

def initialize_ocsort_tracker(config: dict = None):
    """
    OCSort 트래커 초기화

    Args:
        config: 설정 딕셔너리. 미제공 시 기본값 사용.
            - det_thresh, iou_threshold, asso_func, delta_t, inertia,
              w_association_emb, alpha_fixed_emb, embedding_off, cmc_off,
              aw_off, aw_param, new_kf_off, grid_off,
              occlusion_threshold, aspect_ratio_threshold, angle_threshold

    Returns:
        OCSort tracker instance

    Raises:
        ImportError: fast_deep_oc_sort 패키지가 없을 때
    """
    try:
        from fast_deep_oc_sort.trackers.ocsort_tracker.ocsort import OCSort
    except ImportError as e:
        raise ImportError(
            "OCSort를 사용하려면 fast_deep_oc_sort 패키지가 필요합니다: "
            "pip install fast-deep-oc-sort"
        ) from e

    defaults = {
        "det_thresh": 0.6,
        "iou_threshold": 0.3,
        "asso_func": "iou",
        "delta_t": 3,
        "inertia": 0.2,
        "w_association_emb": 0.75,
        "alpha_fixed_emb": 0.95,
        "embedding_off": False,
        "cmc_off": False,
        "aw_off": False,
        "aw_param": 0.5,
        "new_kf_off": False,
        "grid_off": False,
        "occlusion_threshold": 0.2,
        "aspect_ratio_threshold": 0.65,
        "angle_threshold": 45.0,
    }
    if config:
        defaults.update(config)

    return OCSort(
        det_thresh=defaults["det_thresh"],
        iou_threshold=defaults["iou_threshold"],
        asso_func=defaults["asso_func"],
        delta_t=defaults["delta_t"],
        inertia=defaults["inertia"],
        w_association_emb=defaults["w_association_emb"],
        alpha_fixed_emb=defaults["alpha_fixed_emb"],
        embedding_off=defaults["embedding_off"],
        cmc_off=defaults["cmc_off"],
        aw_off=defaults["aw_off"],
        aw_param=defaults["aw_param"],
        new_kf_off=defaults["new_kf_off"],
        grid_off=defaults["grid_off"],
        occlusion_threshold=defaults["occlusion_threshold"],
        aspect_ratio_threshold=defaults["aspect_ratio_threshold"],
        angle_threshold=defaults["angle_threshold"],
    )


def track_with_ocsort(model, tracker_instance, frame, device=None,
                      filter_config: dict = None, draw: bool = True):
    """
    OCSort 알고리즘으로 객체 추적

    Args:
        model: YOLO 모델 인스턴스
        tracker_instance: initialize_ocsort_tracker()로 생성된 tracker
        frame: 입력 프레임 (numpy array)
        device: torch device string. None = 모델 기본값
        filter_config: 필터 파라미터 딕셔너리
            - aspect_ratio_thresh: bbox 종횡비 필터 (기본 1.6)
            - min_box_area: 최소 bbox 면적 (기본 10)
        draw: True면 추적 결과를 frame에 그림

    Returns:
        boxes: [(x_center, y_center, w, h), ...] xywh 형식
        track_ids: 트래킹 ID 리스트
        frame: (draw=True면) 주석이 추가된 프레임
    """
    try:
        from fast_deep_oc_sort.utils import filter_targets
        _has_filter_util = True
    except ImportError:
        _has_filter_util = False

    kwargs = dict(conf=0.7, iou=0.5, classes=[1], verbose=False)
    if device is not None:
        kwargs["device"] = device

    results = model(frame, **kwargs)

    detections_list = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy.cpu().squeeze().numpy()
            conf = box.conf.item()
            detections_list.append([x1, y1, x2 - x1, y2 - y1, conf])

    if not detections_list:
        return [], [], frame

    det_tensor = torch.tensor(detections_list, dtype=torch.float32)
    if torch.cuda.is_available():
        det_tensor = det_tensor.cuda()

    h, w = frame.shape[:2]
    targets = tracker_instance.update(det_tensor, (h, w), frame)

    boxes = []
    track_ids = []
    annotated = frame.copy() if draw else frame

    if targets is not None and len(targets) > 0:
        if filter_config and _has_filter_util:
            tlwhs, ids = filter_targets(
                targets,
                filter_config.get("aspect_ratio_thresh", 1.6),
                filter_config.get("min_box_area", 10),
            )
        else:
            tlwhs = targets[:, :4].copy()
            tlwhs[:, 2] = tlwhs[:, 2] - tlwhs[:, 0]  # x2 → w
            tlwhs[:, 3] = tlwhs[:, 3] - tlwhs[:, 1]  # y2 → h
            ids = targets[:, 4].astype(int)

        for tlwh, obj_id in zip(tlwhs, ids):
            x1, y1, bw, bh = tlwh
            cx, cy = x1 + bw / 2, y1 + bh / 2
            boxes.append((cx, cy, bw, bh))
            track_ids.append(int(obj_id))

            if draw:
                cv2.rectangle(annotated, (int(x1), int(y1)),
                              (int(x1 + bw), int(y1 + bh)), (0, 255, 0), 2)
                cv2.putText(
                    annotated, f"id:{int(obj_id)}",
                    (int(x1) + 10, int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
                )

    return boxes, track_ids, annotated
