import cv2
import numpy as np
import torch


def compute_iou_matrix_vectorized(boxes):
    boxes = np.array(boxes)  # shape (n, 4)
    x1 = boxes[:, 0][:, None]
    y1 = boxes[:, 1][:, None]
    x2 = boxes[:, 2][:, None]
    y2 = boxes[:, 3][:, None]

    area = (x2 - x1) * (y2 - y1)  # shape (n, 1)

    # Broadcast boxes for pairwise comparisons
    x1_inter = np.maximum(x1, x1.T)
    y1_inter = np.maximum(y1, y1.T)
    x2_inter = np.minimum(x2, x2.T)
    y2_inter = np.minimum(y2, y2.T)

    inter_w = np.clip(x2_inter - x1_inter, 0, None)
    inter_h = np.clip(y2_inter - y1_inter, 0, None)
    inter_area = inter_w * inter_h

    union_area = area + area.T - inter_area
    iou = inter_area / np.clip(union_area, 1e-6, None)  # avoid division by zero

    return iou


def triu_where(condition, array):
    mask = torch.triu(torch.ones_like(array), diagonal=1).bool()
    indices = condition & mask
    true_indices = indices.nonzero(as_tuple=False)
    return true_indices


def detect_fight(
    x_centers,
    y_centers,
    track_ids,
    close_count,
    far_count,
    threshold,
    reset_frames,
    flag_frames,
    width,
    height,
):
    def diff(c2, c1):
        return np.tile(c2[None, :], (len(c2), 1)) - np.tile(c1[:, None], (1, len(c1)))

    bbox_coor = np.stack(
        (
            x_centers - width / 2,
            y_centers - height / 2,
            x_centers + width / 2,
            y_centers + height / 2,
        ),
        axis=1,
    )

    iou = torch.from_numpy(compute_iou_matrix_vectorized(bbox_coor))

    tracked_ids = []
    for i, j in triu_where(iou > threshold, iou):
        tracked_ids.append((track_ids[i], track_ids[j]))

    far_count += 1
    for i, j in tracked_ids:
        far_count[i, j] = 0
        far_count[j, i] = 0

    for i, j in tracked_ids:
        close_count[i, j] += 1
        close_count[j, i] += 1

    close_count = close_count * (far_count < reset_frames)

    return triu_where(close_count > flag_frames, close_count)


def detect_inert(
    inert_coor,
    inert_threshold,
    inert_frames,
):
    inert_id = []

    for i, id in enumerate(inert_coor.keys()):
        coor = np.array(inert_coor[id])
        distances = np.sum((coor[1:] - coor[:-1]) ** 2, axis=1)
        displacement = np.sum(distances)

        if len(coor) < inert_frames:
            continue

        if displacement < inert_threshold:
            inert_id.append(id)

    return inert_id


def polygon_selector(temp_img):
    points = []

    clone = temp_img.copy()
    win_name = "Draw Polygon ROI (Enter: Done, ESC: Reset)"

    # 사용자 마우스로부터 x, y 좌표 입력 받음 누르는 수만큼
    def click_polygon(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(temp_img, (x, y), 3, (0, 0, 255), -1)
            if len(points) > 1:
                cv2.line(temp_img, points[-2], points[-1], (255, 0, 0), 2)

    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, click_polygon)

    while True:
        cv2.imshow(win_name, temp_img)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # Enter, 입력 좌표 확정
            if len(points) >= 3:
                cv2.polylines(
                    temp_img,
                    [np.array(points)],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2,
                )
                break
        elif key == 27:  # ESC, 입력 좌표 초기화
            points = []
            temp_img = clone.copy()

    cv2.destroyWindow(win_name)
    return np.array(points, dtype=np.int32)


# polygon 내에 객체 있는지 확인하는 함수
def is_point_in_polygon(point, polygon):
    """
    point: (x, y) 튜플
    polygon: numpy array of shape (N, 2)
    """
    return cv2.pointPolygonTest(polygon, point, measureDist=False) >= 0


def draw_puttext(img, content, point):
    cv2.putText(img, content, point, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


def detect_escape(boxes, track_ids, frame, frame_cnt, polygon, w, h):
    red = (0, 0, 255)

    area = frame.copy()
    cv2.polylines(area, [polygon], isClosed=True, color=red, thickness=2)
    cv2.fillPoly(area, [polygon], color=red)

    i_p1 = (0, 0)
    i_p2 = w, h
    inner_area = [i_p1, i_p2]

    escaped_ids = []
    # obj center point show-up & escaping detection
    for box, id in zip(boxes, track_ids):
        cx, cy, _, _ = box  # .cpu().int().numpy().tolist()

        if not is_point_in_polygon((cx, cy), polygon):
            escaped_ids.append(id)

    alpha = 0.2

    img_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    #  Polygon 영역 alpha 블랜딩
    for c in range(3):  # B, G, R 채널만
        img_bgra[: inner_area[1][1], inner_area[0][0] : inner_area[1][0], c] = (
            alpha
            * area[
                inner_area[0][1] : inner_area[1][1],
                inner_area[0][0] : inner_area[1][0],
                c,
            ]
            + (1 - alpha)
            * img_bgra[
                inner_area[0][1] : inner_area[1][1],
                inner_area[0][0] : inner_area[1][0],
                c,
            ]
        ).astype(np.uint8)

    return img_bgra, escaped_ids
