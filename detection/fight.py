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
    speeds=None,
    fight_speed_threshold=5.0,
):
    """Detect fighting between pets.

    Args:
        speeds: dict mapping track_id -> average speed (pixels/frame).
            When provided, close_count only increments if BOTH dogs in the
            pair exceed ``fight_speed_threshold``.  This filters out calm
            proximity (resting together, playing gently).
        fight_speed_threshold: minimum average speed for both dogs to count
            as a fighting interaction.
    """

    bbox_coor = np.stack((x_centers - width / 2, y_centers - height / 2, x_centers + width / 2, y_centers + height / 2), axis=1)

    iou = torch.from_numpy(compute_iou_matrix_vectorized(bbox_coor))

    tracked_ids = []
    for i, j in triu_where(iou > threshold, iou):
        id_i, id_j = track_ids[i], track_ids[j]

        # If speeds provided, only count when both dogs are moving fast
        if speeds is not None:
            speed_i = speeds.get(id_i, 0.0)
            speed_j = speeds.get(id_j, 0.0)
            if speed_i < fight_speed_threshold or speed_j < fight_speed_threshold:
                continue

        tracked_ids.append((id_i, id_j))

    far_count += 1
    for i, j in tracked_ids:
        far_count[i, j] = 0
        far_count[j, i] = 0

    for i, j in tracked_ids:
        close_count[i, j] += 1
        close_count[j, i] += 1

    close_count = close_count * (far_count < reset_frames)

    return triu_where(close_count > flag_frames, close_count)
