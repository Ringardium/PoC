import numpy as np
import torch


def triu_where(condition, array):
    mask = torch.triu(torch.ones_like(array), diagonal=1).bool()
    indices = condition & mask
    true_indices = indices.nonzero(as_tuple=False)
    return true_indices


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
