import numpy as np


def compute_bowl_overlap(pet_xyxy, bowl_xyxy):
    """
    bowl 기준 겹침 비율을 계산합니다. (intersection / bowl_area)

    Args:
        pet_xyxy: (x1, y1, x2, y2) 강아지 bbox
        bowl_xyxy: (x1, y1, x2, y2) 밥그릇 bbox

    Returns:
        float: bowl 면적 대비 겹침 비율 (0.0 ~ 1.0)
    """
    x1_inter = max(pet_xyxy[0], bowl_xyxy[0])
    y1_inter = max(pet_xyxy[1], bowl_xyxy[1])
    x2_inter = min(pet_xyxy[2], bowl_xyxy[2])
    y2_inter = min(pet_xyxy[3], bowl_xyxy[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    bowl_area = (bowl_xyxy[2] - bowl_xyxy[0]) * (bowl_xyxy[3] - bowl_xyxy[1])

    return inter_area / max(bowl_area, 1e-6)


def check_direction_toward_bowl(coor_history, bowl_center, direction_frames):
    """
    최근 N프레임 동안 강아지의 이동 방향이 bowl을 향했는지 확인합니다.
    이동 벡터와 bowl 방향 벡터의 코사인 유사도로 판단합니다.

    Args:
        coor_history: deque of [x_center, y_center] 좌표 이력
        bowl_center: (bx, by) bowl 중심점
        direction_frames: 이동 방향 판단에 사용할 프레임 수

    Returns:
        bool: bowl 방향으로 이동했으면 True
    """
    coor = np.array(coor_history)
    if len(coor) < direction_frames:
        return False

    # 최근 direction_frames 구간의 이동 벡터 (시작점 → 끝점)
    recent = coor[-direction_frames:]
    movement_vec = recent[-1] - recent[0]

    # 현재 위치 → bowl 방향 벡터
    bowl_vec = np.array(bowl_center) - recent[-1]

    # 벡터 크기
    move_norm = np.linalg.norm(movement_vec)
    bowl_norm = np.linalg.norm(bowl_vec)

    if move_norm < 1e-6 or bowl_norm < 1e-6:
        # 이동이 거의 없거나 이미 bowl 위에 있음 → True
        return bowl_norm < 1e-6

    # 코사인 유사도 (> 0 이면 bowl 방향으로 이동)
    cosine_sim = np.dot(movement_vec, bowl_vec) / (move_norm * bowl_norm)

    return cosine_sim > 0


def detect_eat(
    eat_coor,
    eat_near_count,
    pet_boxes,
    track_ids,
    bowl_boxes,
    iou_threshold=0.3,
    dwell_frames=30,
    direction_frames=10,
):
    """
    식사 행동을 감지합니다.

    판단 기준 (3가지 모두 충족):
    1. 이동 방향이 bowl을 향했음 (코사인 유사도 > 0)
    2. bowl 근처에서 dwell_frames 이상 체류
    3. bowl 기준 겹침 비율이 iou_threshold 이상

    Args:
        eat_coor: dict {track_id: deque([(x, y), ...])}
            각 ID별 중심점 좌표 이력
        eat_near_count: dict {track_id: int}
            각 ID별 bowl 근처 연속 체류 프레임 수
        pet_boxes: list of (x_center, y_center, width, height)
            현재 프레임의 강아지 bbox (xywh)
        track_ids: list of int
            현재 프레임의 track ID
        bowl_boxes: numpy array of shape (N, 4) in xyxy format
            현재 프레임의 bowl bbox
        iou_threshold: float
            bowl 기준 겹침 비율 임계값
        dwell_frames: int
            bowl 근처 최소 체류 프레임 수
        direction_frames: int
            이동 방향 판단에 사용할 프레임 수

    Returns:
        eat_id: list - 식사 중으로 판단된 track_id 리스트
    """
    eat_id = []

    if len(bowl_boxes) == 0:
        # bowl이 없으면 모든 체류 카운트 리셋
        for tid in eat_near_count:
            eat_near_count[tid] = 0
        return eat_id

    for tid, box in zip(track_ids, pet_boxes):
        if hasattr(box, 'tolist'):
            xc, yc, w, h = box.tolist()
        else:
            xc, yc, w, h = box

        # pet bbox를 xyxy로 변환
        pet_xyxy = (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)

        # 초기화
        if tid not in eat_near_count:
            eat_near_count[tid] = 0

        # 모든 bowl에 대해 체크
        is_eating = False
        for bowl in bowl_boxes:
            bowl_xyxy = bowl[:4]

            # 기준 1: bowl 기준 겹침 비율
            overlap = compute_bowl_overlap(pet_xyxy, bowl_xyxy)
            if overlap < iou_threshold:
                continue

            # 기준 2: bowl 방향으로 이동했었는지
            bowl_cx = (bowl_xyxy[0] + bowl_xyxy[2]) / 2
            bowl_cy = (bowl_xyxy[1] + bowl_xyxy[3]) / 2

            if tid in eat_coor and len(eat_coor[tid]) >= direction_frames:
                moved_toward = check_direction_toward_bowl(
                    eat_coor[tid], (bowl_cx, bowl_cy), direction_frames
                )
            else:
                # 이력 부족 → 겹침이 충분하면 방향은 통과
                moved_toward = overlap >= iou_threshold

            if not moved_toward:
                continue

            # 기준 3: 체류 시간
            eat_near_count[tid] += 1
            if eat_near_count[tid] >= dwell_frames:
                is_eating = True
            break  # 하나의 bowl이라도 조건 부분 충족하면 카운트 유지

        else:
            # 어떤 bowl과도 겹치지 않으면 카운트 리셋
            eat_near_count[tid] = 0

        if is_eating:
            eat_id.append(tid)

    return eat_id
