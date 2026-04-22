import numpy as np
from ultralytics import YOLO


# 분류 모델 캐시
_cls_model = None


def _load_cls_model(model_path):
    """YOLO classification 모델을 로드합니다 (최초 1회)."""
    global _cls_model
    if _cls_model is None:
        _cls_model = YOLO(model_path)
        print(f"[INFO] Bathroom classifier 로드: {model_path}")
    return _cls_model


def check_trigger(
    bbox_history,
    coor_history,
    trigger_frames,
    area_drop_ratio=0.25,
    displacement_threshold=30.0,
):
    """
    배변 트리거 조건을 확인합니다.

    조건 1: bbox 투영 면적이 최근 평균 대비 일정 비율 이상 감소 (웅크림).
            각도 불변에 더 가까운 시그널 — 측면/전면 어디서든 웅크리면
            (w*h) 투영이 줄어듦. (기존 height 기반은 각도 의존적이라 교체됨)
    조건 2: 이동량이 적음 (정지 상태)

    Args:
        bbox_history: deque of (width, height) - bbox 크기 이력
        coor_history: deque of (x, y) - 중심 좌표 이력
        trigger_frames: 트리거 판단에 필요한 최소 프레임 수
        area_drop_ratio: 면적 감소 비율 임계값 (0.25 = 25% 이상 감소)
        displacement_threshold: 이동량 임계값

    Returns:
        bool: 트리거 발동 여부
    """
    bbox = np.array(bbox_history)
    coor = np.array(coor_history)

    if len(bbox) < trigger_frames:
        return False

    areas = bbox[:, 0] * bbox[:, 1]

    # 전반부 평균 면적 vs 후반부 평균 면적
    half = len(areas) // 2
    early_area = np.mean(areas[:half])
    recent_area = np.mean(areas[half:])

    if early_area < 1e-6:
        return False

    # 조건 1: 면적 감소율 (angle-invariant)
    drop_ratio = (early_area - recent_area) / early_area
    if drop_ratio < area_drop_ratio:
        return False

    # 조건 2: 후반부 이동량 (정지 상태)
    recent_coor = coor[half:]
    if len(recent_coor) < 2:
        return False

    distances = np.sum((recent_coor[1:] - recent_coor[:-1]) ** 2, axis=1)
    displacement = np.sum(distances)

    return displacement < displacement_threshold


def crop_pet_region(frame, box, padding=20):
    """
    프레임에서 강아지 영역을 crop합니다.

    Args:
        frame: BGR 이미지
        box: (x_center, y_center, width, height) xywh 형식
        padding: crop 영역 여백 (px)

    Returns:
        cropped: crop된 이미지 (numpy array), 실패 시 None
    """
    if hasattr(box, 'tolist'):
        xc, yc, w, h = box.tolist()
    else:
        xc, yc, w, h = box

    h_frame, w_frame = frame.shape[:2]

    x1 = max(0, int(xc - w / 2) - padding)
    y1 = max(0, int(yc - h / 2) - padding)
    x2 = min(w_frame, int(xc + w / 2) + padding)
    y2 = min(h_frame, int(yc + h / 2) + padding)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


def detect_bathroom(
    bathroom_coor,
    bathroom_bbox,
    pet_boxes,
    track_ids,
    frame,
    cls_model_path,
    trigger_frames=30,
    area_drop_ratio=0.25,
    displacement_threshold=30.0,
    cls_confidence=0.5,
    cls_class_name="bathroom",
):
    """
    배변 행동을 감지합니다.

    Phase 1 (트리거): bbox 투영 면적 감소(웅크림) + 정지 상태 (angle-invariant)
    Phase 2 (분류): crop된 이미지를 YOLO classify 모델로 판별

    Args:
        bathroom_coor: dict {track_id: deque([(x, y), ...])}
        bathroom_bbox: dict {track_id: deque([(w, h), ...])}
        pet_boxes: list of (x_center, y_center, width, height)
        track_ids: list of int
        frame: 현재 프레임 (BGR)
        cls_model_path: YOLO classification 모델 경로
        trigger_frames: 트리거 판단 최소 프레임 수
        area_drop_ratio: 면적 감소 비율 임계값
        displacement_threshold: 이동량 임계값
        cls_confidence: 분류 모델 confidence 임계값
        cls_class_name: 배변 클래스 이름

    Returns:
        bathroom_id: list - 배변 중으로 판단된 track_id 리스트
    """
    bathroom_id = []

    for tid, box in zip(track_ids, pet_boxes):
        if tid not in bathroom_coor or tid not in bathroom_bbox:
            continue

        # Phase 1: 트리거 확인
        triggered = check_trigger(
            bathroom_bbox[tid],
            bathroom_coor[tid],
            trigger_frames,
            area_drop_ratio,
            displacement_threshold,
        )

        if not triggered:
            continue

        # Phase 2: YOLO classify 모델로 분류
        cropped = crop_pet_region(frame, box)
        if cropped is None:
            continue

        cls_model = _load_cls_model(cls_model_path)
        results = cls_model.predict(cropped, verbose=False)

        if len(results) == 0:
            continue

        # 분류 결과 확인
        probs = results[0].probs
        if probs is None:
            continue

        names = results[0].names
        top1_idx = probs.top1
        top1_conf = probs.top1conf.item()
        top1_name = names[top1_idx]

        if top1_name == cls_class_name and top1_conf >= cls_confidence:
            bathroom_id.append(tid)

    return bathroom_id
