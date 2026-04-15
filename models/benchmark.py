"""
YOLO vs TFLite 벤치마크

mAP 비교: ultralytics val 파이프라인 사용
속도 비교: 원본 YOLO(PyTorch) vs TFLite 인터프리터 직접 실행
"""

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def benchmark_map(
    pt_path: str,
    tflite_path: str,
    data_yaml: str,
    imgsz: int = 640,
    conf: float = 0.001,
    iou: float = 0.6,
) -> dict:
    """
    원본 YOLO(.pt) vs TFLite 모델 mAP 비교.

    ultralytics의 val 파이프라인을 그대로 사용하므로
    NMS / 전처리가 동일 조건으로 적용된다.

    Args:
        pt_path:     원본 YOLO .pt 경로
        tflite_path: 변환된 .tflite 경로
        data_yaml:   val set이 정의된 data.yaml 경로
        imgsz:       추론 해상도
        conf:        confidence threshold (mAP 계산 시 낮게 설정)
        iou:         NMS IoU threshold
    """
    from ultralytics import YOLO

    print("=" * 55)
    print("mAP 벤치마크")
    print("=" * 55)

    # --- 원본 YOLO ---
    print(f"\n[1/2] 원본 YOLO: {pt_path}")
    pt_model = YOLO(pt_path)
    pt_results = pt_model.val(
        data=data_yaml,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False,
    )
    pt_map50    = float(pt_results.box.map50)
    pt_map5095  = float(pt_results.box.map)
    pt_per_class = {
        name: float(ap)
        for name, ap in zip(pt_results.names.values(), pt_results.box.ap50)
    }
    print(f"  mAP50     : {pt_map50:.4f}")
    print(f"  mAP50-95  : {pt_map5095:.4f}")

    # --- TFLite ---
    print(f"\n[2/2] TFLite: {tflite_path}")
    tflite_map50, tflite_map5095, tflite_per_class = _tflite_map(
        tflite_path=tflite_path,
        data_yaml=data_yaml,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
    )

    print(f"  mAP50     : {tflite_map50:.4f}")
    print(f"  mAP50-95  : {tflite_map5095:.4f}")

    # --- 비교 ---
    drop50    = pt_map50 - tflite_map50
    drop5095  = pt_map5095 - tflite_map5095

    print("\n" + "=" * 55)
    print("결과 비교")
    print("=" * 55)
    print(f"{'':20s} {'YOLO':>10s} {'TFLite':>10s} {'차이':>10s}")
    print(f"{'mAP50':20s} {pt_map50:>10.4f} {tflite_map50:>10.4f} {-drop50:>+10.4f}")
    print(f"{'mAP50-95':20s} {pt_map5095:>10.4f} {tflite_map5095:>10.4f} {-drop5095:>+10.4f}")

    # 클래스 이름 정규화 (YOLO 모델과 data.yaml 이름이 다를 수 있음, e.g. "dog'" vs "dog")
    def _normalize_name(s: str) -> str:
        return s.strip().strip("'\"")

    tflite_norm = {_normalize_name(k): v for k, v in tflite_per_class.items()}

    print("\n클래스별 mAP50:")
    print(f"{'클래스':20s} {'YOLO':>10s} {'TFLite':>10s} {'차이':>10s}")
    for cls in pt_per_class:
        p = pt_per_class.get(cls, 0.0)
        t = tflite_norm.get(_normalize_name(cls), 0.0)
        print(f"{cls:20s} {p:>10.4f} {t:>10.4f} {t-p:>+10.4f}")

    # INT8 quantization 허용 기준: mAP50 drop < 1~2%
    status = "✅ 양호" if drop50 < 0.02 else ("⚠️ 주의" if drop50 < 0.05 else "❌ 재calibration 필요")
    print(f"\nmAP50 drop: {drop50:.4f}  →  {status}")

    return {
        "pt":     {"map50": pt_map50,     "map5095": pt_map5095,     "per_class": pt_per_class},
        "tflite": {"map50": tflite_map50, "map5095": tflite_map5095, "per_class": tflite_per_class},
        "drop":   {"map50": drop50,       "map5095": drop5095},
    }


def _tflite_map(
    tflite_path: str,
    data_yaml: str,
    imgsz: int = 640,
    conf: float = 0.001,
    iou: float = 0.6,
):
    """
    TFLite 모델 mAP 직접 계산.

    두 가지 output 포맷을 자동 감지:
    - NMS 포함: (1, 300, 6) = [x1, y1, x2, y2, conf, class_id]
    - Raw output: (1, nc+4, 8400) = [x,y,w,h, cls0, cls1, ...] (NCHW)
                  또는 (1, 8400, nc+4) (NHWC — onnx2tf 변환 후)
    """
    import yaml
    import torch
    import tensorflow as tf
    from ultralytics.utils.metrics import ap_per_class

    # data.yaml 파싱
    with open(data_yaml) as f:
        data = yaml.safe_load(f)

    yaml_dir = Path(data_yaml).parent
    val_rel = data.get('val', 'train/images')

    # 경로 후보를 순서대로 시도
    candidates = [
        (yaml_dir / val_rel).resolve(),                          # yaml 기준 상대경로
        (yaml_dir / Path(val_rel).name).resolve(),               # 마지막 컴포넌트만
        Path(val_rel).resolve(),                                  # 절대경로
    ]
    val_path = next((p for p in candidates if p.exists()), candidates[0])
    label_path = Path(str(val_path).replace('images', 'labels'))
    nc = int(data.get('nc', 3))
    names = data.get('names', [str(i) for i in range(nc)])

    print(f"  val 이미지: {val_path}")
    print(f"  클래스: {names}")

    # TFLite 로드
    interpreter = tf.lite.Interpreter(model_path=tflite_path, num_threads=4)
    interpreter.allocate_tensors()
    input_detail   = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    input_dtype    = input_detail['dtype']

    # output 포맷 감지
    n_outputs = len(output_details)
    first_shape = tuple(output_details[0]['shape'])

    if n_outputs == 1 and len(first_shape) == 3 and first_shape[1] <= 500 and first_shape[2] == 6:
        output_mode = 'nms'
        print(f"  output 포맷: NMS {first_shape}")
    elif n_outputs >= 2:
        output_mode = 'split'
        for od in output_details:
            print(f"  output: {od['name']} shape={tuple(od['shape'])}")
        print(f"  output 포맷: Split (boxes + scores 분리) → NMS를 직접 수행")
    else:
        output_mode = 'raw'
        print(f"  output 포맷: Raw {first_shape} → NMS를 직접 수행")

    img_files = sorted(val_path.glob("*.jpg")) + sorted(val_path.glob("*.png")) + sorted(val_path.glob("*.jpeg"))

    # mAP 계산용 누적 버퍼
    all_tp       = []
    all_conf     = []
    all_pred_cls = []
    all_target_cls = []

    iou_thresholds = np.linspace(0.5, 0.95, 10)

    n_iou = len(iou_thresholds)

    for img_file in img_files:
        img_bgr = cv2.imread(str(img_file))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

        if input_dtype == np.uint8:
            tensor = img_resized.astype(np.uint8)[np.newaxis]
        else:
            tensor = (img_resized.astype(np.float32) / 255.0)[np.newaxis]

        interpreter.set_tensor(input_detail['index'], tensor)
        interpreter.invoke()

        if output_mode == 'nms':
            raw = interpreter.get_tensor(output_details[0]['index'])[0]  # (300, 6)
            dets = raw[raw[:, 4] >= conf]
        elif output_mode == 'split':
            # 2-output: boxes (1,4,N) + scores (1,nc,N) 분리
            outs = {}
            for od in output_details:
                s = tuple(od['shape'])
                v = interpreter.get_tensor(od['index'])[0]
                # (4, N) → boxes, (nc, N) → scores
                if len(s) == 3 and s[1] == 4:
                    outs['boxes'] = v
                else:
                    outs['scores'] = v
            # boxes와 scores를 concat해서 raw_to_dets에 전달
            combined = np.concatenate([outs['boxes'], outs['scores']], axis=0)  # (nc+4, N)
            dets = _raw_to_dets(combined, nc, imgsz, conf, iou)
        else:
            raw = interpreter.get_tensor(output_details[0]['index'])[0]
            dets = _raw_to_dets(raw, nc, imgsz, conf, iou)

        # GT 라벨 로드
        label_file = label_path / (img_file.stem + '.txt')
        gt_boxes, gt_classes = [], []
        if label_file.exists():
            for line in label_file.read_text().strip().splitlines():
                parts = line.split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                gt_boxes.append([(cx - w/2)*imgsz, (cy - h/2)*imgsz,
                                  (cx + w/2)*imgsz, (cy + h/2)*imgsz])
                gt_classes.append(cls_id)

        gt_boxes   = np.array(gt_boxes,   dtype=np.float32).reshape(-1, 4)
        gt_classes = np.array(gt_classes, dtype=np.int32)

        # 클래스별 TP 계산
        for cls_id in range(nc):
            gt_cls   = gt_boxes[gt_classes == cls_id]
            pred_cls = dets[dets[:, 5].astype(int) == cls_id] if len(dets) else np.zeros((0, 6))

            if len(pred_cls):
                pred_cls = pred_cls[np.argsort(-pred_cls[:, 4])]  # conf 내림차순

            tp = np.zeros((len(pred_cls), n_iou), dtype=bool)

            if len(pred_cls) and len(gt_cls):
                # matched: (n_gt, n_iou) — IoU threshold별 독립 매칭
                matched = np.zeros((len(gt_cls), n_iou), dtype=bool)
                for pi in range(len(pred_cls)):
                    ious = _box_iou(pred_cls[pi, :4], gt_cls)
                    for ti, iou_t in enumerate(iou_thresholds):
                        candidates = np.where((ious >= iou_t) & ~matched[:, ti])[0]
                        if len(candidates):
                            best = candidates[np.argmax(ious[candidates])]
                            tp[pi, ti] = True
                            matched[best, ti] = True

            all_tp.extend(tp.tolist())
            all_conf.extend(pred_cls[:, 4].tolist())
            all_pred_cls.extend([cls_id] * len(pred_cls))
            all_target_cls.extend([cls_id] * len(gt_cls))

    if not all_conf:
        print("  [경고] 예측 결과 없음")
        return 0.0, 0.0, {n: 0.0 for n in names}

    tp_arr   = np.array(all_tp,         dtype=bool)
    conf_arr = np.array(all_conf,       dtype=np.float32)
    pcls_arr = np.array(all_pred_cls,   dtype=np.int32)
    tcls_arr = np.array(all_target_cls, dtype=np.int32)

    results        = ap_per_class(tp_arr, conf_arr, pcls_arr, tcls_arr, names=names)
    ap             = results[5]                  # (n_unique_classes, n_iou_thresholds)
    unique_classes = results[6].astype(int)      # 실제 클래스 ID 배열

    ap50    = ap[:, 0]
    map50   = float(np.mean(ap50))
    map5095 = float(np.mean(ap))

    # unique_classes 기준으로 per-class 매핑 (클래스 누락 시 0으로 채움)
    per_class = {n: 0.0 for n in names}
    for i, cls_id in enumerate(unique_classes):
        if cls_id < len(names):
            per_class[names[cls_id]] = float(ap50[i])

    print(f"  mAP50     : {map50:.4f}")
    print(f"  mAP50-95  : {map5095:.4f}")

    return map50, map5095, per_class


def _raw_to_dets(
    raw: np.ndarray,
    nc: int,
    imgsz: int,
    conf: float,
    iou_thresh: float,
) -> np.ndarray:
    """
    YOLO raw output → NMS 적용된 detections.

    raw 포맷:
    - (nc+4, num_anchors): NCHW 원본 (ONNX 직접)
    - (num_anchors, nc+4): NHWC (onnx2tf 변환 후)

    Returns: (N, 6) = [x1, y1, x2, y2, conf, class_id]
    """
    # shape 정규화: (nc+4, num_anchors) 형태로 통일
    if raw.ndim == 2:
        if raw.shape[0] > raw.shape[1]:
            # (8400, 7) → (7, 8400)
            raw = raw.T
    elif raw.ndim == 3:
        raw = raw[0]
        if raw.shape[0] > raw.shape[1]:
            raw = raw.T

    # raw: (nc+4, num_anchors)
    # 처음 4행: cx, cy, w, h (pixel coords, imgsz 기준)
    # 나머지: class scores
    boxes_xywh = raw[:4].T          # (num_anchors, 4)
    class_scores = raw[4:].T        # (num_anchors, nc)

    # 클래스별 최대 score + class_id
    max_scores = np.max(class_scores, axis=1)       # (num_anchors,)
    class_ids = np.argmax(class_scores, axis=1)     # (num_anchors,)

    # conf threshold
    mask = max_scores >= conf
    if not np.any(mask):
        return np.zeros((0, 6), dtype=np.float32)

    boxes_xywh = boxes_xywh[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    # xywh → x1y1x2y2
    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # 클래스별 NMS
    keep_all = []
    for cls_id in range(nc):
        cls_mask = class_ids == cls_id
        if not np.any(cls_mask):
            continue
        cls_boxes = boxes_xyxy[cls_mask]
        cls_scores = max_scores[cls_mask]
        cls_indices = np.where(cls_mask)[0]

        keep = _nms(cls_boxes, cls_scores, iou_thresh)
        keep_all.extend(cls_indices[keep].tolist())

    if not keep_all:
        return np.zeros((0, 6), dtype=np.float32)

    keep_all = np.array(keep_all)
    dets = np.column_stack([
        boxes_xyxy[keep_all],
        max_scores[keep_all],
        class_ids[keep_all].astype(np.float32),
    ])
    return dets


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Greedy NMS. Returns indices to keep."""
    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = _box_iou(boxes[i], boxes[order[1:]])
        remaining = np.where(ious < iou_thresh)[0]
        order = order[remaining + 1]
    return np.array(keep)


def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """단일 box vs 복수 boxes IoU 계산."""
    xi1 = np.maximum(box[0], boxes[:, 0])
    yi1 = np.maximum(box[1], boxes[:, 1])
    xi2 = np.minimum(box[2], boxes[:, 2])
    yi2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(xi2 - xi1, 0) * np.maximum(yi2 - yi1, 0)
    area_box  = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area_box + area_boxes - inter + 1e-7)


def benchmark_speed(
    pt_path: str,
    tflite_path: str,
    imgsz: int = 640,
    num_runs: int = 100,
    warmup: int = 10,
    image_path: Optional[str] = None,
) -> dict:
    """
    원본 YOLO(.pt) vs TFLite 인터프리터 추론 속도 비교.

    TFLite는 raw 인터프리터로 직접 실행한다.
    (ultralytics 래퍼 오버헤드 없이 순수 추론 시간 측정)

    Args:
        pt_path:     원본 YOLO .pt 경로
        tflite_path: 변환된 .tflite 경로
        imgsz:       추론 해상도
        num_runs:    벤치마크 반복 횟수
        warmup:      워밍업 횟수 (측정에서 제외)
        image_path:  테스트 이미지 경로 (없으면 랜덤 텐서 사용)
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("tensorflow가 필요합니다.")
    from ultralytics import YOLO

    print("=" * 55)
    print("속도 벤치마크")
    print("=" * 55)

    # 테스트 입력 준비
    if image_path and Path(image_path).exists():
        img = cv2.imread(image_path)
        img = cv2.resize(img, (imgsz, imgsz))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pt_input = img_rgb  # YOLO는 BGR/RGB 모두 처리
        tflite_uint8 = img_rgb.astype(np.uint8)[np.newaxis]   # uint8 (NPU INT8 모델)
        tflite_fp32  = (img_rgb.astype(np.float32) / 255.0)[np.newaxis]
        print(f"  테스트 이미지: {image_path}")
    else:
        pt_input = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        tflite_uint8 = np.random.randint(0, 255, (1, imgsz, imgsz, 3), dtype=np.uint8)
        tflite_fp32  = np.random.rand(1, imgsz, imgsz, 3).astype(np.float32)
        print("  테스트 입력: 랜덤 텐서")

    # --- 원본 YOLO 속도 ---
    print(f"\n[1/2] 원본 YOLO: {pt_path}")
    pt_model = YOLO(pt_path)

    for _ in range(warmup):
        pt_model(pt_input, verbose=False)

    pt_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        pt_model(pt_input, verbose=False)
        pt_times.append(time.perf_counter() - t0)

    pt_avg  = float(np.mean(pt_times)) * 1000
    pt_min  = float(np.min(pt_times)) * 1000
    pt_p95  = float(np.percentile(pt_times, 95)) * 1000
    pt_fps  = 1000.0 / pt_avg

    print(f"  평균: {pt_avg:.1f} ms  |  최소: {pt_min:.1f} ms  |  P95: {pt_p95:.1f} ms  |  FPS: {pt_fps:.1f}")

    # --- TFLite 속도 (raw interpreter) ---
    print(f"\n[2/2] TFLite: {tflite_path}")
    interpreter = tf.lite.Interpreter(model_path=tflite_path, num_threads=4)
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    input_idx    = input_detail['index']
    input_dtype  = input_detail['dtype']

    tflite_input = tflite_uint8 if input_dtype == np.uint8 else tflite_fp32

    for _ in range(warmup):
        interpreter.set_tensor(input_idx, tflite_input)
        interpreter.invoke()

    tflite_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_idx, tflite_input)
        interpreter.invoke()
        tflite_times.append(time.perf_counter() - t0)

    tflite_avg = float(np.mean(tflite_times)) * 1000
    tflite_min = float(np.min(tflite_times)) * 1000
    tflite_p95 = float(np.percentile(tflite_times, 95)) * 1000
    tflite_fps = 1000.0 / tflite_avg

    print(f"  평균: {tflite_avg:.1f} ms  |  최소: {tflite_min:.1f} ms  |  P95: {tflite_p95:.1f} ms  |  FPS: {tflite_fps:.1f}")
    print(f"  입력 dtype: {input_dtype.__name__}")

    # --- 비교 ---
    speedup = pt_avg / tflite_avg

    print("\n" + "=" * 55)
    print("결과 비교")
    print("=" * 55)
    print(f"{'':15s} {'YOLO':>10s} {'TFLite':>10s} {'배율':>8s}")
    print(f"{'평균 (ms)':15s} {pt_avg:>10.1f} {tflite_avg:>10.1f} {speedup:>7.2f}x")
    print(f"{'최소 (ms)':15s} {pt_min:>10.1f} {tflite_min:>10.1f}")
    print(f"{'P95 (ms)':15s} {pt_p95:>10.1f} {tflite_p95:>10.1f}")
    print(f"{'FPS':15s} {pt_fps:>10.1f} {tflite_fps:>10.1f}")
    print(f"\n  모델 크기: {Path(tflite_path).stat().st_size / 1024 / 1024:.1f} MB (TFLite)")
    print("  ※ 실제 NPU 속도는 타겟 디바이스에서 delegate 적용 후 측정하세요.")

    return {
        "pt":     {"avg_ms": pt_avg,     "min_ms": pt_min,     "p95_ms": pt_p95,     "fps": pt_fps},
        "tflite": {"avg_ms": tflite_avg, "min_ms": tflite_min, "p95_ms": tflite_p95, "fps": tflite_fps},
        "speedup": speedup,
    }
