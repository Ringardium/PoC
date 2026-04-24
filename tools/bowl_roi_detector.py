"""
스트림/비디오에서 YOLO-World 로 bowl 위치를 자동 검출해 ROI 로 저장합니다.

사용법:
    python tools/bowl_roi_detector.py --input video.mp4 --output roi.json
    python tools/bowl_roi_detector.py --input video.mp4 --output roi.json --debug debug.jpg

동작:
    1. 비디오 앞부분에서 N개 프레임을 균등 샘플링
    2. YOLO-World 에 "dog food bowl" 프롬프트로 검출
    3. 프레임 간 IoU 로 같은 위치 박스끼리 클러스터링 (=stable 박스 찾기)
    4. stability(등장 비율) * confidence 로 스코어링 → top-K 반환
"""
import datetime
import json
import os
from pathlib import Path

import click
import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_PROMPTS = ["dog food bowl", "pet bowl", "dog water bowl"]


def sample_frames(video_path, n_frames, duration_sec):
    """비디오 앞부분 duration_sec 초에서 n_frames 개를 균등 샘플링."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"비디오 열기 실패: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frame = min(total, int(fps * duration_sec))
    if max_frame <= 0:
        cap.release()
        raise RuntimeError(f"유효 프레임이 없음: {video_path}")

    indices = np.linspace(0, max_frame - 1, num=min(n_frames, max_frame), dtype=int)

    samples = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            samples.append((int(idx), frame))

    cap.release()
    if not samples:
        raise RuntimeError("프레임 샘플링 실패")

    h, w = samples[0][1].shape[:2]
    return samples, (h, w)


def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 1e-6 else 0.0


def cluster_detections(detections, iou_threshold=0.5):
    """
    detections: list of (frame_idx, xyxy, conf)
    같은 위치의 박스끼리 묶어서 클러스터 리스트를 반환.
    각 클러스터는 {"boxes": [...], "confs": [...], "frames": set([...])}
    """
    clusters = []
    for frame_idx, box, conf in detections:
        matched = None
        for c in clusters:
            # 클러스터 대표박스(최근 중앙값) 로 비교
            centroid = np.median(np.array(c["boxes"]), axis=0)
            if _iou(box, centroid) >= iou_threshold:
                matched = c
                break

        if matched is None:
            clusters.append({"boxes": [box], "confs": [conf], "frames": {frame_idx}})
        else:
            matched["boxes"].append(box)
            matched["confs"].append(conf)
            matched["frames"].add(frame_idx)

    return clusters


def score_clusters(clusters, total_frames):
    """각 클러스터를 stability * mean_conf 로 스코어링."""
    scored = []
    for c in clusters:
        stability = len(c["frames"]) / max(total_frames, 1)
        mean_conf = float(np.mean(c["confs"]))
        median_box = np.median(np.array(c["boxes"]), axis=0).tolist()
        scored.append({
            "roi_xyxy": [float(v) for v in median_box],
            "confidence": mean_conf,
            "stability": stability,
            "score": stability * mean_conf,
            "n_detections": len(c["boxes"]),
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def detect_bowl_roi(
    video_path,
    model_name="yolov8s-worldv2.pt",
    prompts=None,
    n_frames=60,
    duration_sec=30.0,
    conf_threshold=0.15,
    iou_threshold=0.5,
    min_stability=0.1,
    min_confidence=0.2,
    top_k=3,
    dominance_ratio=2.0,
):
    """비디오에서 bowl ROI 후보를 top-K 반환."""
    prompts = prompts or DEFAULT_PROMPTS

    samples, (frame_h, frame_w) = sample_frames(video_path, n_frames, duration_sec)

    model = YOLO(model_name)
    model.set_classes(prompts)

    frames_only = [f for _, f in samples]
    results = model.predict(frames_only, conf=conf_threshold, verbose=False)

    detections = []
    for (frame_idx, _), res in zip(samples, results):
        if res.boxes is None or len(res.boxes) == 0:
            continue
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        for box, conf in zip(xyxy, confs):
            detections.append((frame_idx, tuple(box.tolist()), float(conf)))

    clusters = cluster_detections(detections, iou_threshold=iou_threshold)
    scored = score_clusters(clusters, total_frames=len(samples))

    filtered = [
        c for c in scored
        if c["stability"] >= min_stability and c["confidence"] >= min_confidence
    ]

    # top-1 이 top-2 대비 dominance_ratio 배 이상이면 단일 bowl 로 확정
    unambiguous = False
    if len(filtered) >= 1:
        top_score = filtered[0]["score"]
        next_score = filtered[1]["score"] if len(filtered) >= 2 else 0.0
        unambiguous = next_score <= 0 or (top_score / next_score) >= dominance_ratio

    return {
        "candidates": filtered[:top_k],
        "all_scored": scored,
        "unambiguous": unambiguous,
        "frame_shape": [frame_h, frame_w],
        "n_sampled": len(samples),
        "prompts": prompts,
    }, samples


def ensure_bowl_roi(video_path, output_path, force=False, **kwargs):
    """output_path 에 ROI JSON 이 없거나 force=True 면 생성, 있으면 그대로 경로 반환.

    main.py 등에서 "eat 시작 시 ROI 없으면 자동 검출" 용도로 사용.
    **kwargs 는 detect_bowl_roi 에 그대로 전달됨.
    """
    if os.path.exists(output_path) and not force:
        return output_path

    result, _ = detect_bowl_roi(video_path=video_path, **kwargs)

    payload = {
        "source": os.path.abspath(video_path),
        "detected_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": kwargs.get("model_name", "yolov8s-worldv2.pt"),
        "prompts": result["prompts"],
        "frame_shape": result["frame_shape"],
        "n_sampled": result["n_sampled"],
        "unambiguous": result["unambiguous"],
        "candidates": result["candidates"],
    }

    if not result["candidates"]:
        raise RuntimeError(
            f"bowl ROI 후보가 검출되지 않았습니다 (video={video_path}). "
            f"threshold 조정 또는 다른 영상 필요."
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return output_path


def draw_debug(samples, candidates, out_path, fallback_scored=None):
    """첫 샘플 프레임에 후보 ROI 를 그려서 저장.

    candidates 가 비어있고 fallback_scored 가 주어지면 거기서 top5 를 회색으로 그림.
    """
    if not samples:
        return
    _, frame = samples[0]
    vis = frame.copy()

    colors = [(0, 255, 0), (0, 165, 255), (255, 0, 255)]
    for i, c in enumerate(candidates):
        x1, y1, x2, y2 = [int(v) for v in c["roi_xyxy"]]
        color = colors[i % len(colors)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"#{i+1} conf={c['confidence']:.2f} stab={c['stability']:.2f}"
        cv2.putText(vis, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 후보 통과 0일 때: 기준 미달이라도 상위 5개를 회색으로 표시 (디버깅)
    if not candidates and fallback_scored:
        for i, c in enumerate(fallback_scored[:5]):
            x1, y1, x2, y2 = [int(v) for v in c["roi_xyxy"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (180, 180, 180), 1)
            label = f"[miss #{i+1}] c={c['confidence']:.2f} s={c['stability']:.2f}"
            cv2.putText(vis, label, (x1, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    cv2.imwrite(out_path, vis)
    print(f"[INFO] 디버그 이미지 저장: {out_path}")


@click.command()
@click.option("--input", "input_path", required=True, help="비디오 파일 경로")
@click.option("--output", default=None, help="ROI JSON 저장 경로 (미지정 시 stdout)")
@click.option("--model", "model_name", default="yolov8s-worldv2.pt",
              help="YOLO-World 가중치")
@click.option("--prompts", default=",".join(DEFAULT_PROMPTS),
              help="쉼표 구분 텍스트 프롬프트")
@click.option("--n-frames", default=60, help="샘플링 프레임 수")
@click.option("--duration", default=30.0, help="샘플링 구간 (초)")
@click.option("--conf", default=0.15, help="검출 confidence 임계값")
@click.option("--min-stability", default=0.1,
              help="최종 후보 최소 등장 비율 (0~1)")
@click.option("--min-confidence", default=0.2,
              help="최종 후보 최소 평균 confidence")
@click.option("--top-k", default=3, help="반환할 후보 개수")
@click.option("--dominance-ratio", default=2.0,
              help="top-1 이 top-2 대비 몇 배 이상이어야 unambiguous 로 판정")
@click.option("--debug", default=None, help="디버그 이미지 저장 경로")
def main(input_path, output, model_name, prompts, n_frames, duration,
         conf, min_stability, min_confidence, top_k, dominance_ratio, debug):
    """YOLO-World 로 bowl ROI 를 자동 검출."""
    prompt_list = [p.strip() for p in prompts.split(",") if p.strip()]

    result, samples = detect_bowl_roi(
        video_path=input_path,
        model_name=model_name,
        prompts=prompt_list,
        n_frames=n_frames,
        duration_sec=duration,
        conf_threshold=conf,
        min_stability=min_stability,
        min_confidence=min_confidence,
        top_k=top_k,
        dominance_ratio=dominance_ratio,
    )

    payload = {
        "source": os.path.abspath(input_path),
        "detected_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "prompts": result["prompts"],
        "frame_shape": result["frame_shape"],
        "n_sampled": result["n_sampled"],
        "unambiguous": result["unambiguous"],
        "candidates": result["candidates"],
    }

    if not result["candidates"]:
        print("[WARN] 기준을 통과한 후보가 없습니다. all_scored:")
        for c in result["all_scored"][:5]:
            print(f"  - score={c['score']:.3f} stab={c['stability']:.2f} "
                  f"conf={c['confidence']:.2f} xyxy={c['roi_xyxy']}")
    elif not result["unambiguous"]:
        print(f"[WARN] top-1 이 top-2 대비 {dominance_ratio}배 미만 → 모호함. "
              f"단일 bowl 확정 보류. 후보들:")
        for i, c in enumerate(result["candidates"]):
            print(f"  #{i+1} score={c['score']:.3f} stab={c['stability']:.2f} "
                  f"conf={c['confidence']:.2f} xyxy={c['roi_xyxy']}")
    else:
        top = result["candidates"][0]
        print(f"[INFO] bowl ROI 확정 (unambiguous): xyxy={top['roi_xyxy']} "
              f"score={top['score']:.3f}")

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[INFO] ROI 저장: {output}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    if debug:
        draw_debug(samples, result["candidates"], debug,
                   fallback_scored=result["all_scored"])


if __name__ == "__main__":
    main()
