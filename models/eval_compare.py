"""
여러 YOLO 모델을 동일 data.yaml 기준으로 비교 평가.

비교 항목:
  - 정확도   : mAP50, mAP50-95, precision, recall  (ultralytics val 파이프라인)
  - 추론속도 : preprocess / inference / postprocess (ms/img), FPS
  - 자원사용 : GPU peak memory(MB), 모델 파일 크기(MB), 파라미터 수(M)

.pt / .engine(TensorRT) / .onnx 모두 같은 인터페이스로 평가한다
(ultralytics YOLO 가 포맷을 알아서 처리).

사용 예:
    python -m models.eval_compare --data dataset/data.yaml \
        --models weights/best.pt weights/best.engine weights/yolo26s.pt \
        --imgsz 640 --device 0 --half

    # 디렉토리에서 자동 탐색
    python -m models.eval_compare --data dataset/data.yaml \
        --models-dir weights --pattern "*.pt" "*.engine" --device 0

    # 결과 저장
    python -m models.eval_compare --data dataset/data.yaml \
        --models weights/*.pt --csv compare.csv --json compare.json
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path
from typing import Optional


def _human_size_mb(path: str) -> Optional[float]:
    """파일 크기(MB). 없으면 None."""
    try:
        return Path(path).stat().st_size / (1024 * 1024)
    except OSError:
        return None


def _param_count_m(model) -> Optional[float]:
    """파라미터 수(백만). .engine/.onnx 처럼 torch 모듈이 없으면 None."""
    try:
        n = sum(p.numel() for p in model.model.parameters())
        return n / 1e6
    except Exception:
        return None


def _resolve_cuda_index(device: str) -> Optional[int]:
    """device 문자열에서 cuda 인덱스 추출. cpu/mps 면 None."""
    d = str(device).strip().lower()
    if d in ("cpu", "mps", ""):
        return None
    # "0", "cuda:0", "cuda" 등
    if d.startswith("cuda:"):
        d = d.split(":", 1)[1]
    if d == "cuda":
        return 0
    try:
        return int(d)
    except ValueError:
        return None


def evaluate_model(
    model_path: str,
    data_yaml: str,
    imgsz: int = 640,
    batch: int = 1,
    device: str = "0",
    half: bool = False,
    conf: float = 0.001,
    iou: float = 0.7,
    split: str = "val",
) -> dict:
    """단일 모델 평가 → 메트릭 dict 반환.

    Args:
        model_path: .pt / .engine / .onnx 경로
        data_yaml:  val set 이 정의된 data.yaml
        imgsz:      추론 해상도 (모델 export 해상도와 맞춰야 함, 특히 .engine)
        batch:      배치 크기. 1 이면 단일 이미지 지연시간(latency) 측정에 적합
        device:     "0" / "cuda:0" / "cpu"
        half:       FP16 추론 (GPU 전용)
        conf:       confidence threshold. mAP 계산은 관례상 낮게(0.001)
        iou:        NMS IoU threshold
        split:      평가 split (val/test)
    """
    from ultralytics import YOLO

    try:
        import torch
    except ImportError:
        torch = None

    cuda_idx = _resolve_cuda_index(device)
    use_cuda = torch is not None and cuda_idx is not None and torch.cuda.is_available()

    result: dict = {"model": model_path, "ok": False}

    # GPU peak memory 측정 준비 (모델 로드 + val 전체 구간)
    # torch 버전에 따라 인덱스 인자 전달이 까다로워서, set_device 후 인자 없이 호출 + 방어적.
    if use_cuda:
        try:
            torch.cuda.set_device(cuda_idx)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    t_load0 = time.perf_counter()
    model = YOLO(model_path, task="detect")
    load_s = time.perf_counter() - t_load0

    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device,
        half=half,
        conf=conf,
        iou=iou,
        split=split,
        verbose=False,
        plots=False,
    )

    # --- 정확도 ---
    box = metrics.box
    result["map50"] = float(box.map50)
    result["map5095"] = float(box.map)
    result["precision"] = float(box.mp)   # mean precision
    result["recall"] = float(box.mr)      # mean recall

    # --- 속도 (ms/img) — ultralytics val 이 측정한 평균값 ---
    speed = getattr(metrics, "speed", {}) or {}
    pre = float(speed.get("preprocess", 0.0))
    inf = float(speed.get("inference", 0.0))
    post = float(speed.get("postprocess", 0.0))
    result["pre_ms"] = pre
    result["inf_ms"] = inf
    result["post_ms"] = post
    result["total_ms"] = pre + inf + post
    result["fps"] = (1000.0 / inf) if inf > 0 else 0.0

    # --- 자원 ---
    result["gpu_mem_mb"] = None
    if use_cuda:
        try:
            result["gpu_mem_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except Exception:
            result["gpu_mem_mb"] = None
    result["size_mb"] = _human_size_mb(model_path)
    result["params_m"] = _param_count_m(model)
    result["load_s"] = load_s
    result["ok"] = True

    # 메모리 정리 (다음 모델 측정 오염 방지)
    del model
    if use_cuda:
        torch.cuda.empty_cache()

    return result


def _collect_model_paths(args) -> list[str]:
    """--models / (--models-dir + --pattern) 로부터 모델 경로 목록 수집."""
    paths: list[str] = []
    if args.models:
        for m in args.models:
            # 셸이 안 풀어준 glob 도 처리
            expanded = sorted(glob.glob(m)) if any(c in m for c in "*?[") else [m]
            paths.extend(expanded or [m])
    if args.models_dir:
        for pat in (args.pattern or ["*.pt", "*.engine", "*.onnx"]):
            paths.extend(sorted(glob.glob(str(Path(args.models_dir) / pat))))
    # 중복 제거(순서 유지)
    seen = set()
    uniq = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _print_table(rows: list[dict]) -> None:
    """비교 결과를 정렬된 표로 출력."""
    header = (
        f"{'model':<28} {'mAP50':>7} {'mAP50-95':>9} {'P':>6} {'R':>6} "
        f"{'inf(ms)':>8} {'FPS':>7} {'GPU(MB)':>8} {'size(MB)':>9} {'params(M)':>10}"
    )
    print("\n" + "=" * len(header))
    print("모델 비교 결과")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        name = Path(r["model"]).name
        if len(name) > 28:
            name = "…" + name[-27:]
        if not r.get("ok"):
            print(f"{name:<28} {'ERROR: ' + str(r.get('error', ''))[:80]}")
            continue

        gpu = r.get("gpu_mem_mb")
        size = r.get("size_mb")
        params = r.get("params_m")
        gpu_s = f"{gpu:.1f}" if gpu is not None else "-"
        size_s = f"{size:.1f}" if size is not None else "-"
        params_s = f"{params:.2f}" if params is not None else "-"
        print(
            f"{name:<28} "
            f"{r['map50']:>7.4f} {r['map5095']:>9.4f} {r['precision']:>6.3f} {r['recall']:>6.3f} "
            f"{r['inf_ms']:>8.2f} {r['fps']:>7.1f} "
            f"{gpu_s:>8} {size_s:>9} {params_s:>10}"
        )
    print("=" * len(header))


def _plot_results(rows: list[dict], out_path: str, dpi: int = 150) -> None:
    """비교 결과를 그래프로 저장.

    2x2 구성:
      (1) 정확도-속도 트레이드오프 산점도 (inference ms vs mAP50-95, 버블=모델크기)
      (2) mAP50 / mAP50-95 막대
      (3) FPS 막대
      (4) GPU peak memory(MB) 막대
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless (서버)
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 미설치 → 그래프 건너뜀 (pip install matplotlib)")
        return

    ok = [r for r in rows if r.get("ok")]
    if not ok:
        print("성공한 모델이 없어 그래프 생략")
        return

    names = [Path(r["model"]).name for r in ok]
    map50 = [r["map50"] for r in ok]
    map5095 = [r["map5095"] for r in ok]
    fps = [r["fps"] for r in ok]
    inf_ms = [r["inf_ms"] for r in ok]
    gpu = [r.get("gpu_mem_mb") or 0.0 for r in ok]
    sizes = [r.get("size_mb") or 0.0 for r in ok]

    x = range(len(names))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("YOLO Model Comparison", fontsize=15, fontweight="bold")

    # (1) accuracy vs speed tradeoff
    ax = axes[0][0]
    smax = max(sizes) or 1.0
    bubble = [80 + 600 * (s / smax) for s in sizes]  # 버블 크기 = 모델 파일크기
    sc = ax.scatter(inf_ms, map5095, s=bubble, alpha=0.6, c=range(len(names)), cmap="viridis")
    for xi, yi, nm in zip(inf_ms, map5095, names):
        ax.annotate(nm, (xi, yi), fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Inference (ms/img)  ← faster")
    ax.set_ylabel("mAP50-95  ↑ better")
    ax.set_title("Accuracy vs Speed (bubble = file size)")
    ax.grid(True, alpha=0.3)

    # (2) mAP bars (grouped)
    ax = axes[0][1]
    w = 0.38
    ax.bar([i - w / 2 for i in x], map50, w, label="mAP50", color="#4C72B0")
    ax.bar([i + w / 2 for i in x], map5095, w, label="mAP50-95", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mAP")
    ax.set_title("Accuracy (mAP)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # (3) FPS bars
    ax = axes[1][0]
    ax.bar(list(x), fps, color="#55A868")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("FPS  ↑ better")
    ax.set_title("Throughput (FPS, 1000/inference)")
    ax.grid(True, axis="y", alpha=0.3)
    for xi, v in zip(x, fps):
        ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    # (4) GPU memory bars
    ax = axes[1][1]
    ax.bar(list(x), gpu, color="#C44E52")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("GPU peak memory (MB)  ↓ better")
    ax.set_title("Resource (GPU memory)")
    ax.grid(True, axis="y", alpha=0.3)
    for xi, v in zip(x, gpu):
        ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"그래프 저장: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="여러 YOLO 모델을 동일 data.yaml 로 정확도/속도/자원 비교",
    )
    ap.add_argument("--data", required=True, help="data.yaml 경로 (val set 정의)")
    ap.add_argument("--models", nargs="*", default=[], help="모델 경로들 (.pt/.engine/.onnx, glob 가능)")
    ap.add_argument("--models-dir", default=None, help="모델 디렉토리 (--pattern 과 함께 자동 탐색)")
    ap.add_argument("--pattern", nargs="*", default=None, help="--models-dir 안에서 찾을 패턴 (기본: *.pt *.engine *.onnx)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1, help="배치 크기 (1=단일이미지 latency 측정 적합)")
    ap.add_argument("--device", default="0", help='"0" / "cuda:0" / "cpu"')
    ap.add_argument("--half", action="store_true", help="FP16 추론 (GPU)")
    ap.add_argument("--conf", type=float, default=0.001, help="confidence threshold (mAP 계산은 낮게 권장)")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    ap.add_argument("--split", default="val", help="평가 split (val/test)")
    ap.add_argument("--csv", default=None, help="결과 CSV 저장 경로")
    ap.add_argument("--json", default=None, help="결과 JSON 저장 경로")
    ap.add_argument("--plot", default=None, help="비교 그래프 PNG 저장 경로 (예: compare.png)")
    args = ap.parse_args()

    model_paths = _collect_model_paths(args)
    if not model_paths:
        ap.error("평가할 모델이 없습니다. --models 또는 --models-dir 를 지정하세요.")

    print(f"data.yaml : {args.data}")
    print(f"설정      : imgsz={args.imgsz} batch={args.batch} device={args.device} "
          f"half={args.half} conf={args.conf} iou={args.iou} split={args.split}")
    print(f"모델 {len(model_paths)}개: " + ", ".join(Path(m).name for m in model_paths))

    rows: list[dict] = []
    for i, mp in enumerate(model_paths, 1):
        print(f"\n[{i}/{len(model_paths)}] 평가 중: {mp}")
        try:
            row = evaluate_model(
                mp, args.data,
                imgsz=args.imgsz, batch=args.batch, device=args.device,
                half=args.half, conf=args.conf, iou=args.iou, split=args.split,
            )
        except Exception as e:  # 한 모델 실패해도 나머지 계속
            import traceback
            print(f"  ! 실패: {e}")
            traceback.print_exc()
            row = {"model": mp, "ok": False, "error": str(e)}
        rows.append(row)

    _print_table(rows)

    if args.csv:
        import csv
        cols = ["model", "ok", "map50", "map5095", "precision", "recall",
                "pre_ms", "inf_ms", "post_ms", "total_ms", "fps",
                "gpu_mem_mb", "size_mb", "params_m", "load_s", "error"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nCSV 저장: {args.csv}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"JSON 저장: {args.json}")

    if args.plot:
        _plot_results(rows, args.plot)


if __name__ == "__main__":
    main()
