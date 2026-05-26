"""Track 기반 강아지 crop 수집기 — ReID 학습 데이터 자동 생성용.

운영 스트림에서 track 별 crop 을 비동기로 저장한다. ReID 학습(models/train/reid_train.py)의
identity 폴더 포맷과 호환되는 구조로 떨어진다:

    {output_dir}/{stream_id}__track_{tid:05d}/{ts_ms}.jpg

S3 업로드(선택):
    s3_bucket 지정 시 로컬 저장 후 백그라운드 스레드가 S3 로 업로드하고,
    delete_after_upload=True 면 업로드 성공한 파일은 로컬에서 삭제한다.
    → 로컬 디스크는 "업로드 대기 버퍼" 로만 쓰이고 채워지지 않는다.
    업로드 실패한 파일은 로컬에 남아 다음 실행 때 정리/재시도 가능 (안전).

설계 원칙 — "메인 추론 루프를 절대 막지 않는다":
    - submit() 은 큐에 put_nowait 만 한다 (논블로킹). 큐 full 이면 조용히 drop.
    - 저장(disk) → 업로드(S3) 는 각각 별도 백그라운드 데몬 스레드.
    - crop 은 .copy() 로 떼어내 메인 루프의 frame 재사용과 분리.

품질/용량 제어:
    - sample_interval_sec: track 당 N초에 1장 (다양한 포즈 + 디스크 절약)
    - max_per_track: track 당 최대 장수
    - min_box_size: 너무 작은 crop 제외
    - blur_threshold: Laplacian variance 미만이면 흐린 crop 으로 보고 제외
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CropCollector:
    """track 별 crop 을 비동기로 저장(+선택적 S3 업로드)하는 수집기. 스레드 안전."""

    def __init__(
        self,
        output_dir: str,
        enabled: bool = True,
        sample_interval_sec: float = 1.0,
        max_per_track: int = 60,
        min_box_size: int = 48,
        blur_threshold: float = 20.0,
        margin: int = 8,
        jpeg_quality: int = 90,
        queue_size: int = 256,
        # --- S3 옵션 ---
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "crops",
        s3_region: Optional[str] = None,
        delete_after_upload: bool = True,
        upload_queue_size: int = 512,
    ):
        self.output_dir = output_dir
        self.enabled = enabled
        self.sample_interval_sec = sample_interval_sec
        self.max_per_track = max_per_track
        self.min_box_size = min_box_size
        self.blur_threshold = blur_threshold
        self.margin = margin
        self.jpeg_quality = int(jpeg_quality)

        # S3
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix.rstrip("/")
        self.s3_region = s3_region or os.environ.get("AWS_REGION", "ap-northeast-2")
        self.delete_after_upload = delete_after_upload
        self._s3_enabled = bool(s3_bucket)
        self._s3 = None

        self._queue: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self._upload_queue: "queue.Queue" = queue.Queue(maxsize=upload_queue_size)
        self._save_thread: Optional[threading.Thread] = None
        self._upload_thread: Optional[threading.Thread] = None
        self._running = threading.Event()

        # 메인 스레드에서만 접근하는 샘플링 상태 (lock 불필요)
        self._last_saved: Dict[Tuple[str, int], float] = {}
        self._saved_count: Dict[Tuple[str, int], int] = {}

        # 통계
        self._dropped = 0
        self._saved = 0
        self._skipped_blur = 0
        self._uploaded = 0
        self._upload_dropped = 0
        self._upload_failed = 0

    # ------------------------------------------------------------------
    # 라이프사이클
    # ------------------------------------------------------------------

    def start(self) -> None:
        if not self.enabled or self._save_thread is not None:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        self._running.set()

        self._save_thread = threading.Thread(
            target=self._save_worker, daemon=True, name="CropSave"
        )
        self._save_thread.start()

        if self._s3_enabled:
            try:
                import boto3
                self._s3 = boto3.client("s3", region_name=self.s3_region)
                self._upload_thread = threading.Thread(
                    target=self._upload_worker, daemon=True, name="CropUpload"
                )
                self._upload_thread.start()
            except Exception as e:
                logger.warning(f"CropCollector S3 init failed, local-only: {e}")
                self._s3_enabled = False
                self._s3 = None

        logger.info(
            f"CropCollector started → {self.output_dir} "
            f"(interval={self.sample_interval_sec}s, max/track={self.max_per_track}, "
            f"min_size={self.min_box_size}, blur_thr={self.blur_threshold}, "
            f"s3={'on→' + self.s3_bucket if self._s3_enabled else 'off'})"
        )

    def stop(self, timeout: float = 10.0) -> None:
        if self._save_thread is None:
            return
        self._running.clear()
        try:
            self._queue.put_nowait(None)  # save sentinel
        except queue.Full:
            pass
        self._save_thread.join(timeout=timeout)
        self._save_thread = None

        if self._upload_thread is not None:
            try:
                self._upload_queue.put_nowait(None)  # upload sentinel
            except queue.Full:
                pass
            self._upload_thread.join(timeout=timeout)
            self._upload_thread = None

        logger.info(
            f"CropCollector stopped (saved={self._saved}, uploaded={self._uploaded}, "
            f"dropped={self._dropped}, skipped_blur={self._skipped_blur}, "
            f"upload_dropped={self._upload_dropped}, upload_failed={self._upload_failed})"
        )

    # ------------------------------------------------------------------
    # 메인 루프에서 호출 (논블로킹)
    # ------------------------------------------------------------------

    def submit(self, stream_id: str, frame: Optional[np.ndarray], boxes, track_ids) -> None:
        """track 결과를 받아 샘플링 조건을 통과한 crop 만 큐에 적재. boxes 는 center-xywh."""
        if not self.enabled or frame is None:
            return
        if boxes is None or track_ids is None or len(track_ids) == 0:
            return

        now = time.time()
        fh, fw = frame.shape[:2]

        for box, tid in zip(boxes, track_ids):
            try:
                tid = int(tid)
            except (TypeError, ValueError):
                continue
            key = (stream_id, tid)

            if now - self._last_saved.get(key, 0.0) < self.sample_interval_sec:
                continue
            if self._saved_count.get(key, 0) >= self.max_per_track:
                continue

            cx, cy, bw, bh = box
            if bw < self.min_box_size or bh < self.min_box_size:
                continue

            x1 = max(0, int(cx - bw / 2) - self.margin)
            y1 = max(0, int(cy - bh / 2) - self.margin)
            x2 = min(fw, int(cx + bw / 2) + self.margin)
            y2 = min(fh, int(cy + bh / 2) + self.margin)
            if x2 - x1 < self.min_box_size or y2 - y1 < self.min_box_size:
                continue

            crop = frame[y1:y2, x1:x2].copy()  # frame 재사용과 분리

            try:
                self._queue.put_nowait((stream_id, tid, crop, now))
            except queue.Full:
                self._dropped += 1
                continue
            self._last_saved[key] = now
            self._saved_count[key] = self._saved_count.get(key, 0) + 1

    # ------------------------------------------------------------------
    # 백그라운드 워커: 저장
    # ------------------------------------------------------------------

    def _save_worker(self) -> None:
        while self._running.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            stream_id, tid, crop, ts = item
            try:
                self._save(stream_id, tid, crop, ts)
            except Exception as e:
                logger.debug(f"crop save failed ({stream_id} tid={tid}): {e}")

    def _save(self, stream_id: str, tid: int, crop: np.ndarray, ts: float) -> None:
        if crop is None or crop.size == 0:
            return
        if self.blur_threshold > 0:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < self.blur_threshold:
                self._skipped_blur += 1
                return

        safe_stream = stream_id.replace("/", "_").replace("..", "_")
        rel_dir = f"{safe_stream}__track_{tid:05d}"
        track_dir = os.path.join(self.output_dir, rel_dir)
        os.makedirs(track_dir, exist_ok=True)

        fname = f"{int(ts * 1000)}.jpg"
        path = os.path.join(track_dir, fname)
        if not cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]):
            return
        self._saved += 1

        # S3 업로드 큐에 적재 (S3 enabled 시). 큐 full 이면 로컬에 남겨둠 (다음 기회).
        if self._s3_enabled:
            s3_key = f"{self.s3_prefix}/{rel_dir}/{fname}"
            try:
                self._upload_queue.put_nowait((path, s3_key))
            except queue.Full:
                self._upload_dropped += 1

    # ------------------------------------------------------------------
    # 백그라운드 워커: S3 업로드 → 성공 시 로컬 삭제
    # ------------------------------------------------------------------

    def _upload_worker(self) -> None:
        while self._running.is_set() or not self._upload_queue.empty():
            try:
                item = self._upload_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            path, s3_key = item
            if not os.path.exists(path):
                continue
            try:
                self._s3.upload_file(path, self.s3_bucket, s3_key)
                self._uploaded += 1
                if self.delete_after_upload:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            except Exception as e:
                # 업로드 실패 → 로컬 파일 보존 (재시도/수동 정리 가능)
                self._upload_failed += 1
                logger.debug(f"S3 upload failed ({s3_key}): {e}")

    # ------------------------------------------------------------------
    # 모니터링
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        return {
            "saved": self._saved,
            "uploaded": self._uploaded,
            "dropped": self._dropped,
            "skipped_blur": self._skipped_blur,
            "upload_dropped": self._upload_dropped,
            "upload_failed": self._upload_failed,
            "queued": self._queue.qsize(),
            "upload_queued": self._upload_queue.qsize(),
            "tracked_keys": len(self._saved_count),
        }
