"""Track 기반 강아지 crop 수집기 — ReID 학습 데이터 자동 생성용.

운영 스트림에서 track 별 crop 을 비동기로 저장한다. ReID 학습(models/train/reid_train.py)의
identity 폴더 포맷과 호환되는 구조로 떨어진다:

    {output_dir}/{stream_id}__track_{tid:05d}/{ts_ms}.jpg

설계 원칙 — "메인 추론 루프를 절대 막지 않는다":
    - submit() 은 메인 루프에서 호출되며 큐에 put_nowait 만 한다 (논블로킹).
    - 큐가 가득 차면 조용히 drop (운영 안정성 > 데이터 완전성).
    - 실제 디스크 I/O 와 blur 필터는 백그라운드 데몬 스레드에서 처리.
    - crop 은 .copy() 로 떼어내 메인 루프의 frame 재사용과 분리.

품질/용량 제어:
    - sample_interval_sec: track 당 N초에 1장만 (다양한 포즈 확보 + 디스크 절약)
    - max_per_track: track 당 최대 장수 (한 폴더 폭증 방지)
    - min_box_size: 너무 작은 crop 제외 (저해상도 노이즈)
    - blur_threshold: Laplacian variance 미만이면 흐린 crop 으로 보고 제외
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CropCollector:
    """track 별 crop 을 비동기로 저장하는 수집기. 스레드 안전."""

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
    ):
        self.output_dir = output_dir
        self.enabled = enabled
        self.sample_interval_sec = sample_interval_sec
        self.max_per_track = max_per_track
        self.min_box_size = min_box_size
        self.blur_threshold = blur_threshold
        self.margin = margin
        self.jpeg_quality = int(jpeg_quality)

        self._queue: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()

        # 메인 스레드에서만 접근하는 샘플링 상태 (lock 불필요)
        self._last_saved: Dict[Tuple[str, int], float] = {}   # (stream, tid) -> 마지막 저장 ts
        self._saved_count: Dict[Tuple[str, int], int] = {}    # (stream, tid) -> 누적 장수

        # 통계 (디버깅/모니터링용)
        self._dropped = 0
        self._saved = 0
        self._skipped_blur = 0

    # ------------------------------------------------------------------
    # 라이프사이클
    # ------------------------------------------------------------------

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        self._running.set()
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="CropCollector"
        )
        self._thread.start()
        logger.info(
            f"CropCollector started → {self.output_dir} "
            f"(interval={self.sample_interval_sec}s, max/track={self.max_per_track}, "
            f"min_size={self.min_box_size}, blur_thr={self.blur_threshold})"
        )

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        self._running.clear()
        try:
            self._queue.put_nowait(None)  # sentinel → 워커 즉시 깨우기
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)
        self._thread = None
        logger.info(
            f"CropCollector stopped (saved={self._saved}, dropped={self._dropped}, "
            f"skipped_blur={self._skipped_blur})"
        )

    # ------------------------------------------------------------------
    # 메인 루프에서 호출 (논블로킹)
    # ------------------------------------------------------------------

    def submit(
        self,
        stream_id: str,
        frame: Optional[np.ndarray],
        boxes,
        track_ids,
    ) -> None:
        """track 결과를 받아 샘플링 조건을 통과한 crop 만 큐에 적재.

        Args:
            stream_id: 스트림 식별자 (출력 폴더 네임스페이스)
            frame: 원본 BGR 프레임 (numpy). YOLO 가 실제 실행된 프레임만 넘길 것.
            boxes: center-xywh 박스 리스트 [(cx, cy, w, h), ...]
            track_ids: 박스와 1:1 대응되는 track id 리스트
        """
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

            # 1) 시간 샘플링 — track 당 N초에 1장
            if now - self._last_saved.get(key, 0.0) < self.sample_interval_sec:
                continue
            # 2) track 당 최대 장수
            if self._saved_count.get(key, 0) >= self.max_per_track:
                continue

            cx, cy, bw, bh = box
            # 3) 원본 박스 크기 필터
            if bw < self.min_box_size or bh < self.min_box_size:
                continue

            # 4) margin 포함 좌표 (프레임 경계 clamp)
            x1 = max(0, int(cx - bw / 2) - self.margin)
            y1 = max(0, int(cy - bh / 2) - self.margin)
            x2 = min(fw, int(cx + bw / 2) + self.margin)
            y2 = min(fh, int(cy + bh / 2) + self.margin)
            if x2 - x1 < self.min_box_size or y2 - y1 < self.min_box_size:
                continue

            crop = frame[y1:y2, x1:x2].copy()  # copy: 메인 루프의 frame 재사용과 분리

            try:
                self._queue.put_nowait((stream_id, tid, crop, now))
            except queue.Full:
                self._dropped += 1
                continue
            # 큐 적재 성공한 것만 샘플링 상태 갱신
            self._last_saved[key] = now
            self._saved_count[key] = self._saved_count.get(key, 0) + 1

    # ------------------------------------------------------------------
    # 백그라운드 워커
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        # running 이 꺼져도 큐에 남은 것은 모두 비운 뒤 종료
        while self._running.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:  # sentinel
                break
            stream_id, tid, crop, ts = item
            try:
                self._save(stream_id, tid, crop, ts)
            except Exception as e:
                logger.debug(f"crop save failed ({stream_id} tid={tid}): {e}")

    def _save(self, stream_id: str, tid: int, crop: np.ndarray, ts: float) -> None:
        if crop is None or crop.size == 0:
            return
        # blur 필터 — Laplacian variance 가 낮으면 흐린 crop 으로 보고 버림
        if self.blur_threshold > 0:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < self.blur_threshold:
                self._skipped_blur += 1
                return

        safe_stream = stream_id.replace("/", "_").replace("..", "_")
        track_dir = os.path.join(self.output_dir, f"{safe_stream}__track_{tid:05d}")
        os.makedirs(track_dir, exist_ok=True)

        path = os.path.join(track_dir, f"{int(ts * 1000)}.jpg")
        if cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]):
            self._saved += 1

    # ------------------------------------------------------------------
    # 모니터링
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        return {
            "saved": self._saved,
            "dropped": self._dropped,
            "skipped_blur": self._skipped_blur,
            "queued": self._queue.qsize(),
            "tracked_keys": len(self._saved_count),
        }
