"""Event clip recorder: capture short MP4 clips around behavior events and upload to S3."""

import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional

import boto3
import numpy as np

logger = logging.getLogger(__name__)


class _PendingClip:
    """A clip that is still collecting post-event frames."""

    __slots__ = ("pre_frames", "post_frames", "post_remaining", "event_info", "s3_key")

    def __init__(self, pre_frames: List[np.ndarray], post_needed: int, event_info: dict, s3_key: str):
        self.pre_frames = pre_frames
        self.post_frames: List[np.ndarray] = []
        self.post_remaining = post_needed
        self.event_info = event_info
        self.s3_key = s3_key


class EventClipRecorder:
    """Maintain a frame ring buffer per stream, record short clips on event trigger.

    Usage::

        recorder = EventClipRecorder(stream_id="cam1", fps=15, width=1920, height=1080,
                                     s3_bucket="my-bucket")
        # In main loop:
        recorder.push_frame(frame)

        # When event detected:
        recorder.trigger({"dogId": "뽀삐", "behaviorType": "fight"})

        # On shutdown:
        recorder.stop()
    """

    def __init__(
        self,
        stream_id: str,
        fps: int,
        width: int,
        height: int,
        s3_bucket: str,
        s3_prefix: str = "clips/events",
        pre_seconds: float = 2.0,
        post_seconds: float = 2.0,
        max_pending: int = 5,
    ):
        self._stream_id = stream_id
        self._fps = fps
        self._width = width
        self._height = height
        self._s3_bucket = s3_bucket
        self._s3_prefix = s3_prefix.rstrip("/")
        self._pre_frames_count = max(1, int(fps * pre_seconds))
        self._post_frames_count = max(1, int(fps * post_seconds))
        self._max_pending = max_pending

        # Ring buffer for recent frames (pre-event)
        self._ring: Deque[np.ndarray] = deque(maxlen=self._pre_frames_count)

        # Pending clips collecting post-event frames
        self._pending: List[_PendingClip] = []
        self._lock = threading.Lock()

        # Per-(dog, behavior) clip counter for sequential numbering
        self._clip_counter: Dict[str, int] = {}

        # Background thread for encoding + upload
        self._running = True
        self._encode_queue: List[tuple] = []  # (frames, event_info)
        self._encode_lock = threading.Lock()
        self._encode_event = threading.Event()
        self._worker = threading.Thread(target=self._encode_worker, daemon=True)
        self._worker.start()

        # S3 client (reuse same pattern as HLSUploader)
        self._s3 = boto3.client(
            "s3",
            region_name=os.environ.get("AWS_REGION", "ap-northeast-2"),
        )

        logger.info(
            f"EventClipRecorder started: stream={stream_id}, "
            f"pre={pre_seconds}s ({self._pre_frames_count}f), "
            f"post={post_seconds}s ({self._post_frames_count}f)"
        )

    def push_frame(self, frame: np.ndarray):
        """Add a frame to the ring buffer and feed any pending clip recordings."""
        self._ring.append(frame.copy())

        with self._lock:
            completed = []
            for clip in self._pending:
                clip.post_frames.append(frame.copy())
                clip.post_remaining -= 1
                if clip.post_remaining <= 0:
                    completed.append(clip)

            for clip in completed:
                self._pending.remove(clip)
                all_frames = clip.pre_frames + clip.post_frames
                self._submit_encode(all_frames, clip.event_info, clip.s3_key)

    def trigger(self, event_info: dict) -> Optional[str]:
        """Trigger clip recording for an event. Returns the S3 key for the clip."""
        from datetime import timedelta as _td
        KST = timezone(_td(hours=9))
        now = datetime.now(KST)
        date_str = now.strftime("%Y%m%d")
        behavior = event_info.get("behaviorType", "unknown")
        dog_id = str(event_info.get("dogId", "unknown"))
        safe_dog_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in dog_id)

        # Sequential behavior number per (dog, behavior)
        counter_key = f"{safe_dog_id}_{behavior}"
        self._clip_counter[counter_key] = self._clip_counter.get(counter_key, 0) + 1
        seq = self._clip_counter[counter_key]

        filename = f"{date_str}_{safe_dog_id}_{behavior}_{seq:03d}.mp4"
        s3_key = f"{self._s3_prefix}/{self._stream_id}/{filename}"

        with self._lock:
            if len(self._pending) >= self._max_pending:
                logger.warning(
                    f"[{self._stream_id}] Clip queue full ({self._max_pending}), "
                    f"dropping clip for {behavior}"
                )
                return None

            pre_frames = [f.copy() for f in self._ring]
            clip = _PendingClip(pre_frames, self._post_frames_count, event_info, s3_key)
            self._pending.append(clip)

        logger.info(
            f"[{self._stream_id}] Clip triggered: {behavior} "
            f"dog={dog_id}, pre={len(pre_frames)}f, post={self._post_frames_count}f, "
            f"key={s3_key}"
        )
        return s3_key

    def _submit_encode(self, frames: List[np.ndarray], event_info: dict, s3_key: str):
        """Queue frames for background encoding + upload."""
        with self._encode_lock:
            self._encode_queue.append((frames, event_info, s3_key))
        self._encode_event.set()

    def _encode_worker(self):
        """Background thread: encode frames to MP4 and upload to S3."""
        while self._running:
            self._encode_event.wait(timeout=1.0)
            self._encode_event.clear()

            while True:
                with self._encode_lock:
                    if not self._encode_queue:
                        break
                    frames, event_info, s3_key = self._encode_queue.pop(0)

                try:
                    self._encode_and_upload(frames, event_info, s3_key)
                except Exception as e:
                    logger.warning(f"[{self._stream_id}] Clip encode/upload failed: {e}")

    def _encode_and_upload(self, frames: List[np.ndarray], event_info: dict, s3_key: str):
        """Encode frame list to MP4 via FFmpeg, then upload to S3."""
        if not frames:
            return

        filename = s3_key.rsplit("/", 1)[-1]

        # Temp file for MP4 output
        tmp_dir = tempfile.mkdtemp(prefix="clip_")
        mp4_path = os.path.join(tmp_dir, filename)

        try:
            # FFmpeg: raw BGR frames → H.264 MP4
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{self._width}x{self._height}",
                "-r", str(self._fps),
                "-i", "pipe:0",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                mp4_path,
            ]
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            for frame in frames:
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            proc.wait(timeout=30)

            if proc.returncode != 0:
                stderr = proc.stderr.read().decode("utf-8", errors="replace")
                logger.warning(f"[{self._stream_id}] FFmpeg clip encode failed: {stderr[:500]}")
                return

            # Upload to S3
            self._s3.upload_file(
                mp4_path,
                self._s3_bucket,
                s3_key,
                ExtraArgs={"ContentType": "video/mp4"},
            )
            logger.info(
                f"[{self._stream_id}] Clip uploaded: s3://{self._s3_bucket}/{s3_key} "
                f"({len(frames)} frames, {len(frames)/self._fps:.1f}s)"
            )

        finally:
            # Clean up temp files
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    def stop(self):
        """Stop the recorder and flush remaining clips."""
        self._running = False

        # Flush any pending clips with whatever frames they have
        with self._lock:
            for clip in self._pending:
                all_frames = clip.pre_frames + clip.post_frames
                if all_frames:
                    self._submit_encode(all_frames, clip.event_info, clip.s3_key)
            self._pending.clear()

        # Signal worker and wait
        self._encode_event.set()
        self._worker.join(timeout=30.0)
        logger.info(f"EventClipRecorder stopped: {self._stream_id}")
