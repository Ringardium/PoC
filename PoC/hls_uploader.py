"""HLS segmenter + S3 uploader for annotated video frames."""

import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, Set

import boto3
import numpy as np

logger = logging.getLogger(__name__)


class HLSUploader:
    """Pipe annotated frames → FFmpeg HLS → S3 upload.

    FFmpeg encodes raw BGR frames into H.264 HLS segments locally.
    A background thread watches the output directory and uploads new
    .ts segments and the updated .m3u8 playlist to S3.
    """

    def __init__(
        self,
        stream_id: str,
        fps: int,
        width: int,
        height: int,
        s3_bucket: str,
        s3_prefix: str = "hls/live",
        segment_duration: int = 1,
        playlist_size: int = 8,
    ):
        self._stream_id = stream_id
        self._fps = fps
        self._width = width
        self._height = height
        self._s3_bucket = s3_bucket
        self._s3_prefix = s3_prefix.rstrip("/")
        self._segment_duration = segment_duration
        self._playlist_size = playlist_size

        # Local temp directory for HLS output (sanitize slashes for dir name)
        safe_id = stream_id.replace("/", "_").replace("\\", "_")
        self._tmp_dir = Path(tempfile.mkdtemp(prefix=f"hls_{safe_id}_"))
        self._segment_pattern = str(self._tmp_dir / "seg_%05d.ts")
        self._playlist_path = str(self._tmp_dir / "index.m3u8")

        # S3 client
        self._s3 = boto3.client(
            "s3",
            region_name=os.environ.get("AWS_REGION", "ap-northeast-2"),
        )

        # Track uploaded segments for cleanup
        self._uploaded_segments: Set[str] = set()
        self._running = True

        # Clean up any leftover S3 files from previous runs
        self._purge_s3()

        # Start FFmpeg process
        self._proc = self._start_ffmpeg()

        # Start S3 upload watcher thread
        self._upload_thread = threading.Thread(
            target=self._upload_worker, daemon=True
        )
        self._upload_thread.start()

        logger.info(
            f"HLSUploader started: stream={stream_id}, "
            f"s3://{s3_bucket}/{self._s3_prefix}/{stream_id}/, "
            f"seg={segment_duration}s, list={playlist_size}"
        )

    def _purge_s3(self):
        """Delete all existing HLS files on S3 for this stream (leftover from previous runs)."""
        s3_key_prefix = f"{self._s3_prefix}/{self._stream_id}"
        try:
            paginator = self._s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._s3_bucket, Prefix=s3_key_prefix + "/"):
                for obj in page.get("Contents", []):
                    self._s3.delete_object(Bucket=self._s3_bucket, Key=obj["Key"])
                    logger.debug(f"Purged s3://{self._s3_bucket}/{obj['Key']}")
            logger.info(f"S3 purged: s3://{self._s3_bucket}/{s3_key_prefix}/")
        except Exception as e:
            logger.warning(f"S3 purge failed: {e}")

    def _start_ffmpeg(self) -> subprocess.Popen:
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self._width}x{self._height}",
            "-r", str(self._fps),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-g", str(self._fps * self._segment_duration),
            "-f", "hls",
            "-hls_time", str(self._segment_duration),
            "-hls_list_size", str(self._playlist_size),
            "-hls_flags", "delete_segments+omit_endlist",
            "-hls_segment_filename", self._segment_pattern,
            self._playlist_path,
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        logger.debug(f"FFmpeg HLS process started: PID {proc.pid}")
        return proc

    def write_frame(self, frame: np.ndarray):
        """Write a single BGR frame to FFmpeg stdin."""
        if self._proc.stdin is None or self._proc.poll() is not None:
            return
        try:
            self._proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            logger.warning(f"[{self._stream_id}] FFmpeg pipe broken")

    def _upload_worker(self):
        """Watch local HLS dir and upload new files to S3."""
        s3_key_prefix = f"{self._s3_prefix}/{self._stream_id}"
        last_playlist_mtime = 0.0

        while self._running:
            time.sleep(0.5)
            try:
                self._upload_segments(s3_key_prefix)
                last_playlist_mtime = self._upload_playlist(
                    s3_key_prefix, last_playlist_mtime
                )
                self._cleanup_old_segments(s3_key_prefix)
            except Exception as e:
                logger.warning(f"[{self._stream_id}] S3 upload error: {e}")

        # Final upload on shutdown
        try:
            self._upload_segments(s3_key_prefix)
            self._upload_playlist(s3_key_prefix, 0.0)
        except Exception:
            pass

    def _upload_segments(self, s3_key_prefix: str):
        """Upload new .ts segment files."""
        for f in self._tmp_dir.glob("seg_*.ts"):
            name = f.name
            if name not in self._uploaded_segments:
                key = f"{s3_key_prefix}/{name}"
                self._s3.upload_file(
                    str(f), self._s3_bucket, key,
                    ExtraArgs={"ContentType": "video/MP2T"},
                )
                self._uploaded_segments.add(name)
                logger.info(f"S3 upload: s3://{self._s3_bucket}/{key}")

    def _upload_playlist(self, s3_key_prefix: str, last_mtime: float) -> float:
        """Upload .m3u8 if modified, excluding the last segment (may be incomplete)."""
        p = Path(self._playlist_path)
        if not p.exists():
            return last_mtime
        mtime = p.stat().st_mtime
        if mtime > last_mtime:
            # Read and strip last segment entry to avoid playback stutter
            lines = p.read_text().splitlines()
            # Find last #EXTINF + segment pair and remove it
            last_extinf = -1
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].startswith("#EXTINF:"):
                    last_extinf = i
                    break
            if last_extinf >= 0:
                lines = lines[:last_extinf]
            trimmed = "\n".join(lines) + "\n"

            key = f"{s3_key_prefix}/index.m3u8"
            self._s3.put_object(
                Bucket=self._s3_bucket,
                Key=key,
                Body=trimmed.encode("utf-8"),
                ContentType="application/vnd.apple.mpegurl",
                CacheControl="no-cache, no-store",
            )
            logger.debug(f"Uploaded s3://{self._s3_bucket}/{key}")
            return mtime
        return last_mtime

    def _cleanup_old_segments(self, s3_key_prefix: str):
        """Remove S3 segments that FFmpeg already deleted locally."""
        local_segments = {f.name for f in self._tmp_dir.glob("seg_*.ts")}
        stale = self._uploaded_segments - local_segments
        for name in stale:
            key = f"{s3_key_prefix}/{name}"
            try:
                self._s3.delete_object(Bucket=self._s3_bucket, Key=key)
                logger.info(f"S3 delete: s3://{self._s3_bucket}/{key}")
            except Exception:
                pass
            self._uploaded_segments.discard(name)

    def stop(self):
        """Shutdown FFmpeg, delete all S3 files, and clean up."""
        self._running = False

        # Close FFmpeg stdin to trigger graceful exit
        if self._proc.stdin:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
        self._proc.wait(timeout=10)

        # Wait for upload thread to finish
        self._upload_thread.join(timeout=5.0)

        # Delete all segments + playlist from S3
        s3_key_prefix = f"{self._s3_prefix}/{self._stream_id}"
        for name in list(self._uploaded_segments):
            key = f"{s3_key_prefix}/{name}"
            try:
                self._s3.delete_object(Bucket=self._s3_bucket, Key=key)
            except Exception:
                pass
        try:
            self._s3.delete_object(
                Bucket=self._s3_bucket,
                Key=f"{s3_key_prefix}/index.m3u8",
            )
        except Exception:
            pass
        self._uploaded_segments.clear()
        logger.info(f"S3 cleaned up: s3://{self._s3_bucket}/{s3_key_prefix}/")

        # Clean up temp directory
        try:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
        except Exception:
            pass

        logger.info(f"HLSUploader stopped: {self._stream_id}")
