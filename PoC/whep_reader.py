"""WHEP/WebRTC pull adapter that mimics cv2.VideoCapture interface.

Drop-in replacement for VideoCapture when input_source starts with
``whep://`` or ``webrtc://``. Runs aiortc in a dedicated background thread
with its own asyncio loop, exposes sync ``read()`` / ``isOpened()`` / ``get()``
so existing stream_processor code does not need to become async-aware.

URL format:
    whep://HOST:PORT/APP/STREAM[?vhost=X&mode=whep|srs&https=true]

Example:
    whep://118.41.173.32:2985/live/facility-ddnapet_gmail-every1?vhost=kr
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from typing import Optional, Tuple
from urllib.parse import urlparse, parse_qs

import numpy as np

logger = logging.getLogger(__name__)


# cv2.VideoCapture property IDs we care about
CAP_PROP_POS_MSEC = 0
CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4
CAP_PROP_FPS = 5


def parse_whep_url(source: str) -> dict:
    """Parse whep://host:port/app/stream?vhost=...&mode=...&https=true

    Returns dict with keys: base_url, app, stream, vhost, mode.
    """
    parsed = urlparse(source)
    if parsed.scheme not in ("whep", "webrtc"):
        raise ValueError(f"Not a WHEP URL: {source}")
    if not parsed.hostname or not parsed.port:
        raise ValueError(f"WHEP URL requires host:port — got {source}")

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"WHEP URL path must be /app/stream — got {parsed.path}")
    app, stream = parts[0], "/".join(parts[1:])

    q = {k: v[0] for k, v in parse_qs(parsed.query).items()}
    scheme = "https" if q.get("https", "").lower() in ("1", "true", "yes") else "http"
    return {
        "base_url": f"{scheme}://{parsed.hostname}:{parsed.port}",
        "app": app,
        "stream": stream,
        "vhost": q.get("vhost", ""),
        "mode": q.get("mode", "whep"),
    }


class WHEPCapture:
    """cv2.VideoCapture-compatible wrapper around aiortc WHEP pull.

    Thread model:
        - Background thread runs its own asyncio loop with aiortc PeerConnection
        - Incoming video frames are converted to BGR ndarray + (source wallclock)
          and pushed to a thread-safe queue (drops oldest on overflow)
        - ``read()`` (called from stream_processor's executor) pops from the queue
    """

    def __init__(self, source: str, frame_buffer: int = 2, connect_timeout: float = 15.0):
        cfg = parse_whep_url(source)
        self._base_url: str = cfg["base_url"]
        self._app: str = cfg["app"]
        self._stream: str = cfg["stream"]
        self._vhost: str = cfg["vhost"]
        self._mode: str = cfg["mode"]
        self._connect_timeout = connect_timeout

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._pc = None  # RTCPeerConnection, lazy-imported

        self._frame_q: "queue.Queue[Tuple[np.ndarray, float]]" = queue.Queue(maxsize=frame_buffer)
        self._ready = threading.Event()    # set when first frame arrived (open() unblocks)
        self._closed = threading.Event()   # set when release() called
        self._error: Optional[BaseException] = None

        # Stream metadata populated when first frame arrives
        self._width: int = 0
        self._height: int = 0
        self._fps: float = 30.0  # best-effort; aiortc doesn't expose source fps directly
        self._first_pts: Optional[int] = None
        self._first_pts_time_base = None
        self._first_pts_wallclock: Optional[float] = None
        self._last_frame_wallclock: Optional[float] = None

        self._opened = False

    # ------------------------------------------------------------------
    # Public cv2.VideoCapture-like interface
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """Start background session. Returns True if first frame arrived within timeout."""
        if self._thread is not None:
            return self._opened
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"WHEP-{self._stream}")
        self._thread.start()
        ok = self._ready.wait(timeout=self._connect_timeout) and self._error is None
        self._opened = ok
        if not ok and self._error:
            logger.error(f"WHEP open failed: {self._error}")
        return ok

    def isOpened(self) -> bool:
        return (
            self._opened
            and self._thread is not None
            and self._thread.is_alive()
            and not self._closed.is_set()
            and self._error is None
        )

    def read(self, timeout: float = 5.0) -> Tuple[bool, Optional[np.ndarray]]:
        """Block until a frame is available (or timeout). Matches cv2 signature."""
        try:
            img, wall_ts = self._frame_q.get(timeout=timeout)
            self._last_frame_wallclock = wall_ts
            return True, img
        except queue.Empty:
            return False, None

    def get(self, prop: int) -> float:
        if prop == CAP_PROP_FPS:
            return float(self._fps)
        if prop == CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop == CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        if prop == CAP_PROP_POS_MSEC:
            return float((self._last_frame_wallclock or 0.0) * 1000.0)
        return 0.0

    def release(self):
        self._closed.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

    def get_last_wallclock(self) -> Optional[float]:
        """Unix timestamp of the last frame (source-aligned via RTP pts)."""
        return self._last_frame_wallclock

    # ------------------------------------------------------------------
    # Background asyncio loop
    # ------------------------------------------------------------------

    def _run(self):
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._async_main())
        except BaseException as e:
            self._error = e
            self._ready.set()  # unblock open()
            logger.warning(f"WHEP background loop exited: {e}")
        finally:
            if self._loop and not self._loop.is_closed():
                self._loop.close()

    async def _async_main(self):
        # Lazy imports — allow module to load even if aiortc missing
        from aiortc import RTCPeerConnection, RTCSessionDescription
        import aiohttp

        self._pc = RTCPeerConnection()
        self._pc.addTransceiver("video", direction="recvonly")
        self._pc.addTransceiver("audio", direction="recvonly")

        @self._pc.on("track")
        async def _on_track(track):
            if track.kind != "video":
                return
            await self._consume_track(track)

        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)

        answer_sdp = await self._signal(aiohttp)
        await self._pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type="answer"))

        # Sleep until release(); polling avoids needing a separate async Event
        while not self._closed.is_set():
            await asyncio.sleep(0.2)

        try:
            await self._pc.close()
        except Exception:
            pass

    async def _signal(self, aiohttp_mod) -> str:
        qs = f"?app={self._app}&stream={self._stream}"
        if self._vhost:
            qs += f"&vhost={self._vhost}"

        async with aiohttp_mod.ClientSession() as sess:
            if self._mode == "whep":
                base = self._base_url.rstrip("/")
                if not base.endswith("/rtc/v1/whep"):
                    base = f"{base}/rtc/v1/whep"
                url = f"{base}/{qs}"
                async with sess.post(
                    url,
                    data=self._pc.localDescription.sdp,
                    headers={"Content-Type": "application/sdp"},
                ) as resp:
                    body = await resp.text()
                    if resp.status not in (200, 201):
                        raise RuntimeError(f"WHEP signaling failed: {resp.status}\n{body}")
                    return body
            else:  # SRS JSON API
                url = f"{self._base_url.rstrip('/')}/rtc/v1/play/"
                host = self._base_url.split("//")[1].split(":")[0]
                stream_host = self._vhost or host
                payload = {
                    "api": url,
                    "streamurl": f"webrtc://{stream_host}/{self._app}/{self._stream}",
                    "sdp": self._pc.localDescription.sdp,
                }
                async with sess.post(url, json=payload) as resp:
                    body = await resp.text()
                    if resp.status != 200:
                        raise RuntimeError(f"SRS play failed: {resp.status}\n{body}")
                    data = json.loads(body)
                    if data.get("code") != 0:
                        raise RuntimeError(f"SRS play error: {data}")
                    return data["sdp"]

    async def _consume_track(self, track):
        while not self._closed.is_set():
            try:
                frame = await track.recv()
            except Exception as e:
                logger.debug(f"WHEP track ended: {e}")
                return
            try:
                img = frame.to_ndarray(format="bgr24")
            except Exception as e:
                logger.debug(f"frame convert failed: {e}")
                continue
            now = time.time()

            if self._first_pts is None:
                self._first_pts = frame.pts
                self._first_pts_time_base = frame.time_base
                self._first_pts_wallclock = now
                self._width = int(frame.width)
                self._height = int(frame.height)
                self._ready.set()

            # Map RTP pts → wallclock using first-frame anchor.
            # Drift vs true source wallclock is bounded by the RTP/RTCP SR accuracy;
            # good enough for HLS/WebRTC player offset sync.
            try:
                pts_sec = float((frame.pts - self._first_pts) * self._first_pts_time_base)
            except Exception:
                pts_sec = 0.0
            wall_ts = (self._first_pts_wallclock or now) + pts_sec

            # Drop-oldest policy: latency > completeness for live tracking.
            if self._frame_q.full():
                try:
                    self._frame_q.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._frame_q.put_nowait((img, wall_ts))
            except queue.Full:
                pass
