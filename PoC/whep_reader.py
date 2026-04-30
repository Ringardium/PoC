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

        # Latest RTCP Sender Report anchor — (rtp_timestamp, ntp_unix_seconds).
        # Populated by hook installed on the video RTCRtpReceiver's
        # _handle_rtcp_packet. SR is sent every ~5s by SRS so this drifts
        # within an SR interval but resyncs each report → max drift ~few ms
        # vs source wallclock (vs unbounded drift in first-PTS anchor mode).
        self._sr_anchor: Optional[Tuple[int, float]] = None

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

    def _compute_wallclock(self, pts: int, time_base, now: float) -> float:
        """Convert RTP pts → unix wallclock.

        Uses RTCP SR anchor when available (NTP-aligned, drift-resistant).
        Otherwise falls back to the first-frame `time.time()` anchor.

        SR anchor format: ``(rtp_timestamp_at_sr, ntp_unix_seconds_at_sr)``.
        Frame wallclock = ntp_unix + (frame.pts - rtp_at_sr) * time_base.

        ``time_base`` is a fractions.Fraction (e.g. 1/90000 for video).
        """
        if self._sr_anchor is not None and time_base is not None:
            try:
                rtp_anchor, ntp_anchor = self._sr_anchor
                # RTP timestamps wrap at 2^32 — handle wrap by treating diff
                # as signed 32-bit (positive small = forward, negative = wrap)
                diff = (pts - rtp_anchor) & 0xFFFFFFFF
                if diff & 0x80000000:
                    diff -= 0x100000000
                pts_sec = float(diff * time_base)
                return ntp_anchor + pts_sec
            except Exception:
                pass
        # Fallback: first-frame anchor (legacy behavior)
        try:
            pts_sec = float((pts - (self._first_pts or 0)) * (self._first_pts_time_base or 0))
        except Exception:
            pts_sec = 0.0
        return (self._first_pts_wallclock or now) + pts_sec

    def has_sr_anchor(self) -> bool:
        """True once at least one RTCP Sender Report has been received."""
        return self._sr_anchor is not None

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

        # Install RTCP Sender Report hook on the video receiver.
        # Failure is non-fatal — falls back to first-PTS anchor.
        try:
            self._install_rtcp_sr_hook()
        except Exception as e:
            logger.warning(f"WHEP RTCP SR hook install failed (will use first-PTS anchor): {e}")

        # Sleep until release(); polling avoids needing a separate async Event
        while not self._closed.is_set():
            await asyncio.sleep(0.2)

        try:
            await self._pc.close()
        except Exception:
            pass

    def _install_rtcp_sr_hook(self):
        """Wrap the video receiver's _handle_rtcp_packet to capture Sender Reports.

        aiortc consumes RTCP SR internally for stats but doesn't expose the
        rtp↔ntp anchor publicly. We monkey-patch the bound method on the
        receiver instance — this is a private API path but the alternative is
        forking aiortc.
        """
        # Pick the video receiver from the active transceivers
        video_recv = None
        for tr in self._pc.getTransceivers():
            if getattr(tr.receiver, "track", None) and tr.receiver.track.kind == "video":
                video_recv = tr.receiver
                break
            # Fallback: first non-stopped recvonly transceiver
            if tr.direction in ("recvonly", "sendrecv") and getattr(tr, "kind", None) == "video":
                video_recv = tr.receiver
                break
        if video_recv is None:
            # Last resort: first receiver from getReceivers
            recvs = self._pc.getReceivers()
            for r in recvs:
                if getattr(r.track, "kind", None) == "video":
                    video_recv = r
                    break
        if video_recv is None:
            raise RuntimeError("video receiver not found")

        from aiortc.rtcp import RtcpSrPacket  # type: ignore

        original = video_recv._handle_rtcp_packet

        async def _wrapped(packet):
            try:
                if isinstance(packet, RtcpSrPacket):
                    info = packet.sender_info
                    # NTP timestamp is a 64-bit field: high 32 bits seconds since
                    # 1900-01-01, low 32 bits fractional seconds. Convert to unix.
                    ntp_seconds = (info.ntp_timestamp >> 32) & 0xFFFFFFFF
                    ntp_frac = info.ntp_timestamp & 0xFFFFFFFF
                    NTP_UNIX_OFFSET = 2208988800  # seconds from 1900 to 1970
                    ntp_unix = (ntp_seconds - NTP_UNIX_OFFSET) + (ntp_frac / 2**32)
                    self._sr_anchor = (info.rtp_timestamp, ntp_unix)
            except Exception:
                pass
            return await original(packet)

        video_recv._handle_rtcp_packet = _wrapped
        logger.info("WHEP RTCP SR hook installed — wallclock will use NTP-aligned anchor")

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

            # Map RTP pts → wallclock.
            # Preferred: use the most recent RTCP Sender Report anchor (NTP-aligned,
            # resyncs every ~5s, eliminates clock drift between source and our local
            # `time.time()`). Fallback: first-PTS anchor (drift unbounded over time).
            wall_ts = self._compute_wallclock(frame.pts, frame.time_base, now)

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
