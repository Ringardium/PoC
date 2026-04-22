"""WebSocket broadcaster for per-frame detection metadata.

Runs a small aiohttp WebSocket server in a background thread so mobile
clients can subscribe to bbox / track_id / global_id / behavior streams
and render overlays client-side without the server burning CPU on drawing.

Wire format (JSON text frame):
    {
      "type": "frame_metadata",
      "stream_id": "facility-ddnapet_gmail-every1",
      "ts": 1712345678.123,           // unix seconds (source-aligned)
      "tracks": [
        {
          "tid": 12,                  // tracker-assigned ID (may reset)
          "gid": 3,                   // ReID global ID (stable across streams), or null
          "pet_name": "뽀삐",          // from pet_profiles, or null
          "bbox_xywh": [cx, cy, w, h],// center-xywh in source pixels
          "behavior": "sleeping"      // one of: normal, fight, escape, sleeping,
                                      //         bathroom, feeding, playing, inactive
        }, ...
      ],
      "person_boxes": [               // optional — present when privacy=True
        [x1, y1, x2, y2], ...         // corner-xyxy in source pixels (padded)
      ],
      "privacy_method": "blur"        // optional hint: blur | mosaic | black
    }

Usage:
    sender = MetadataSender(port=8766)
    sender.start()
    ...
    sender.push_frame(stream_id, ts, tracks)
    ...
    sender.stop()
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


class MetadataSender:
    """Thread-safe WebSocket broadcaster. Clients receive live JSON frames."""

    def __init__(self, port: int = 8766, host: str = "0.0.0.0", path: str = "/ws/metadata",
                 queue_size: int = 512):
        self._port = port
        self._host = host
        self._path = path
        self._ingress: "queue.Queue[dict]" = queue.Queue(maxsize=queue_size)

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._clients: Set = set()  # aiohttp WebSocketResponse (touched only from loop)
        self._started = threading.Event()
        self._running = threading.Event()
        self._runner = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, wait_timeout: float = 3.0) -> bool:
        if self._thread is not None:
            return self._started.is_set()
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True, name="MetadataSender")
        self._thread.start()
        ok = self._started.wait(timeout=wait_timeout)
        if ok:
            logger.info(f"MetadataSender listening on ws://{self._host}:{self._port}{self._path}")
        else:
            logger.warning("MetadataSender failed to start within timeout")
        return ok

    def stop(self):
        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

    # ------------------------------------------------------------------
    # Ingress (thread-safe, called from stream_processor executor)
    # ------------------------------------------------------------------

    def push(self, message: dict):
        """Non-blocking enqueue; drops message on overflow to protect live loop."""
        try:
            self._ingress.put_nowait(message)
        except queue.Full:
            pass

    def push_frame(self, stream_id: str, ts: float, tracks: List[dict]):
        self.push({
            "type": "frame_metadata",
            "stream_id": stream_id,
            "ts": float(ts),
            "tracks": tracks,
        })

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _run(self):
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            logger.error(f"MetadataSender loop error: {e}")
        finally:
            if self._loop and not self._loop.is_closed():
                self._loop.close()

    async def _serve(self):
        # aiohttp is already an aiortc dependency; avoid adding new deps.
        from aiohttp import web

        app = web.Application()
        app.router.add_get(self._path, self._ws_handler)
        app.router.add_get("/health", self._health_handler)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        try:
            await site.start()
        except Exception as e:
            logger.error(f"MetadataSender bind failed on {self._host}:{self._port}: {e}")
            return

        self._started.set()
        drain = asyncio.create_task(self._drain_loop())
        try:
            while self._running.is_set():
                await asyncio.sleep(0.2)
        finally:
            drain.cancel()
            for ws in list(self._clients):
                try:
                    await ws.close()
                except Exception:
                    pass
            self._clients.clear()
            await self._runner.cleanup()

    async def _health_handler(self, request):
        from aiohttp import web
        return web.json_response({"clients": len(self._clients), "queued": self._ingress.qsize()})

    async def _ws_handler(self, request):
        from aiohttp import web

        ws = web.WebSocketResponse(heartbeat=15)
        await ws.prepare(request)
        self._clients.add(ws)
        peer = request.remote
        logger.info(f"metadata client connected: {peer} (total={len(self._clients)})")
        try:
            async for _msg in ws:
                # Ignore any client->server messages for now
                pass
        finally:
            self._clients.discard(ws)
            logger.info(f"metadata client disconnected: {peer} (total={len(self._clients)})")
        return ws

    async def _drain_loop(self):
        """Pump messages from the thread-safe ingress queue into all WS clients."""
        loop = asyncio.get_event_loop()
        while self._running.is_set():
            message = await loop.run_in_executor(None, self._blocking_get, 0.2)
            if message is None:
                continue
            if not self._clients:
                continue
            payload = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
            dead = []
            for client in list(self._clients):
                try:
                    if client.closed:
                        dead.append(client)
                        continue
                    await client.send_str(payload)
                except Exception:
                    dead.append(client)
            for d in dead:
                self._clients.discard(d)

    def _blocking_get(self, timeout: float):
        try:
            return self._ingress.get(timeout=timeout)
        except queue.Empty:
            return None
