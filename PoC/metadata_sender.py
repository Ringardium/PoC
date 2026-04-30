"""WebSocket broadcaster for per-frame detection metadata.

Runs a small aiohttp WebSocket server in a background thread so mobile
clients can subscribe to bbox / track_id / global_id / behavior streams
and render overlays client-side without the server burning CPU on drawing.

Server → client wire format (JSON text frame):
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

Client → server messages (JSON text frame):
    {"type": "subscribe", "stream_ids": ["s1", "s2"]}
        — only forward frames whose stream_id is in this set.
          stream_ids=[] or omitted = subscribe to ALL streams (default).
    {"type": "unsubscribe", "stream_ids": ["s1"]}
        — drop these stream_ids from the subscription.
    {"type": "unsubscribe_all"}
        — clear filter; receive all streams again.
    {"type": "request_snapshot"}
        — server replies with the most recent frame_metadata for each
          currently-subscribed stream (or all known streams if no filter).
    {"type": "ping"}
        — server replies with {"type":"pong","ts":<server_unix_seconds>}.

Server → client control messages:
    {"type": "hello", "streams": [...], "ts": ...}     — sent on connect.
    {"type": "ack", "action": "...", "stream_ids": [...]} — subscribe/unsubscribe ack.
    {"type": "snapshot", "stream_id": "...", "payload": {...}} — snapshot reply.
    {"type": "pong", "ts": ...}                         — ping reply.
    {"type": "error", "message": "..."}                 — malformed request.

Usage:
    sender = MetadataSender(port=8766)
    sender.start()
    ...
    sender.push_frame(stream_id, ts, tracks)            # legacy
    sender.push(payload)                                # full payload (preferred)
    ...
    sender.stop()
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class MetadataSender:
    """Thread-safe WebSocket broadcaster with per-client stream filtering.

    Each connected client may opt in to a subset of stream_ids via the
    ``subscribe`` / ``unsubscribe`` control messages. Without any
    subscription the client receives every stream (legacy behavior).
    """

    def __init__(self, port: int = 8766, host: str = "0.0.0.0", path: str = "/ws/metadata",
                 queue_size: int = 512):
        self._port = port
        self._host = host
        self._path = path
        self._ingress: "queue.Queue[dict]" = queue.Queue(maxsize=queue_size)

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        # Map: WebSocketResponse -> Optional[Set[str]]; None = receive ALL.
        self._clients: "Dict[object, Optional[Set[str]]]" = {}
        # Most recent frame_metadata payload per stream_id (used for snapshots
        # and to expose the live stream set in the hello message).
        self._latest: Dict[str, dict] = {}
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
            for ws in list(self._clients.keys()):
                try:
                    await ws.close()
                except Exception:
                    pass
            self._clients.clear()
            await self._runner.cleanup()

    async def _health_handler(self, request):
        from aiohttp import web
        return web.json_response({
            "clients": len(self._clients),
            "queued": self._ingress.qsize(),
            "streams": list(self._latest.keys()),
        })

    async def _ws_handler(self, request):
        from aiohttp import web

        ws = web.WebSocketResponse(heartbeat=15)
        await ws.prepare(request)
        # None subscription = receive every stream (legacy default).
        self._clients[ws] = None
        peer = request.remote
        logger.info(f"metadata client connected: {peer} (total={len(self._clients)})")

        # Hello — let the client know which stream_ids are currently producing
        # frames so it can subscribe selectively without guessing.
        try:
            await ws.send_str(json.dumps({
                "type": "hello",
                "streams": list(self._latest.keys()),
                "ts": time.time(),
            }, ensure_ascii=False, separators=(",", ":")))
        except Exception:
            pass

        try:
            from aiohttp import WSMsgType
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    await self._handle_client_message(ws, msg.data)
                elif msg.type == WSMsgType.ERROR:
                    logger.warning(f"metadata client {peer} ws error: {ws.exception()}")
                    break
        finally:
            self._clients.pop(ws, None)
            logger.info(f"metadata client disconnected: {peer} (total={len(self._clients)})")
        return ws

    async def _handle_client_message(self, ws, raw: str):
        """Parse a client control message and update its subscription state."""
        try:
            msg = json.loads(raw)
        except Exception:
            await self._safe_send(ws, {"type": "error", "message": "invalid_json"})
            return
        if not isinstance(msg, dict):
            await self._safe_send(ws, {"type": "error", "message": "expected_object"})
            return

        mtype = msg.get("type")
        if mtype == "subscribe":
            ids = msg.get("stream_ids") or []
            if not isinstance(ids, list):
                await self._safe_send(ws, {"type": "error", "message": "stream_ids_must_be_list"})
                return
            ids = [str(s) for s in ids]
            if not ids:
                # Empty list = subscribe to all (clear filter).
                self._clients[ws] = None
            else:
                cur = self._clients.get(ws)
                if cur is None:
                    self._clients[ws] = set(ids)
                else:
                    cur.update(ids)
            await self._safe_send(ws, {
                "type": "ack", "action": "subscribe",
                "stream_ids": sorted(self._clients[ws]) if self._clients[ws] else [],
            })

        elif mtype == "unsubscribe":
            ids = msg.get("stream_ids") or []
            if not isinstance(ids, list):
                await self._safe_send(ws, {"type": "error", "message": "stream_ids_must_be_list"})
                return
            cur = self._clients.get(ws)
            if cur is not None:
                for sid in ids:
                    cur.discard(str(sid))
                if not cur:
                    # Drop the empty filter — but interpret as "no streams"
                    # (i.e. the client explicitly wants nothing). Use empty set.
                    self._clients[ws] = set()
            await self._safe_send(ws, {
                "type": "ack", "action": "unsubscribe",
                "stream_ids": sorted(self._clients[ws]) if self._clients[ws] else [],
            })

        elif mtype == "unsubscribe_all":
            self._clients[ws] = None
            await self._safe_send(ws, {"type": "ack", "action": "unsubscribe_all"})

        elif mtype == "request_snapshot":
            subs = self._clients.get(ws)
            target_ids = list(self._latest.keys()) if subs is None else list(subs)
            for sid in target_ids:
                payload = self._latest.get(sid)
                await self._safe_send(ws, {
                    "type": "snapshot",
                    "stream_id": sid,
                    "payload": payload,  # may be None if never produced yet
                })

        elif mtype == "ping":
            await self._safe_send(ws, {"type": "pong", "ts": time.time()})

        else:
            await self._safe_send(ws, {"type": "error", "message": f"unknown_type:{mtype}"})

    async def _safe_send(self, ws, message: dict):
        try:
            if ws.closed:
                return
            await ws.send_str(json.dumps(message, ensure_ascii=False, separators=(",", ":")))
        except Exception:
            pass

    async def _drain_loop(self):
        """Pump messages from the thread-safe ingress queue into matching WS clients."""
        loop = asyncio.get_event_loop()
        while self._running.is_set():
            message = await loop.run_in_executor(None, self._blocking_get, 0.2)
            if message is None:
                continue

            # Cache latest per-stream payload for snapshots / hello.
            sid = message.get("stream_id")
            if sid and message.get("type") == "frame_metadata":
                self._latest[sid] = message

            if not self._clients:
                continue

            payload = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
            dead = []
            for client, subs in list(self._clients.items()):
                # Filter: None = all streams; empty set = none; otherwise check membership.
                if subs is not None:
                    if not subs or (sid is not None and sid not in subs):
                        continue
                try:
                    if client.closed:
                        dead.append(client)
                        continue
                    await client.send_str(payload)
                except Exception:
                    dead.append(client)
            for d in dead:
                self._clients.pop(d, None)

    def _blocking_get(self, timeout: float):
        try:
            return self._ingress.get(timeout=timeout)
        except queue.Empty:
            return None
