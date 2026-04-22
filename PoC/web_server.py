#!/usr/bin/env python3
"""
Web Viewer for Multi-Stream Pet Tracking System

Provides real-time browser-based monitoring of all streams via WebSocket.

Usage:
    python web_server.py --config config.json --port 8000
    python web_server.py --config config.json --no-web   # headless, processor only
    # Then open http://localhost:8000 in browser
"""

import asyncio
import logging
import sys
import threading
from pathlib import Path
from typing import Optional

import click

# Install uvloop if available (Linux, 2-3x faster than default asyncio selector).
# Safe no-op on macOS / Windows or when uvloop is not installed.
_UVLOOP_ACTIVE = False
try:
    import uvloop
    uvloop.install()
    _UVLOOP_ACTIVE = True
except ImportError:
    pass

from config import SystemConfig, StreamConfig
from stream_processor import MultiStreamProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Conditional web imports — FastAPI / uvicorn are only needed when web is enabled
_HAS_WEB = False
try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    _HAS_WEB = True
except ImportError:
    pass


def create_app(proc_ref: dict) -> "FastAPI":
    """Factory that builds the FastAPI application.

    ``proc_ref`` is a mutable dict ``{"processor": MultiStreamProcessor | None}``
    so the app can reference the processor without global state.
    """
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="Pet Tracker - Multi-Stream Monitor")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        template_path = Path(__file__).parent / "templates" / "index.html"
        if template_path.exists():
            return HTMLResponse(content=template_path.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>Template not found</h1>", status_code=404)

    @app.get("/api/stats")
    async def get_stats():
        processor = proc_ref.get("processor")
        if processor is None:
            return JSONResponse({"error": "Processor not initialized"}, status_code=503)
        return JSONResponse(processor.get_stats())

    @app.get("/api/streams")
    async def get_streams():
        processor = proc_ref.get("processor")
        if processor is None:
            return JSONResponse({"error": "Processor not initialized"}, status_code=503)
        streams = []
        for sid, config in processor.streams.items():
            streams.append({
                "stream_id": sid,
                "input_source": config.input_source,
                "method": config.method,
                "target_fps": config.target_fps,
                "use_reid": config.use_reid,
                "reid_global_id": config.reid_global_id,
                "tasks": {
                    "fight": config.task_fight,
                    "escape": config.task_escape,
                    "inert": config.task_inert,
                }
            })
        return JSONResponse({"streams": streams})

    @app.post("/api/streams")
    async def add_stream(request: Request):
        processor = proc_ref.get("processor")
        if processor is None:
            return JSONResponse({"error": "Processor not initialized"}, status_code=503)
        body = await request.json()
        stream_id = body.get("stream_id", "").strip()
        input_source = "".join(body.get("input_source", "").split())
        if not stream_id or not input_source:
            return JSONResponse({"error": "stream_id and input_source are required"}, status_code=400)
        if stream_id in processor.streams:
            return JSONResponse({"error": f"Stream '{stream_id}' already exists"}, status_code=409)
        stream_config = StreamConfig(
            stream_id=stream_id,
            input_source=input_source,
            output_path=body.get("output_path"),
            method=body.get("method", "bytetrack"),
            task_fight=body.get("task_fight", True),
            task_escape=body.get("task_escape", False),
            task_inert=body.get("task_inert", True),
            target_fps=body.get("target_fps", 30),
            use_reid=body.get("use_reid", True),
            reid_method=body.get("reid_method", "adaptive"),
            reid_threshold=body.get("reid_threshold", 0.5),
            reid_global_id=body.get("reid_global_id", False),
        )
        success = processor.add_stream_dynamic(stream_config)
        if not success:
            return JSONResponse({"error": "Max streams reached"}, status_code=400)
        logger.info(f"Stream added via API: {stream_id}")
        return JSONResponse({"ok": True, "stream_id": stream_id})

    @app.delete("/api/streams/{stream_id}")
    async def delete_stream(stream_id: str):
        processor = proc_ref.get("processor")
        if processor is None:
            return JSONResponse({"error": "Processor not initialized"}, status_code=503)
        if stream_id not in processor.streams:
            return JSONResponse({"error": f"Stream '{stream_id}' not found"}, status_code=404)
        processor.remove_stream_dynamic(stream_id)
        logger.info(f"Stream removed via API: {stream_id}")
        return JSONResponse({"ok": True, "stream_id": stream_id})

    @app.websocket("/ws/frames/{stream_id}")
    async def ws_stream_frames(websocket: WebSocket, stream_id: str):
        await websocket.accept()
        logger.info(f"WebSocket connected for stream: {stream_id}")
        try:
            last_frame = None
            while True:
                processor = proc_ref.get("processor")
                if processor is None or not processor.running:
                    await asyncio.sleep(0.5)
                    continue
                with processor.web_frames_lock:
                    frame_data = processor.web_frames.get(stream_id)
                if frame_data is not None and frame_data is not last_frame:
                    await websocket.send_bytes(frame_data)
                    last_frame = frame_data
                await asyncio.sleep(0.066)
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for stream: {stream_id}")
        except Exception as e:
            logger.error(f"WebSocket error for {stream_id}: {e}")

    @app.websocket("/ws/frames/all")
    async def ws_all_frames(websocket: WebSocket):
        await websocket.accept()
        logger.info("WebSocket connected for all streams")
        try:
            last_frames = {}
            while True:
                processor = proc_ref.get("processor")
                if processor is None or not processor.running:
                    await asyncio.sleep(0.5)
                    continue
                with processor.web_frames_lock:
                    current_frames = dict(processor.web_frames)
                for sid, frame_data in current_frames.items():
                    if frame_data is not last_frames.get(sid):
                        header = sid.encode('utf-8') + b'\x00'
                        await websocket.send_bytes(header + frame_data)
                        last_frames[sid] = frame_data
                await asyncio.sleep(0.066)
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected for all streams")
        except Exception as e:
            logger.error(f"WebSocket error (all): {e}")

    return app


def run_processor_in_thread(proc: MultiStreamProcessor):
    """Run the processor's asyncio loop in a separate thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(proc.start())
    except Exception as e:
        logger.error(f"Processor error: {e}")
    finally:
        loop.close()


# === CLI ===

@click.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="JSON configuration file path")
@click.option("--port", "-p", default=8000, type=int, help="Web server port")
@click.option("--host", default="0.0.0.0", help="Web server host")
@click.option("--no-web", is_flag=True, default=False,
              help="Run processor only without web viewer (headless mode)")
def main(config: str, port: int, host: str, no_web: bool):
    """Start the web viewer with multi-stream processing"""

    logger.info(f"Loading configuration from {config}")
    if _UVLOOP_ACTIVE:
        logger.info("uvloop enabled")
    sys_config = SystemConfig.load(config)

    if not sys_config.streams:
        logger.error("No streams configured")
        sys.exit(1)

    web_enabled = not no_web

    if web_enabled and not _HAS_WEB:
        logger.error("Web mode requires fastapi and uvicorn. Install them or use --no-web.")
        sys.exit(1)

    # Create processor
    processor = MultiStreamProcessor(sys_config, web_enabled=web_enabled)

    logger.info(f"Starting {len(sys_config.streams)} streams...")
    logger.info(f"Model: {sys_config.model_path}")

    if web_enabled:
        logger.info(f"Web viewer: http://localhost:{port}")

        proc_ref = {"processor": processor}
        app = create_app(proc_ref)

        # Start processor in a separate thread (its own asyncio loop)
        processor_thread = threading.Thread(
            target=run_processor_in_thread,
            args=(processor,),
            daemon=True
        )
        processor_thread.start()
        logger.info("Processor thread started")

        # Run uvicorn on the main thread
        try:
            uvicorn.run(
                app,
                host=host,
                port=port,
                log_level="info",
                loop="uvloop" if _UVLOOP_ACTIVE else "auto",
            )
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            processor.stop()
            processor_thread.join(timeout=5)
            logger.info("Shutdown complete")
    else:
        # Headless mode — run processor directly
        logger.info("Headless mode (no web viewer)")
        try:
            asyncio.run(processor.start())
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            processor.stop()
            logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
