#!/usr/bin/env python3
"""
Web Viewer for Multi-Stream Pet Tracking System

Provides real-time browser-based monitoring of all streams via WebSocket.

Usage:
    python web_server.py --config config.json --port 8000
    # Then open http://localhost:8000 in browser
"""

import asyncio
import json
import logging
import platform
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import click
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse

from config import SystemConfig, StreamConfig
from stream_processor import MultiStreamProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pet Tracker - Multi-Stream Monitor")

# Global processor reference (set during startup)
processor: Optional[MultiStreamProcessor] = None


# === API Endpoints ===

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main viewer page"""
    template_path = Path(__file__).parent / "templates" / "index.html"
    if template_path.exists():
        return HTMLResponse(content=template_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Template not found</h1>", status_code=404)


@app.get("/api/stats")
async def get_stats():
    """Get system and stream statistics"""
    if processor is None:
        return JSONResponse({"error": "Processor not initialized"}, status_code=503)
    return JSONResponse(processor.get_stats())


@app.get("/api/streams")
async def get_streams():
    """Get list of active streams"""
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
    """Add a new stream at runtime"""
    if processor is None:
        return JSONResponse({"error": "Processor not initialized"}, status_code=503)

    body = await request.json()

    # Validate required fields
    stream_id = body.get("stream_id", "").strip()
    input_source = "".join(body.get("input_source", "").split())  # remove all whitespace
    if not stream_id or not input_source:
        return JSONResponse({"error": "stream_id and input_source are required"}, status_code=400)

    if stream_id in processor.streams:
        return JSONResponse({"error": f"Stream '{stream_id}' already exists"}, status_code=409)

    # Build StreamConfig with defaults
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
    """Remove a stream at runtime"""
    if processor is None:
        return JSONResponse({"error": "Processor not initialized"}, status_code=503)

    if stream_id not in processor.streams:
        return JSONResponse({"error": f"Stream '{stream_id}' not found"}, status_code=404)

    processor.remove_stream_dynamic(stream_id)
    logger.info(f"Stream removed via API: {stream_id}")
    return JSONResponse({"ok": True, "stream_id": stream_id})


@app.websocket("/ws/frames/{stream_id}")
async def ws_stream_frames(websocket: WebSocket, stream_id: str):
    """WebSocket: stream JPEG frames for a specific stream"""
    await websocket.accept()
    logger.info(f"WebSocket connected for stream: {stream_id}")

    try:
        last_frame = None
        while True:
            if processor is None or not processor.running:
                await asyncio.sleep(0.5)
                continue

            with processor.web_frames_lock:
                frame_data = processor.web_frames.get(stream_id)

            if frame_data is not None and frame_data is not last_frame:
                await websocket.send_bytes(frame_data)
                last_frame = frame_data

            # Target ~15fps for web streaming (save bandwidth)
            await asyncio.sleep(0.066)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for stream: {stream_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {stream_id}: {e}")


@app.websocket("/ws/frames/all")
async def ws_all_frames(websocket: WebSocket):
    """WebSocket: stream all frames as tagged binary"""
    await websocket.accept()
    logger.info("WebSocket connected for all streams")

    try:
        last_frames = {}
        while True:
            if processor is None or not processor.running:
                await asyncio.sleep(0.5)
                continue

            with processor.web_frames_lock:
                current_frames = dict(processor.web_frames)

            # Send each changed frame with stream_id prefix
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
def main(config: str, port: int, host: str):
    """Start the web viewer with multi-stream processing"""
    global processor

    logger.info(f"Loading configuration from {config}")
    sys_config = SystemConfig.load(config)

    if not sys_config.streams:
        logger.error("No streams configured")
        sys.exit(1)

    # Create processor
    processor = MultiStreamProcessor(sys_config)

    logger.info(f"Starting {len(sys_config.streams)} streams...")
    logger.info(f"Model: {sys_config.model_path}")
    logger.info(f"Web viewer: http://localhost:{port}")

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
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        processor.stop()
        processor_thread.join(timeout=5)
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
