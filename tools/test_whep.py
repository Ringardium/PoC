"""Standalone WHEP pull tester for verifying GPU <- SRS WebRTC connectivity.

Usage:
    pip install aiortc aiohttp
    python tools/test_whep.py \\
        --url http://<srs-host>:1985/rtc/v1/whep \\
        --app live --stream camera_1 --duration 10

Reports:
    - Time-to-first-frame (TTFF)
    - Resolution, codec
    - Average FPS over the test window
    - First frame pts / time_base (for timestamp sync design)

Fallback mode --mode srs uses SRS's proprietary JSON API (/rtc/v1/play/)
when WHEP endpoint is not enabled on your SRS build.
"""
import argparse
import asyncio
import json
import time

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription


async def pull_whep(base_url: str, app: str, stream: str, vhost: str, mode: str):
    """Build SDP offer, POST to SRS, return (pc, remote_sdp)."""
    pc = RTCPeerConnection()
    pc.addTransceiver("video", direction="recvonly")
    pc.addTransceiver("audio", direction="recvonly")

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    qs = f"?app={app}&stream={stream}"
    if vhost:
        qs += f"&vhost={vhost}"

    async with aiohttp.ClientSession() as session:
        if mode == "whep":
            url = f"{base_url.rstrip('/')}/{qs}"
            async with session.post(
                url,
                data=pc.localDescription.sdp,
                headers={"Content-Type": "application/sdp"},
            ) as resp:
                body = await resp.text()
                if resp.status not in (200, 201):
                    raise RuntimeError(f"WHEP failed: status={resp.status}\n{body}")
                answer_sdp = body
        else:  # srs JSON mode
            url = f"{base_url.rstrip('/')}/rtc/v1/play/"
            host = base_url.split("//")[1].split(":")[0]
            stream_host = vhost if vhost else host
            payload = {
                "api": url,
                "streamurl": f"webrtc://{stream_host}/{app}/{stream}",
                "sdp": pc.localDescription.sdp,
            }
            async with session.post(url, json=payload) as resp:
                body = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(f"SRS play failed: status={resp.status}\n{body}")
                data = json.loads(body)
                if data.get("code") != 0:
                    raise RuntimeError(f"SRS play error: {data}")
                answer_sdp = data["sdp"]

    return pc, answer_sdp


async def run_test(base_url: str, app: str, stream: str, vhost: str, duration: float, mode: str):
    start = time.time()
    try:
        pc, answer_sdp = await pull_whep(base_url, app, stream, vhost, mode)
    except Exception as e:
        print(f"[signaling] FAILED: {e}")
        return

    print(f"[signaling] OK ({(time.time() - start) * 1000:.0f}ms)")

    frame_count = 0
    first_frame_time = None
    first_pts = None
    first_time_base = None
    first_resolution = None

    @pc.on("track")
    async def on_track(track):
        nonlocal frame_count, first_frame_time, first_pts, first_time_base, first_resolution
        print(f"[track] received: kind={track.kind}")
        if track.kind != "video":
            return
        while True:
            try:
                frame = await track.recv()
            except Exception as e:
                print(f"[track] stream ended: {e}")
                return
            if first_frame_time is None:
                first_frame_time = time.time()
                first_pts = frame.pts
                first_time_base = frame.time_base
                first_resolution = (frame.width, frame.height)
                ttff_ms = (first_frame_time - start) * 1000
                print(
                    f"[track] first frame — {frame.width}x{frame.height}, "
                    f"pts={frame.pts}, time_base={frame.time_base}, ttff={ttff_ms:.0f}ms"
                )
            frame_count += 1

    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type="answer"))
    print(f"[pull] streaming for {duration}s...")

    await asyncio.sleep(duration)

    elapsed = time.time() - start
    fps = frame_count / (elapsed - (first_frame_time - start)) if first_frame_time else 0

    print("=" * 60)
    print(f"[result] frames received: {frame_count}")
    print(f"[result] effective fps:   {fps:.2f}")
    print(f"[result] resolution:      {first_resolution}")
    print(f"[result] first_pts:       {first_pts} (unit={first_time_base})")
    if first_pts and first_time_base:
        print(f"[result] first_pts_sec:   {float(first_pts * first_time_base):.3f}s")
    print("=" * 60)

    await pc.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True,
                    help="SRS base URL, e.g. http://srs-host:1985 (path auto-appended)")
    ap.add_argument("--app", default="live")
    ap.add_argument("--stream", required=True)
    ap.add_argument("--vhost", default="",
                    help="SRS vhost (required when SRS routes by vhost, e.g. SRT tcUrl srt://kr/live -> vhost=kr)")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--mode", choices=["whep", "srs"], default="whep",
                    help="whep = standard WHEP POST (SDP body); "
                         "srs = SRS proprietary /rtc/v1/play/ (JSON body)")
    args = ap.parse_args()

    base = args.url.rstrip("/")
    if args.mode == "whep" and not base.endswith("/rtc/v1/whep"):
        base = f"{base}/rtc/v1/whep"

    asyncio.run(run_test(base, args.app, args.stream, args.vhost, args.duration, args.mode))


if __name__ == "__main__":
    main()
