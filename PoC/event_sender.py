"""Sleep event tracking and backend API sender."""

import logging
import math
import queue
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# Map internal behavior names to backend API spec
BEHAVIOR_TYPE_MAP = {
    "sleep": "sleeping",
    "inert": "inactive",
    "eat": "feeding",
    "active": "playing",
    "fight": "fight",
    "escape": "escape",
    "bathroom": "bathroom",
}


class BehaviorStateTracker:
    """Track per-dog behavior state transitions (start/end) with duration.

    Each frame, call ``update()`` with the current list of detected track IDs.
    When a dog stops the behavior, the tracker yields a completed event with duration.
    """

    def __init__(self, behavior: str, min_duration_min: float = 0.0):
        self._behavior = behavior
        self._min_duration_min = min_duration_min
        # track_id -> {"start_time": float, "start_frame": int}
        self._active: Dict[int, dict] = {}

    def update(self, detected_ids: List[int], frame_cnt: int, fps: int,
            current_time: Optional[float] = None) -> List[dict]:
        """Compare current detected_ids against active states.

        Args:
            detected_ids: list of currently detected track/behavior IDs.
            frame_cnt: current frame number.
            fps: target FPS (unused, kept for compat).
            current_time: optional SyncClock-aligned timestamp; falls back to
                ``time.time()`` when *None*.

        Returns list of completed events (dogs that stopped the behavior).
        """
        current_set = set(detected_ids)
        ended_events: List[dict] = []
        now = current_time if current_time is not None else time.time()

        # Detect newly started dogs
        for tid in current_set:
            if tid not in self._active:
                self._active[tid] = {
                    "start_time": now,
                    "start_frame": frame_cnt,
                }

        # Detect dogs that stopped
        finished = [tid for tid in self._active if tid not in current_set]

        for tid in finished:
            state = self._active.pop(tid)
            duration_sec = int(now - state["start_time"])

            if duration_sec <= 0:
                continue

            KST = timezone(timedelta(hours=9))
            now_kst = datetime.now(KST)
            ended_events.append({
                "dogId": tid,
                "behaviorType": BEHAVIOR_TYPE_MAP[self._behavior],
                "durationSeconds": duration_sec,
                "detectedAt": now_kst.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            logger.info(
                f"{self._behavior} ended: dog {tid}, duration {duration_sec:.1f}s "
                f"(frames {state['start_frame']}-{frame_cnt})"
            )

        return ended_events


# Backward compatibility
SleepStateTracker = BehaviorStateTracker


class EventSender:
    """Non-blocking HTTP POST sender with background thread.

    Events are queued and sent in a daemon thread so the main
    processing loop is never blocked by network I/O.
    """

    def __init__(self, api_url: str, timeout: float = 5.0):
        self._api_url = api_url
        self._timeout = timeout
        self._queue: queue.Queue = queue.Queue(maxsize=1000)
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        logger.info(f"EventSender started — target: {api_url}")

    def send(self, event: dict):
        """Queue an event for async delivery."""
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            logger.warning("Event queue full, dropping event")

    def _worker(self):
        while self._running:
            try:
                event = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                resp = requests.post(
                    self._api_url,
                    json=event,
                    timeout=self._timeout,
                )
                if resp.status_code >= 400:
                    logger.warning(f"Event API returned {resp.status_code}: {resp.text[:200]}")
                else:
                    logger.debug(f"Event sent: {event}")
            except requests.RequestException as e:
                logger.warning(f"Event send failed: {e}")

    def stop(self):
        """Drain remaining events and stop the worker thread."""
        self._running = False
        self._thread.join(timeout=5.0)


def send_event_cli():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Send behavior event(s) to API")
    sub = parser.add_subparsers(dest="command")

    # --- single event ---
    p_single = sub.add_parser("send", help="Send a single event")
    p_single.add_argument("--url", required=True, help="API endpoint URL")
    p_single.add_argument("--dog-id", required=True, help="Dog ID (name or number)")
    p_single.add_argument(
        "--behavior", required=True,
        choices=list(BEHAVIOR_TYPE_MAP.values()),
        help="Behavior type",
    )
    p_single.add_argument("--duration", type=float, default=None, help="Duration in seconds")
    p_single.add_argument("--clip-url", default=None, help="Clip video URL (e.g. CDN or S3 presigned)")

    # --- batch from JSON ---
    p_batch = sub.add_parser("batch", help="Send events from a JSON file")
    p_batch.add_argument("--url", required=True, help="API endpoint URL")
    p_batch.add_argument("--file", required=True, help="JSON file with events list")

    args = parser.parse_args()

    if args.command == "send" or args.command is None:
        # Backward compat: if no subcommand, treat as single send
        if args.command is None:
            # Re-parse with old-style args
            parser2 = argparse.ArgumentParser()
            parser2.add_argument("--url", required=True)
            parser2.add_argument("--dog-id", required=True)
            parser2.add_argument("--behavior", required=True, choices=list(BEHAVIOR_TYPE_MAP.values()))
            parser2.add_argument("--duration", type=float, default=None)
            parser2.add_argument("--clip-url", default=None)
            args = parser2.parse_args()

        KST = timezone(timedelta(hours=9))
        now_kst = datetime.now(KST)
        event = {
            "dogId": args.dog_id,
            "behaviorType": args.behavior,
            "detectedAt": now_kst.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if args.duration is not None:
            event["durationSeconds"] = args.duration
        if args.clip_url is not None:
            event["clipUrl"] = args.clip_url

        resp = requests.post(args.url, json=event, timeout=5.0)
        print(f"[{resp.status_code}] {resp.text}")

    elif args.command == "batch":
        with open(args.file, encoding="utf-8") as f:
            data = json.load(f)

        events = data if isinstance(data, list) else data.get("events", [])
        print(f"Sending {len(events)} events to {args.url}")

        ok, fail = 0, 0
        for i, event in enumerate(events):
            try:
                resp = requests.post(args.url, json=event, timeout=5.0)
                status = resp.status_code
                if status < 400:
                    ok += 1
                    print(f"  [{i+1}/{len(events)}] OK ({status})")
                else:
                    fail += 1
                    print(f"  [{i+1}/{len(events)}] FAIL ({status}) {resp.text[:100]}")
            except requests.RequestException as e:
                fail += 1
                print(f"  [{i+1}/{len(events)}] ERROR: {e}")

        print(f"\nDone: {ok} sent, {fail} failed")


if __name__ == "__main__":
    send_event_cli()
