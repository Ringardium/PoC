import asyncio
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from detect_escape import detect_escape
from detect_fight import detect_fight
from detect_inert import detect_inert
from tracking import track_with_botsort, track_with_bytetrack, track_with_deepsort


@dataclass
class StreamConfig:
    stream_id: str
    input_source: str  # video file path or RTSP URL or webcam index
    output_path: Optional[str] = None
    method: str = "bytetrack"
    task_fight: bool = False
    task_escape: bool = False
    task_inert: bool = False
    threshold: float = 0.1
    inert_threshold: float = 50
    inert_frames: int = 100
    reset_frames: int = 20
    flag_frames: int = 40
    max_number: int = 500
    escape_polygon: Optional[List[Tuple[int, int]]] = None


@dataclass
class StreamState:
    inert_coor: Dict = None
    close_count: torch.Tensor = None
    far_count: torch.Tensor = None
    frame_cnt: int = 0
    tracker: Optional[Tracker] = None

    def __post_init__(self):
        if self.inert_coor is None:
            self.inert_coor = {}
        if self.close_count is None:
            self.close_count = torch.zeros((500, 500))
        if self.far_count is None:
            self.far_count = torch.zeros((500, 500))


class FrameBuffer:
    def __init__(self, maxsize: int = 30):
        self.buffer = deque(maxlen=maxsize)
        self.lock = threading.Lock()

    def put(self, frame: np.ndarray):
        with self.lock:
            self.buffer.append(frame)

    def get_all(self) -> List[np.ndarray]:
        with self.lock:
            frames = list(self.buffer)
            self.buffer.clear()
            return frames


class MultiStreamProcessor:
    def __init__(self, model_path: str = "weights/best.pt", max_workers: int = 4):
        self.model = YOLO(model_path)
        self.max_workers = max_workers
        self.streams: Dict[str, StreamConfig] = {}
        self.stream_states: Dict[str, StreamState] = {}
        self.frame_buffers: Dict[str, FrameBuffer] = {}
        self.output_queues: Dict[str, Queue] = {}
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Shared model for batch inference
        self.inference_queue = Queue(maxsize=100)
        self.result_queues: Dict[str, Queue] = {}

    def add_stream(self, config: StreamConfig):
        """Add a new stream to process"""
        self.streams[config.stream_id] = config
        self.stream_states[config.stream_id] = StreamState()
        self.frame_buffers[config.stream_id] = FrameBuffer()
        self.output_queues[config.stream_id] = Queue(maxsize=100)
        self.result_queues[config.stream_id] = Queue(maxsize=100)

        # Initialize DeepSORT tracker if needed
        if config.method == "deepsort":
            max_cosine_distance = 0.5
            nn_budget = None
            metric = nn_matching.NearestNeighborDistanceMetric(
                "cosine", max_cosine_distance, nn_budget
            )
            self.stream_states[config.stream_id].tracker = Tracker(metric)

    def remove_stream(self, stream_id: str):
        """Remove a stream from processing"""
        if stream_id in self.streams:
            del self.streams[stream_id]
            del self.stream_states[stream_id]
            del self.frame_buffers[stream_id]
            del self.output_queues[stream_id]
            del self.result_queues[stream_id]

    async def start_processing(self):
        """Start processing all streams"""
        self.running = True

        # Start inference worker
        inference_task = asyncio.create_task(self._batch_inference_worker())

        # Start stream processors
        stream_tasks = []
        for stream_id in self.streams:
            task = asyncio.create_task(self._process_stream(stream_id))
            stream_tasks.append(task)

        # Start output writers
        output_tasks = []
        for stream_id in self.streams:
            if self.streams[stream_id].output_path:
                task = asyncio.create_task(self._write_output(stream_id))
                output_tasks.append(task)

        try:
            await asyncio.gather(inference_task, *stream_tasks, *output_tasks)
        except Exception as e:
            print(f"Error in processing: {e}")
        finally:
            self.running = False

    async def _process_stream(self, stream_id: str):
        """Process a single video stream"""
        config = self.streams[stream_id]
        state = self.stream_states[stream_id]

        # Open video capture
        if config.input_source.isdigit():
            cap = cv2.VideoCapture(int(config.input_source))
        else:
            cap = cv2.VideoCapture(config.input_source)

        if not cap.isOpened():
            print(f"Error opening stream {stream_id}: {config.input_source}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))

        try:
            while self.running and cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                state.frame_cnt += 1

                # Add frame to inference queue
                self.inference_queue.put((stream_id, frame.copy()))

                # Get inference results
                try:
                    boxes, track_ids, processed_frame = self.result_queues[stream_id].get(timeout=0.1)

                    # Process behaviors
                    processed_frame = await self._process_behaviors(
                        stream_id, boxes, track_ids, processed_frame, state, config
                    )

                    # Add to output queue
                    if config.output_path:
                        self.output_queues[stream_id].put(processed_frame)

                except:
                    continue

                # Control frame rate
                await asyncio.sleep(1.0 / max(fps, 30))

        finally:
            cap.release()

    async def _batch_inference_worker(self):
        """Worker for batch model inference"""
        batch_frames = {}
        batch_size = min(len(self.streams), 4)

        while self.running:
            # Collect frames for batch processing
            for _ in range(batch_size):
                try:
                    stream_id, frame = self.inference_queue.get(timeout=0.05)
                    batch_frames[stream_id] = frame
                except:
                    break

            if not batch_frames:
                await asyncio.sleep(0.01)
                continue

            # Process batch
            await self._process_batch(batch_frames)
            batch_frames.clear()

    async def _process_batch(self, batch_frames: Dict[str, np.ndarray]):
        """Process a batch of frames from multiple streams"""
        loop = asyncio.get_event_loop()

        # Run inference in thread pool to avoid blocking
        tasks = []
        for stream_id, frame in batch_frames.items():
            config = self.streams[stream_id]
            state = self.stream_states[stream_id]

            task = loop.run_in_executor(
                self.executor,
                self._run_tracking,
                config.method,
                frame,
                state.tracker
            )
            tasks.append((stream_id, task))

        # Wait for all inference results
        for stream_id, task in tasks:
            try:
                boxes, track_ids, processed_frame = await task
                self.result_queues[stream_id].put((boxes, track_ids, processed_frame))
            except Exception as e:
                print(f"Error processing stream {stream_id}: {e}")

    def _run_tracking(self, method: str, frame: np.ndarray, tracker: Optional[Tracker]):
        """Run tracking algorithm on a single frame"""
        if method == "bytetrack":
            return track_with_bytetrack(self.model, frame)
        elif method == "botsort":
            return track_with_botsort(self.model, frame)
        elif method == "deepsort":
            return track_with_deepsort(self.model, tracker, frame)
        else:
            raise ValueError(f"Unknown tracking method: {method}")

    async def _process_behaviors(
        self,
        stream_id: str,
        boxes: List,
        track_ids: List,
        frame: np.ndarray,
        state: StreamState,
        config: StreamConfig
    ) -> np.ndarray:
        """Process behavior detection for a stream"""
        if len(boxes) == 0:
            return frame

        # Update inert coordinates
        for id in track_ids:
            if id not in state.inert_coor:
                state.inert_coor[id] = deque([], maxlen=config.inert_frames)

        id_to_index = {id: index for index, id in enumerate(track_ids)}

        # Extract centers
        x_centers, y_centers = [], []
        for id, box in zip(track_ids, boxes):
            x_center, y_center, width, height = box
            x_centers.append(x_center)
            y_centers.append(y_center)
            state.inert_coor[id].append([x_center, y_center])

            # Draw center point
            cv2.circle(frame, (int(x_center), int(y_center)), 5, (0, 0, 255), 3)

        x_centers = np.array(x_centers)
        y_centers = np.array(y_centers)
        status_text = " "

        # Detect fight
        if config.task_fight:
            fight_indices = detect_fight(
                x_centers, y_centers, track_ids, state.close_count, state.far_count,
                config.threshold, config.reset_frames, config.flag_frames,
                int(width), int(height)
            )

            for ids in fight_indices:
                for i in ids:
                    if i.item() in id_to_index:
                        x_center, y_center, width, height = boxes[id_to_index[i.item()]]
                        pt1 = (int(x_center - width/2), int(y_center - height/2))
                        pt2 = (int(x_center + width/2), int(y_center + height/2))
                        cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)

            if len(fight_indices) > 0:
                status_text += "Fight "

        # Detect escape
        if config.task_escape and config.escape_polygon:
            w, h = frame.shape[1], frame.shape[0]
            frame, escaped_ids = detect_escape(
                boxes, track_ids, frame, state.frame_cnt, config.escape_polygon, w, h
            )

            for id in escaped_ids:
                if id in id_to_index:
                    x_center, y_center, width, height = boxes[id_to_index[id]]
                    pt1 = (int(x_center - width/2), int(y_center - height/2))
                    pt2 = (int(x_center + width/2), int(y_center + height/2))
                    cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 2)

            if len(escaped_ids) > 0:
                status_text += "Escape "

        # Detect inert
        if config.task_inert:
            inert_indices = detect_inert(
                state.inert_coor, config.inert_threshold, config.inert_frames
            )

            for id in inert_indices:
                if id in id_to_index:
                    x_center, y_center, width, height = boxes[id_to_index[id]]
                    pt1 = (int(x_center - width/2), int(y_center - height/2))
                    pt2 = (int(x_center + width/2), int(y_center + height/2))
                    cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)

            if len(inert_indices) > 0:
                status_text += "Inert"

        # Draw status
        cv2.putText(frame, f"State: {status_text}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame

    async def _write_output(self, stream_id: str):
        """Write processed frames to output file"""
        config = self.streams[stream_id]
        if not config.output_path:
            return

        # Get first frame to determine video properties
        first_frame = None
        while first_frame is None and self.running:
            try:
                first_frame = self.output_queues[stream_id].get(timeout=1.0)
            except:
                continue

        if first_frame is None:
            return

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30  # Default FPS
        height, width = first_frame.shape[:2]

        out = cv2.VideoWriter(config.output_path, fourcc, fps, (width, height))

        try:
            # Write first frame
            out.write(cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))

            # Write remaining frames
            while self.running:
                try:
                    frame = self.output_queues[stream_id].get(timeout=1.0)
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                except:
                    continue
        finally:
            out.release()

    def stop(self):
        """Stop all processing"""
        self.running = False
        self.executor.shutdown(wait=True)

    def get_stats(self) -> Dict[str, Dict]:
        """Get processing statistics for all streams"""
        stats = {}
        for stream_id, state in self.stream_states.items():
            stats[stream_id] = {
                "frames_processed": state.frame_cnt,
                "tracked_objects": len(state.inert_coor),
                "queue_size": self.output_queues[stream_id].qsize() if stream_id in self.output_queues else 0
            }
        return stats


# Real-time monitoring
class ResourceMonitor:
    def __init__(self):
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.gpu_usage = 0.0

    def get_system_stats(self) -> Dict[str, float]:
        """Get current system resource usage"""
        try:
            import psutil
            self.cpu_usage = psutil.cpu_percent(interval=1)
            self.memory_usage = psutil.virtual_memory().percent

            # GPU stats if available
            if torch.cuda.is_available():
                self.gpu_usage = torch.cuda.utilization()

            return {
                "cpu": self.cpu_usage,
                "memory": self.memory_usage,
                "gpu": self.gpu_usage
            }
        except ImportError:
            return {"cpu": 0, "memory": 0, "gpu": 0}

    def should_scale_down(self) -> bool:
        """Determine if we should reduce processing load"""
        return self.cpu_usage > 85 or self.memory_usage > 90

    def should_scale_up(self) -> bool:
        """Determine if we can increase processing load"""
        return self.cpu_usage < 60 and self.memory_usage < 70