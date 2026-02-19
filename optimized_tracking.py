import numpy as np
import torch
import cv2
from typing import List, Tuple, Optional
from ultralytics import YOLO
from deep_sort.detection import Detection


class OptimizedTracker:
    """Optimized tracking with batch processing and memory efficiency"""

    def __init__(self, model_path: str, device: str = "auto"):
        self.model = YOLO(model_path)

        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model.to(self.device)

        # Optimization settings
        self.batch_size = 4
        self.half_precision = torch.cuda.is_available()

        if self.half_precision:
            self.model.model.half()

    def track_batch_bytetrack(self, frames: List[np.ndarray]) -> List[Tuple]:
        """Process multiple frames in batch for ByteTrack"""
        results = []

        # Process in batches
        for i in range(0, len(frames), self.batch_size):
            batch_frames = frames[i:i + self.batch_size]

            # Run batch inference
            batch_results = self.model.track(
                batch_frames,
                persist=True,
                conf=0.7,
                iou=0.5,
                classes=[1],
                tracker="bytetrack.yaml",
                device=self.device,
                half=self.half_precision
            )

            # Process results
            for result, frame in zip(batch_results, batch_frames):
                ids = result.boxes.id

                if ids is None:
                    boxes, track_ids = [], []
                    processed_frame = frame
                else:
                    boxes = result.boxes.xywh.cpu().numpy()
                    track_ids = result.boxes.id.int().cpu().tolist()
                    processed_frame = result.plot()

                results.append((boxes, track_ids, processed_frame))

        return results

    def track_batch_botsort(self, frames: List[np.ndarray]) -> List[Tuple]:
        """Process multiple frames in batch for BoT-SORT"""
        results = []

        for i in range(0, len(frames), self.batch_size):
            batch_frames = frames[i:i + self.batch_size]

            batch_results = self.model.track(
                batch_frames,
                persist=True,
                conf=0.7,
                iou=0.5,
                classes=[1],
                tracker="botsort.yaml",
                device=self.device,
                half=self.half_precision
            )

            for result, frame in zip(batch_results, batch_frames):
                ids = result.boxes.id

                if ids is None:
                    boxes, track_ids = [], []
                    processed_frame = frame
                else:
                    boxes = result.boxes.xywh.cpu().numpy()
                    track_ids = result.boxes.id.int().cpu().tolist()
                    processed_frame = result.plot()

                results.append((boxes, track_ids, processed_frame))

        return results

    def optimize_frame(self, frame: np.ndarray, target_size: int = 640) -> np.ndarray:
        """Optimize frame for faster processing"""
        h, w = frame.shape[:2]

        # Resize if too large
        if max(h, w) > target_size:
            scale = target_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        return frame

    def preprocess_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Preprocess batch of frames for optimal inference"""
        processed_frames = []

        for frame in frames:
            # Optimize frame size
            optimized = self.optimize_frame(frame)
            processed_frames.append(optimized)

        return processed_frames


class MemoryEfficientBuffer:
    """Memory-efficient circular buffer for frames"""

    def __init__(self, maxsize: int = 30, frame_skip: int = 1):
        self.maxsize = maxsize
        self.frame_skip = frame_skip
        self.buffer = []
        self.frame_count = 0

    def add_frame(self, frame: np.ndarray) -> bool:
        """Add frame to buffer, returns True if frame was added"""
        self.frame_count += 1

        # Skip frames based on frame_skip
        if self.frame_count % self.frame_skip != 0:
            return False

        # Add to buffer
        if len(self.buffer) >= self.maxsize:
            self.buffer.pop(0)  # Remove oldest

        # Store compressed frame to save memory
        compressed = self._compress_frame(frame)
        self.buffer.append(compressed)
        return True

    def get_frames(self, count: Optional[int] = None) -> List[np.ndarray]:
        """Get frames from buffer"""
        if count is None:
            count = len(self.buffer)

        frames = []
        for compressed in self.buffer[-count:]:
            frame = self._decompress_frame(compressed)
            frames.append(frame)

        return frames

    def _compress_frame(self, frame: np.ndarray) -> bytes:
        """Compress frame to save memory"""
        # Use JPEG compression
        _, compressed = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return compressed.tobytes()

    def _decompress_frame(self, compressed: bytes) -> np.ndarray:
        """Decompress frame"""
        nparr = np.frombuffer(compressed, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame

    def clear(self):
        """Clear buffer"""
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class AdaptiveFrameSkipper:
    """Adaptive frame skipping based on processing speed"""

    def __init__(self):
        self.processing_times = []
        self.target_fps = 30
        self.current_skip = 1
        self.max_skip = 5

    def update_processing_time(self, processing_time: float):
        """Update processing time statistics"""
        self.processing_times.append(processing_time)

        # Keep only recent times
        if len(self.processing_times) > 10:
            self.processing_times.pop(0)

        # Adjust skip rate
        avg_time = np.mean(self.processing_times)
        target_time = 1.0 / self.target_fps

        if avg_time > target_time * 1.5:  # Too slow
            self.current_skip = min(self.current_skip + 1, self.max_skip)
        elif avg_time < target_time * 0.8:  # Fast enough
            self.current_skip = max(self.current_skip - 1, 1)

    def should_process_frame(self, frame_number: int) -> bool:
        """Determine if frame should be processed"""
        return frame_number % self.current_skip == 0

    def get_current_skip(self) -> int:
        """Get current skip rate"""
        return self.current_skip


class GPUMemoryManager:
    """Manage GPU memory for optimal performance"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def clear_cache(self):
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_memory_stats(self) -> dict:
        """Get current GPU memory usage"""
        if not torch.cuda.is_available():
            return {"allocated": 0, "cached": 0, "max_allocated": 0}

        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "cached": torch.cuda.memory_reserved() / 1024**3,     # GB
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3  # GB
        }

    def optimize_memory(self):
        """Optimize GPU memory usage"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

            # Set memory fraction if needed
            memory_stats = self.get_memory_stats()
            if memory_stats["allocated"] > 2.0:  # If using more than 2GB
                torch.cuda.set_per_process_memory_fraction(0.8)


# Performance monitoring
class PerformanceProfiler:
    """Profile performance of different components"""

    def __init__(self):
        self.timings = {}
        self.current_timings = {}

    def start_timer(self, name: str):
        """Start timing an operation"""
        import time
        self.current_timings[name] = time.time()

    def end_timer(self, name: str):
        """End timing an operation"""
        import time
        if name in self.current_timings:
            elapsed = time.time() - self.current_timings[name]

            if name not in self.timings:
                self.timings[name] = []

            self.timings[name].append(elapsed)

            # Keep only recent timings
            if len(self.timings[name]) > 100:
                self.timings[name] = self.timings[name][-50:]

            del self.current_timings[name]
            return elapsed
        return 0

    def get_stats(self) -> dict:
        """Get performance statistics"""
        stats = {}
        for name, times in self.timings.items():
            if times:
                stats[name] = {
                    "avg": np.mean(times),
                    "min": np.min(times),
                    "max": np.max(times),
                    "count": len(times)
                }
        return stats

    def print_stats(self):
        """Print performance statistics"""
        stats = self.get_stats()
        print("\n=== Performance Stats ===")
        for name, stat in stats.items():
            print(f"{name}: avg={stat['avg']:.3f}s, min={stat['min']:.3f}s, max={stat['max']:.3f}s")