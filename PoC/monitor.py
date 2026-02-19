"""Resource monitoring and performance profiling"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class StreamStats:
    """Statistics for a single stream"""
    stream_id: str
    frames_processed: int = 0
    frames_dropped: int = 0
    current_fps: float = 0.0
    avg_latency_ms: float = 0.0
    tracked_objects: int = 0
    detections: Dict[str, int] = field(default_factory=lambda: {
        "fight": 0, "escape": 0, "inert": 0, "reid_corrections": 0
    })

    # Internal tracking
    _fps_timestamps: deque = field(default_factory=lambda: deque(maxlen=30))
    _latencies: deque = field(default_factory=lambda: deque(maxlen=100))

    def update_fps(self):
        """Update FPS calculation"""
        now = time.time()
        self._fps_timestamps.append(now)

        if len(self._fps_timestamps) >= 2:
            duration = self._fps_timestamps[-1] - self._fps_timestamps[0]
            if duration > 0:
                self.current_fps = (len(self._fps_timestamps) - 1) / duration

    def add_latency(self, latency_ms: float):
        """Add latency measurement"""
        self._latencies.append(latency_ms)
        if self._latencies:
            self.avg_latency_ms = sum(self._latencies) / len(self._latencies)


@dataclass
class GPUStats:
    """GPU resource statistics"""
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    memory_percent: float = 0.0
    gpu_utilization: float = 0.0
    temperature: float = 0.0


@dataclass
class SystemStats:
    """Overall system statistics"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu: GPUStats = field(default_factory=GPUStats)
    total_fps: float = 0.0
    active_streams: int = 0


class ResourceMonitor:
    """Monitor system resources (CPU, Memory, GPU)"""

    def __init__(self):
        self._gpu_available = False
        self._pynvml_available = False
        self._nvml_handle = None
        self._init_gpu_monitoring()

    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring"""
        try:
            import torch
            self._gpu_available = torch.cuda.is_available()
        except ImportError:
            self._gpu_available = False

        # Try pynvml for detailed GPU stats
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._pynvml_available = True
        except Exception:
            self._pynvml_available = False

    def get_gpu_stats(self) -> GPUStats:
        """Get GPU statistics"""
        stats = GPUStats()

        if not self._gpu_available:
            return stats

        try:
            import torch
            stats.memory_used_gb = torch.cuda.memory_allocated() / (1024**3)
            stats.memory_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            stats.memory_percent = (stats.memory_used_gb / stats.memory_total_gb) * 100
        except Exception:
            pass

        # Detailed stats from pynvml
        if self._pynvml_available:
            try:
                import pynvml
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                stats.gpu_utilization = util.gpu

                temp = pynvml.nvmlDeviceGetTemperature(
                    self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU
                )
                stats.temperature = temp
            except Exception:
                pass

        return stats

    def get_system_stats(self) -> SystemStats:
        """Get overall system statistics"""
        stats = SystemStats()

        try:
            import psutil
            stats.cpu_percent = psutil.cpu_percent(interval=None)
            stats.memory_percent = psutil.virtual_memory().percent
        except ImportError:
            pass

        stats.gpu = self.get_gpu_stats()
        return stats

    def should_reduce_load(self) -> bool:
        """Check if system is under heavy load"""
        stats = self.get_system_stats()
        return (
            stats.cpu_percent > 90 or
            stats.memory_percent > 90 or
            stats.gpu.memory_percent > 95 or
            stats.gpu.gpu_utilization > 95
        )

    def cleanup(self):
        """Cleanup resources"""
        if self._pynvml_available:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass


class PerformanceProfiler:
    """Profile performance of different operations"""

    def __init__(self):
        self._timings: Dict[str, deque] = {}
        self._active_timers: Dict[str, float] = {}
        self._lock = threading.Lock()

    def start(self, name: str):
        """Start timing an operation"""
        self._active_timers[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """Stop timing and return elapsed time in ms"""
        if name not in self._active_timers:
            return 0.0

        elapsed = (time.perf_counter() - self._active_timers[name]) * 1000
        del self._active_timers[name]

        with self._lock:
            if name not in self._timings:
                self._timings[name] = deque(maxlen=100)
            self._timings[name].append(elapsed)

        return elapsed

    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for an operation"""
        with self._lock:
            if name not in self._timings or not self._timings[name]:
                return {"avg": 0, "min": 0, "max": 0, "count": 0}

            times = list(self._timings[name])
            return {
                "avg": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
                "count": len(times)
            }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations"""
        with self._lock:
            return {name: self.get_stats(name) for name in self._timings}

    def print_summary(self):
        """Print performance summary"""
        stats = self.get_all_stats()
        logger.info("=== Performance Summary ===")
        for name, stat in sorted(stats.items()):
            logger.info(
                f"{name}: avg={stat['avg']:.2f}ms, "
                f"min={stat['min']:.2f}ms, max={stat['max']:.2f}ms"
            )


class StatsAggregator:
    """Aggregate and display statistics from all streams"""

    def __init__(self):
        self.stream_stats: Dict[str, StreamStats] = {}
        self.resource_monitor = ResourceMonitor()
        self.profiler = PerformanceProfiler()
        self._lock = threading.Lock()

    def register_stream(self, stream_id: str):
        """Register a new stream for monitoring"""
        with self._lock:
            self.stream_stats[stream_id] = StreamStats(stream_id=stream_id)

    def get_stream_stats(self, stream_id: str) -> Optional[StreamStats]:
        """Get stats for a specific stream"""
        return self.stream_stats.get(stream_id)

    def update_stream(self, stream_id: str, **kwargs):
        """Update stream statistics"""
        with self._lock:
            if stream_id in self.stream_stats:
                stats = self.stream_stats[stream_id]
                for key, value in kwargs.items():
                    if hasattr(stats, key):
                        setattr(stats, key, value)

    def get_summary(self) -> Dict:
        """Get summary of all statistics"""
        system_stats = self.resource_monitor.get_system_stats()

        with self._lock:
            total_fps = sum(s.current_fps for s in self.stream_stats.values())
            total_processed = sum(s.frames_processed for s in self.stream_stats.values())
            total_dropped = sum(s.frames_dropped for s in self.stream_stats.values())

            streams_summary = {}
            for sid, stats in self.stream_stats.items():
                streams_summary[sid] = {
                    "fps": round(stats.current_fps, 1),
                    "latency_ms": round(stats.avg_latency_ms, 1),
                    "processed": stats.frames_processed,
                    "dropped": stats.frames_dropped,
                    "objects": stats.tracked_objects,
                    "detections": dict(stats.detections)
                }

        return {
            "system": {
                "cpu_percent": round(system_stats.cpu_percent, 1),
                "memory_percent": round(system_stats.memory_percent, 1),
                "gpu_memory_gb": round(system_stats.gpu.memory_used_gb, 2),
                "gpu_utilization": round(system_stats.gpu.gpu_utilization, 1),
                "gpu_temp": round(system_stats.gpu.temperature, 1)
            },
            "totals": {
                "fps": round(total_fps, 1),
                "frames_processed": total_processed,
                "frames_dropped": total_dropped,
                "active_streams": len(self.stream_stats)
            },
            "streams": streams_summary
        }

    def print_status(self):
        """Print current status to console"""
        summary = self.get_summary()
        sys_stats = summary["system"]
        totals = summary["totals"]

        status_line = (
            f"\r[STATS] Streams: {totals['active_streams']} | "
            f"Total FPS: {totals['fps']:.1f} | "
            f"CPU: {sys_stats['cpu_percent']:.0f}% | "
            f"GPU: {sys_stats['gpu_utilization']:.0f}% | "
            f"GPU Mem: {sys_stats['gpu_memory_gb']:.1f}GB | "
            f"Temp: {sys_stats['gpu_temp']:.0f}C"
        )
        print(status_line, end='', flush=True)

    def cleanup(self):
        """Cleanup resources"""
        self.resource_monitor.cleanup()
