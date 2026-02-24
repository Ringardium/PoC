#!/usr/bin/env python3
"""
Multi-Stream Pet Tracking System - PoC
Optimized for A100 GPU with 1-8 streams at 640p 15fps

Usage:
    python main.py run --config config.json
    python main.py run --streams "video1.mp4,video2.mp4" --model ../weights/best.pt
    python main.py create-config --streams 4 --output my_config.json
    python main.py benchmark --config config.json --duration 60
"""

import asyncio
import logging
import platform
import signal
import sys
from pathlib import Path
from typing import Optional

import click

from config import SystemConfig, StreamConfig, create_sample_config
from stream_processor import MultiStreamProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class GracefulExit:
    """Handle graceful shutdown"""

    def __init__(self, processor: Optional[MultiStreamProcessor] = None):
        self.processor = processor
        self.should_exit = False

    def setup_signals(self):
        """Setup signal handlers"""
        if platform.system() != 'Windows':
            signal.signal(signal.SIGTERM, self._handler)
        signal.signal(signal.SIGINT, self._handler)

    def _handler(self, signum, frame):
        logger.info("\nShutdown signal received...")
        self.should_exit = True
        if self.processor:
            self.processor.stop()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Multi-Stream Pet Tracking System optimized for A100 GPU"""
    pass


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True),
              help="JSON configuration file path")
@click.option("--streams", "-s",
              help="Comma-separated list of input sources (video files, RTSP URLs, or webcam indices)")
@click.option("--outputs", "-o",
              help="Comma-separated list of output file paths")
@click.option("--model", "-m", default="../weights/best.pt",
              help="YOLO model path")
@click.option("--method", default="bytetrack",
              type=click.Choice(["bytetrack", "botsort", "deepsort"]),
              help="Tracking method")
@click.option("--max-streams", default=6, type=int,
              help="Maximum number of streams (max 6 for 30fps real-time)")
@click.option("--target-fps", default=30, type=int,
              help="Target FPS per stream")
@click.option("--task-fight/--no-task-fight", default=True,
              help="Enable fight detection")
@click.option("--task-inert/--no-task-inert", default=True,
              help="Enable inert detection")
@click.option("--task-escape/--no-task-escape", default=False,
              help="Enable escape detection")
@click.option("--half/--no-half", default=True,
              help="Enable FP16 half precision")
@click.option("--batch-size", default=6, type=int,
              help="Batch size for inference (max 6 for 30fps real-time)")
@click.option("--use-reid/--no-use-reid", default=False,
              help="Enable ReID-based ID correction")
@click.option("--reid-method", default="adaptive",
              type=click.Choice(["adaptive", "histogram", "mobilenet"]),
              help="ReID feature extraction method")
@click.option("--reid-threshold", default=0.5, type=float,
              help="ReID similarity threshold")
@click.option("--reid-global-id/--no-reid-global-id", default=False,
              help="Enable global ID assignment across streams")
@click.option("--yolo-conf", default=0.7, type=float,
              help="YOLO confidence threshold (0.0-1.0)")
@click.option("--yolo-iou", default=0.5, type=float,
              help="YOLO IoU threshold for NMS (0.0-1.0)")
@click.option("--yolo-augment/--no-yolo-augment", default=False,
              help="Enable test-time augmentation (slower but more accurate)")
@click.option("--yolo-agnostic-nms/--no-yolo-agnostic-nms", default=False,
              help="Enable class-agnostic NMS")
@click.option("--yolo-verbose/--no-yolo-verbose", default=False,
              help="Enable verbose YOLO inference logging")
@click.option("--yolo-persist/--no-yolo-persist", default=True,
              help="Persist tracks across frames")
def run(
    config: Optional[str],
    streams: Optional[str],
    outputs: Optional[str],
    model: str,
    method: str,
    max_streams: int,
    target_fps: int,
    task_fight: bool,
    task_inert: bool,
    task_escape: bool,
    half: bool,
    batch_size: int,
    use_reid: bool,
    reid_method: str,
    reid_threshold: float,
    reid_global_id: bool,
    yolo_conf: float,
    yolo_iou: float,
    yolo_augment: bool,
    yolo_agnostic_nms: bool,
    yolo_verbose: bool,
    yolo_persist: bool
):
    """Run multi-stream processing"""

    # Load or create config
    if config:
        logger.info(f"Loading configuration from {config}")
        sys_config = SystemConfig.load(config)
    elif streams:
        # Create config from CLI arguments
        stream_list = [s.strip() for s in streams.split(',')]
        output_list = [o.strip() for o in outputs.split(',')] if outputs else [None] * len(stream_list)

        # Pad output list
        while len(output_list) < len(stream_list):
            output_list.append(None)

        sys_config = SystemConfig(
            model_path=model,
            processing=__import__('config').ProcessingConfig(
                max_streams=max_streams,
                target_resolution=(640, 640)
            ),
            gpu=__import__('config').GPUConfig(
                half_precision=half,
                batch_size=batch_size
            )
        )

        for i, (src, out) in enumerate(zip(stream_list, output_list)):
            stream_config = StreamConfig(
                stream_id=f"stream_{i+1}",
                input_source=src,
                output_path=out,
                method=method,
                task_fight=task_fight,
                task_inert=task_inert,
                task_escape=task_escape,
                target_fps=target_fps,
                use_reid=use_reid,
                reid_method=reid_method,
                reid_threshold=reid_threshold,
                reid_global_id=reid_global_id,
                yolo_conf=yolo_conf,
                yolo_iou=yolo_iou,
                yolo_augment=yolo_augment,
                yolo_agnostic_nms=yolo_agnostic_nms,
                yolo_verbose=yolo_verbose,
                yolo_persist=yolo_persist
            )
            sys_config.streams.append(stream_config)
    else:
        logger.error("Either --config or --streams must be provided")
        sys.exit(1)

    # Validate
    if not sys_config.streams:
        logger.error("No streams configured")
        sys.exit(1)

    if len(sys_config.streams) > max_streams:
        logger.warning(f"Too many streams ({len(sys_config.streams)}), limiting to {max_streams}")
        sys_config.streams = sys_config.streams[:max_streams]

    # Create processor (headless — no web viewer)
    processor = MultiStreamProcessor(sys_config, web_enabled=False)

    # Setup graceful exit
    exit_handler = GracefulExit(processor)
    exit_handler.setup_signals()

    # Run
    logger.info(f"Starting {len(sys_config.streams)} streams...")
    logger.info(f"Model: {sys_config.model_path}")
    logger.info(f"Device: cuda (FP16: {sys_config.gpu.half_precision})")
    logger.info("Press Ctrl+C to stop")
    print()

    try:
        asyncio.run(processor.start())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        processor.stop()
        logger.info("Shutdown complete")


@cli.command("create-config")
@click.option("--streams", "-n", default=4, type=int,
              help="Number of streams to configure")
@click.option("--output", "-o", default="config.json",
              help="Output configuration file path")
@click.option("--template", type=click.Choice(["video", "rtsp", "webcam", "mixed"]),
              default="video", help="Configuration template type")
def create_config_cmd(streams: int, output: str, template: str):
    """Create a sample configuration file"""

    config = create_sample_config(streams)

    # Adjust based on template
    if template == "video":
        for i, stream in enumerate(config.streams):
            stream.input_source = f"video_{i+1}.mp4"
            stream.output_path = f"output/result_{i+1}.mp4"

    elif template == "rtsp":
        for i, stream in enumerate(config.streams):
            stream.input_source = f"rtsp://192.168.1.{100+i}:8080/video"
            stream.output_path = None  # Real-time only

    elif template == "webcam":
        for i, stream in enumerate(config.streams):
            stream.input_source = str(i)
            stream.output_path = f"output/webcam_{i}.mp4"

    elif template == "mixed":
        sources = [
            ("video1.mp4", "output/video1_result.mp4"),
            ("rtsp://192.168.1.100:8080/video", None),
            ("0", "output/webcam.mp4"),
            ("video2.mp4", "output/video2_result.mp4"),
        ]
        for i, stream in enumerate(config.streams):
            if i < len(sources):
                stream.input_source, stream.output_path = sources[i]

    config.save(output)
    logger.info(f"Configuration saved to {output}")
    logger.info(f"Configured {streams} streams with '{template}' template")
    print(f"\nTo run: python main.py run --config {output}")


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="Configuration file to benchmark")
@click.option("--duration", "-d", default=60, type=int,
              help="Benchmark duration in seconds")
@click.option("--warmup", default=10, type=int,
              help="Warmup duration in seconds")
def benchmark(config: str, duration: int, warmup: int):
    """Benchmark performance with the given configuration"""
    import time

    logger.info(f"Loading configuration from {config}")
    sys_config = SystemConfig.load(config)

    # Create processor (headless — no web viewer)
    processor = MultiStreamProcessor(sys_config, web_enabled=False)

    # Setup exit handler
    exit_handler = GracefulExit(processor)
    exit_handler.setup_signals()

    results = {
        "warmup_fps": [],
        "benchmark_fps": [],
        "latencies": []
    }

    async def benchmark_runner():
        # Start processing in background
        process_task = asyncio.create_task(processor.start())

        try:
            # Warmup
            logger.info(f"Warming up for {warmup} seconds...")
            await asyncio.sleep(warmup)

            # Benchmark
            logger.info(f"Benchmarking for {duration} seconds...")
            start_time = time.time()

            while time.time() - start_time < duration:
                if exit_handler.should_exit:
                    break

                stats = processor.get_stats()
                results["benchmark_fps"].append(stats["totals"]["fps"])
                await asyncio.sleep(1)

            # Results
            processor.stop()
            await asyncio.sleep(1)

        except asyncio.CancelledError:
            pass

    try:
        asyncio.run(benchmark_runner())
    except KeyboardInterrupt:
        pass

    # Print results
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)

    if results["benchmark_fps"]:
        avg_fps = sum(results["benchmark_fps"]) / len(results["benchmark_fps"])
        max_fps = max(results["benchmark_fps"])
        min_fps = min(results["benchmark_fps"])

        print(f"Streams:     {len(sys_config.streams)}")
        print(f"Duration:    {duration}s")
        print(f"Avg FPS:     {avg_fps:.1f}")
        print(f"Max FPS:     {max_fps:.1f}")
        print(f"Min FPS:     {min_fps:.1f}")
        print(f"Target FPS:  {sys_config.streams[0].target_fps * len(sys_config.streams)}")

        efficiency = (avg_fps / (sys_config.streams[0].target_fps * len(sys_config.streams))) * 100
        print(f"Efficiency:  {efficiency:.1f}%")
    else:
        print("No benchmark data collected")

    print("="*50)


@cli.command()
def info():
    """Show system information"""
    import torch

    print("\n=== System Information ===")
    print(f"Python:      {sys.version.split()[0]}")
    print(f"Platform:    {platform.system()} {platform.release()}")
    print(f"PyTorch:     {torch.__version__}")

    if torch.cuda.is_available():
        print(f"\n=== GPU Information ===")
        print(f"CUDA:        {torch.version.cuda}")
        print(f"Device:      {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"Memory:      {props.total_memory / 1024**3:.1f} GB")
        print(f"Compute:     {props.major}.{props.minor}")

        # Check for A100 features
        if "A100" in torch.cuda.get_device_name(0):
            print("\n=== A100 Optimizations ===")
            print(f"TF32:        Supported")
            print(f"FP16:        Supported")
            print(f"Recommended batch size: 6")
            print(f"Recommended streams: 6 (30fps real-time guaranteed)")
    else:
        print("\nNo CUDA GPU available")

    print()


@cli.command("list-streams")
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="Configuration file")
def list_streams(config: str):
    """List streams in a configuration file"""
    sys_config = SystemConfig.load(config)

    print(f"\n{'ID':<15} {'Source':<40} {'Method':<12} {'Tasks'}")
    print("-" * 80)

    for stream in sys_config.streams:
        tasks = []
        if stream.task_fight:
            tasks.append("fight")
        if stream.task_escape:
            tasks.append("escape")
        if stream.task_inert:
            tasks.append("inert")

        task_str = ", ".join(tasks) if tasks else "none"
        source = stream.input_source[:37] + "..." if len(stream.input_source) > 40 else stream.input_source

        print(f"{stream.stream_id:<15} {source:<40} {stream.method:<12} {task_str}")

    print(f"\nTotal streams: {len(sys_config.streams)}")


if __name__ == "__main__":
    cli()
