import asyncio
import json
import signal
import sys
from pathlib import Path
from typing import List, Optional

import click

from multi_stream_processor import MultiStreamProcessor, ResourceMonitor, StreamConfig


class MultiStreamApp:
    def __init__(self):
        self.processor: Optional[MultiStreamProcessor] = None
        self.monitor = ResourceMonitor()
        self.running = False

    async def run_config_file(self, config_path: str, model_path: str):
        """Run multi-stream processing from a JSON config file"""
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        self.processor = MultiStreamProcessor(
            model_path=model_path,
            max_workers=config_data.get('max_workers', 4)
        )

        # Add streams from config
        for stream_data in config_data['streams']:
            config = StreamConfig(**stream_data)
            self.processor.add_stream(config)

        print(f"Starting processing of {len(config_data['streams'])} streams...")

        # Setup signal handlers
        def signal_handler(signum, frame):
            print("\nStopping gracefully...")
            self.running = False
            if self.processor:
                self.processor.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        self.running = True

        # Start monitoring task
        monitor_task = asyncio.create_task(self._monitor_resources())

        # Start processing
        try:
            await self.processor.start_processing()
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.running = False
            monitor_task.cancel()
            if self.processor:
                self.processor.stop()

    async def _monitor_resources(self):
        """Monitor system resources and print stats"""
        while self.running:
            try:
                stats = self.monitor.get_system_stats()
                processing_stats = self.processor.get_stats() if self.processor else {}

                print(f"\r[STATS] CPU: {stats['cpu']:.1f}% | "
                      f"Memory: {stats['memory']:.1f}% | "
                      f"GPU: {stats['gpu']:.1f}% | "
                      f"Streams: {len(processing_stats)}", end='')

                # Auto-scaling logic
                if self.monitor.should_scale_down():
                    print("\n[WARNING] High resource usage detected")

                await asyncio.sleep(5)
            except Exception as e:
                print(f"\n[ERROR] Monitoring error: {e}")
                await asyncio.sleep(10)


@click.group()
def cli():
    """Multi-stream pet tracking system"""
    pass


@cli.command()
@click.option("--config", required=True, help="JSON configuration file path")
@click.option("--model", default="weights/best.pt", help="YOLO model path")
def run_config(config: str, model: str):
    """Run multi-stream processing from configuration file"""
    app = MultiStreamApp()
    asyncio.run(app.run_config_file(config, model))


@cli.command()
@click.option("--streams", required=True, help="Comma-separated list of input sources")
@click.option("--outputs", help="Comma-separated list of output paths (optional)")
@click.option("--methods", default="bytetrack", help="Comma-separated tracking methods")
@click.option("--model", default="weights/best.pt", help="YOLO model path")
@click.option("--task-fight", is_flag=True, help="Enable fight detection")
@click.option("--task-escape", is_flag=True, help="Enable escape detection")
@click.option("--task-inert", is_flag=True, help="Enable inert detection")
@click.option("--max-workers", default=4, help="Maximum worker threads")
def run_streams(
    streams: str,
    outputs: Optional[str],
    methods: str,
    model: str,
    task_fight: bool,
    task_escape: bool,
    task_inert: bool,
    max_workers: int
):
    """Run multi-stream processing with CLI arguments"""

    stream_list = [s.strip() for s in streams.split(',')]
    output_list = [o.strip() for o in outputs.split(',')] if outputs else [None] * len(stream_list)
    method_list = [m.strip() for m in methods.split(',')]

    # Pad lists to match stream count
    if len(output_list) < len(stream_list):
        output_list.extend([None] * (len(stream_list) - len(output_list)))
    if len(method_list) < len(stream_list):
        method_list.extend([method_list[-1]] * (len(stream_list) - len(method_list)))

    app = MultiStreamApp()

    async def run_cli_streams():
        app.processor = MultiStreamProcessor(model_path=model, max_workers=max_workers)

        # Add streams
        for i, (stream_source, output_path, method) in enumerate(zip(stream_list, output_list, method_list)):
            config = StreamConfig(
                stream_id=f"stream_{i}",
                input_source=stream_source,
                output_path=output_path,
                method=method,
                task_fight=task_fight,
                task_escape=task_escape,
                task_inert=task_inert
            )
            app.processor.add_stream(config)

        print(f"Starting processing of {len(stream_list)} streams...")

        # Setup signal handlers
        def signal_handler(signum, frame):
            print("\nStopping gracefully...")
            app.running = False
            if app.processor:
                app.processor.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        app.running = True

        # Start monitoring
        monitor_task = asyncio.create_task(app._monitor_resources())

        try:
            await app.processor.start_processing()
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            app.running = False
            monitor_task.cancel()
            if app.processor:
                app.processor.stop()

    asyncio.run(run_cli_streams())


@cli.command()
@click.option("--rtsp-urls", required=True, help="Comma-separated RTSP URLs")
@click.option("--outputs", help="Comma-separated output paths")
@click.option("--model", default="weights/best.pt", help="YOLO model path")
@click.option("--buffer-size", default=30, help="Frame buffer size per stream")
def rtsp(rtsp_urls: str, outputs: Optional[str], model: str, buffer_size: int):
    """Process multiple RTSP streams in real-time"""

    url_list = [url.strip() for url in rtsp_urls.split(',')]
    output_list = [o.strip() for o in outputs.split(',')] if outputs else [None] * len(url_list)

    if len(output_list) < len(url_list):
        output_list.extend([None] * (len(url_list) - len(output_list)))

    app = MultiStreamApp()

    async def run_rtsp():
        app.processor = MultiStreamProcessor(model_path=model, max_workers=len(url_list))

        for i, (rtsp_url, output_path) in enumerate(zip(url_list, output_list)):
            config = StreamConfig(
                stream_id=f"rtsp_{i}",
                input_source=rtsp_url,
                output_path=output_path,
                method="bytetrack",  # ByteTrack is fastest for real-time
                task_fight=True,
                task_escape=False,  # Disable escape for RTSP by default
                task_inert=True
            )
            app.processor.add_stream(config)

        print(f"Starting RTSP processing of {len(url_list)} streams...")
        print("Press Ctrl+C to stop")

        def signal_handler(signum, frame):
            print("\nStopping RTSP processing...")
            app.running = False
            if app.processor:
                app.processor.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        app.running = True

        monitor_task = asyncio.create_task(app._monitor_resources())

        try:
            await app.processor.start_processing()
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            app.running = False
            monitor_task.cancel()
            if app.processor:
                app.processor.stop()

    asyncio.run(run_rtsp())


@cli.command()
def create_config():
    """Create a sample configuration file"""
    sample_config = {
        "max_workers": 4,
        "streams": [
            {
                "stream_id": "camera_1",
                "input_source": "video1.mp4",
                "output_path": "output1.mp4",
                "method": "bytetrack",
                "task_fight": True,
                "task_escape": True,
                "task_inert": True,
                "threshold": 0.1,
                "inert_threshold": 50,
                "inert_frames": 100
            },
            {
                "stream_id": "camera_2",
                "input_source": "0",  # Webcam
                "output_path": "webcam_output.mp4",
                "method": "botsort",
                "task_fight": True,
                "task_escape": False,
                "task_inert": True
            },
            {
                "stream_id": "rtsp_feed",
                "input_source": "rtsp://192.168.1.100:8080/video",
                "output_path": None,  # No output file
                "method": "bytetrack",
                "task_fight": True,
                "task_escape": False,
                "task_inert": True
            }
        ]
    }

    config_path = "multi_stream_config.json"
    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)

    print(f"Sample configuration saved to {config_path}")
    print("\nTo run: python main_multi.py run-config --config multi_stream_config.json")


if __name__ == "__main__":
    cli()