import click
import json
import os
from pathlib import Path
from typing import List, Optional

from ultralytics import YOLO
from models.lightning_training import LightningTrainer, quick_train_setup
from models.lightning_inference import LightningYOLOInference


@click.group()
def cli():
    """AI Model Conversion and Optimization CLI"""
    pass


# PyTorch Mobile Commands (No TensorFlow dependency)
@cli.group()
def mobile():
    """PyTorch Mobile optimization commands"""
    pass


@mobile.command()
@click.option("--model", required=True, help="Path to YOLO model (.pt file)")
@click.option("--output", help="Output path (optional, auto-generates if not provided)")
@click.option("--format", default="torchscript", type=click.Choice(["torchscript", "onnx", "coreml", "engine"]), help="Export format")
@click.option("--optimize", is_flag=True, default=True, help="Apply mobile optimizations")
@click.option("--half", is_flag=True, help="Use FP16 precision")
@click.option("--int8", is_flag=True, help="Use INT8 quantization")
def export(model: str, output: Optional[str], format: str, optimize: bool, half: bool, int8: bool):
    """Export YOLO model for mobile deployment"""
    click.echo(f"Exporting {model} to {format} format...")

    try:
        yolo_model = YOLO(model)

        # Export with specified format and optimizations
        exported_path = yolo_model.export(
            format=format,
            optimize=optimize,
            half=half,
            int8=int8,
            dynamic=False,  # Static shapes for mobile
            simplify=True   # Simplify for better mobile performance
        )

        # Move to custom output path if specified
        if output and output != exported_path:
            import shutil
            shutil.move(exported_path, output)
            exported_path = output

        # Get file size
        file_size = os.path.getsize(exported_path) / (1024 * 1024)
        click.echo(f"✅ Export successful: {exported_path}")
        click.echo(f"📱 Model size: {file_size:.2f} MB")

        if format == "torchscript":
            click.echo("💡 This model is ready for PyTorch Mobile/ExecuTorch")
        elif format == "onnx":
            click.echo("💡 This model can be used with ONNX Runtime Mobile")
        elif format == "coreml":
            click.echo("💡 This model is optimized for iOS devices")

    except Exception as e:
        click.echo(f"❌ Export failed: {e}")


@mobile.command()
@click.option("--model", required=True, help="Path to exported model")
@click.option("--image", required=True, help="Test image path")
@click.option("--format", required=True, type=click.Choice(["torchscript", "onnx"]), help="Model format")
@click.option("--num-runs", default=50, help="Number of benchmark runs")
def benchmark(model: str, image: str, format: str, num_runs: int):
    """Benchmark mobile model performance"""
    import cv2
    import torch
    import time
    import numpy as np

    click.echo(f"Benchmarking {format} model: {model}")

    try:
        # Load image
        img = cv2.imread(image)
        if img is None:
            click.echo(f"❌ Could not load image: {image}")
            return

        # Prepare input
        img_resized = cv2.resize(img, (640, 640))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        # Load model based on format
        if format == "torchscript":
            model_loaded = torch.jit.load(model)
            model_loaded.eval()
        elif format == "onnx":
            try:
                import onnxruntime as ort
                model_loaded = ort.InferenceSession(model)
            except ImportError:
                click.echo("❌ ONNX Runtime not installed. Install with: pip install onnxruntime")
                return

        # Warmup
        for _ in range(5):
            if format == "torchscript":
                with torch.no_grad():
                    _ = model_loaded(img_tensor)
            elif format == "onnx":
                _ = model_loaded.run(None, {"images": img_tensor.numpy()})

        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            if format == "torchscript":
                with torch.no_grad():
                    _ = model_loaded(img_tensor)
            elif format == "onnx":
                _ = model_loaded.run(None, {"images": img_tensor.numpy()})
            end_time = time.time()
            times.append(end_time - start_time)

        # Results
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1.0 / avg_time

        # Model size
        model_size_mb = os.path.getsize(model) / (1024 * 1024)

        click.echo("\n=== Mobile Benchmark Results ===")
        click.echo(f"📱 Model size: {model_size_mb:.2f} MB")
        click.echo(f"⚡ Average time: {avg_time*1000:.2f} ms")
        click.echo(f"🚀 FPS: {fps:.1f}")
        click.echo(f"⏱️  Min time: {min_time*1000:.2f} ms")
        click.echo(f"⏱️  Max time: {max_time*1000:.2f} ms")

    except Exception as e:
        click.echo(f"❌ Benchmark failed: {e}")


@mobile.command()
@click.option("--model", required=True, help="Path to YOLO model")
@click.option("--target-size-mb", default=50.0, help="Target model size in MB")
@click.option("--output-dir", default="./mobile_optimized", help="Output directory")
def optimize(model: str, target_size_mb: float, output_dir: str):
    """Auto-optimize YOLO model for mobile deployment"""
    click.echo(f"Optimizing {model} for mobile (target: {target_size_mb} MB)...")

    os.makedirs(output_dir, exist_ok=True)

    try:
        yolo_model = YOLO(model)
        base_name = Path(model).stem

        results = {}

        # Try different optimization strategies
        configs = [
            {"format": "torchscript", "half": False, "optimize": True, "name": "torchscript_fp32"},
            {"format": "torchscript", "half": True, "optimize": True, "name": "torchscript_fp16"},
            {"format": "onnx", "half": False, "optimize": True, "name": "onnx_fp32"},
            {"format": "onnx", "half": True, "optimize": True, "name": "onnx_fp16"},
        ]

        best_model = None
        best_score = 0

        for config in configs:
            try:
                click.echo(f"  Trying {config['name']}...")

                exported_path = yolo_model.export(
                    format=config["format"],
                    optimize=config["optimize"],
                    half=config["half"],
                    dynamic=False,
                    simplify=True
                )

                # Move to output directory
                output_path = os.path.join(output_dir, f"{base_name}_{config['name']}.{config['format']}")
                if config["format"] == "torchscript":
                    output_path = output_path.replace(".torchscript", ".pt")

                import shutil
                shutil.move(exported_path, output_path)

                # Check size
                size_mb = os.path.getsize(output_path) / (1024 * 1024)

                # Score based on size constraint
                if size_mb <= target_size_mb:
                    score = 100 - (size_mb / target_size_mb) * 50  # Prefer smaller within constraint
                else:
                    score = 50 - (size_mb - target_size_mb) * 2    # Penalize oversized

                results[config['name']] = {
                    "path": output_path,
                    "size_mb": size_mb,
                    "score": score,
                    "meets_target": size_mb <= target_size_mb
                }

                if score > best_score:
                    best_score = score
                    best_model = config['name']

                click.echo(f"    ✅ {config['name']}: {size_mb:.2f} MB")

            except Exception as e:
                click.echo(f"    ❌ {config['name']} failed: {e}")

        # Save results
        report_path = os.path.join(output_dir, "optimization_report.json")
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Summary
        click.echo("\n=== Mobile Optimization Results ===")
        if best_model:
            best = results[best_model]
            click.echo(f"🏆 Best model: {best_model}")
            click.echo(f"📱 Size: {best['size_mb']:.2f} MB")
            click.echo(f"📍 Path: {best['path']}")
            click.echo(f"🎯 Meets target: {'✅' if best['meets_target'] else '❌'}")
        else:
            click.echo("❌ No successful optimizations")

        click.echo(f"\n📊 Detailed report: {report_path}")

    except Exception as e:
        click.echo(f"❌ Optimization failed: {e}")


@mobile.command()
@click.option("--model", required=True, help="Path to YOLO model")
@click.option("--sparsity", default=0.5, help="Pruning sparsity (0.0-0.9)")
@click.option("--output", help="Output path for pruned model")
def prune(model: str, sparsity: float, output: Optional[str]):
    """Apply structured pruning to reduce model size"""
    import torch
    import torch.nn.utils.prune as prune

    click.echo(f"Applying {sparsity*100:.1f}% pruning to {model}...")

    if not output:
        output = model.replace('.pt', f'_pruned_{int(sparsity*100)}.pt')

    try:
        # Load YOLO model
        yolo_model = YOLO(model)
        model_state = yolo_model.model

        # Apply magnitude-based unstructured pruning
        parameters_to_prune = []
        for name, module in model_state.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                parameters_to_prune.append((module, 'weight'))
            elif isinstance(module, torch.nn.Linear):
                parameters_to_prune.append((module, 'weight'))

        # Apply global magnitude pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity,
        )

        # Make pruning permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)

        # Save pruned model
        torch.save({
            'model': model_state.state_dict(),
            'metadata': {
                'sparsity': sparsity,
                'original_model': model
            }
        }, output)

        # Get file sizes
        original_size = os.path.getsize(model) / (1024 * 1024)
        pruned_size = os.path.getsize(output) / (1024 * 1024)
        reduction = (1 - pruned_size / original_size) * 100

        click.echo(f"✅ Pruning completed:")
        click.echo(f"📱 Original size: {original_size:.2f} MB")
        click.echo(f"✂️  Pruned size: {pruned_size:.2f} MB")
        click.echo(f"📉 Size reduction: {reduction:.1f}%")
        click.echo(f"📍 Output: {output}")

    except Exception as e:
        click.echo(f"❌ Pruning failed: {e}")


@mobile.command()
@click.option("--model", required=True, help="Path to YOLO model")
@click.option("--backend", default="fbgemm", type=click.Choice(["fbgemm", "qnnpack", "onednn"]), help="Quantization backend")
@click.option("--calibration-data", help="Directory with calibration images for quantization")
@click.option("--output", help="Output path for quantized model")
def quantize(model: str, backend: str, calibration_data: Optional[str], output: Optional[str]):
    """Apply post-training quantization (PTQ)"""
    import torch
    import torch.quantization as quant
    import cv2
    import numpy as np

    click.echo(f"Applying post-training quantization to {model}...")

    if not output:
        output = model.replace('.pt', f'_quantized_{backend}.pt')

    try:
        # Set quantization backend
        torch.backends.quantized.engine = backend

        # Load model
        yolo_model = YOLO(model)
        model_fp32 = yolo_model.model
        model_fp32.eval()

        # Prepare model for quantization
        model_fp32.qconfig = quant.get_default_qconfig(backend)
        model_prepared = quant.prepare(model_fp32)

        # Calibration with sample data
        if calibration_data:
            click.echo("Running calibration...")
            cal_dir = Path(calibration_data)
            cal_images = list(cal_dir.glob("*.jpg")) + list(cal_dir.glob("*.png"))

            with torch.no_grad():
                for i, img_path in enumerate(cal_images[:50]):  # Use 50 images for calibration
                    img = cv2.imread(str(img_path))
                    img = cv2.resize(img, (640, 640))
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    img_tensor = img_tensor.unsqueeze(0)

                    _ = model_prepared(img_tensor)

                    if (i + 1) % 10 == 0:
                        click.echo(f"  Calibrated {i + 1} images...")
        else:
            click.echo("⚠️  No calibration data provided, using dummy data")
            # Use dummy data for calibration
            dummy_input = torch.randn(1, 3, 640, 640)
            for _ in range(10):
                _ = model_prepared(dummy_input)

        # Convert to quantized model
        model_quantized = quant.convert(model_prepared)

        # Save quantized model
        torch.save({
            'model': model_quantized.state_dict(),
            'metadata': {
                'backend': backend,
                'original_model': model,
                'quantized': True
            }
        }, output)

        # Get file sizes
        original_size = os.path.getsize(model) / (1024 * 1024)
        quantized_size = os.path.getsize(output) / (1024 * 1024)
        reduction = (1 - quantized_size / original_size) * 100

        click.echo(f"✅ Quantization completed:")
        click.echo(f"📱 Original size: {original_size:.2f} MB")
        click.echo(f"🔢 Quantized size: {quantized_size:.2f} MB")
        click.echo(f"📉 Size reduction: {reduction:.1f}%")
        click.echo(f"⚙️  Backend: {backend}")
        click.echo(f"📍 Output: {output}")

    except Exception as e:
        click.echo(f"❌ Quantization failed: {e}")


# PyTorch Lightning Commands
@cli.group()
def lightning():
    """PyTorch Lightning training commands"""
    pass


@lightning.command()
@click.option("--data", required=True, help="Path to data YAML file")
@click.option("--epochs", default=100, help="Number of training epochs")
@click.option("--batch-size", default=16, help="Batch size")
@click.option("--img-size", default=640, help="Image size")
@click.option("--model", default="yolo11n.yaml", help="Model configuration")
@click.option("--device", default="auto", help="Training device")
@click.option("--save-dir", default="./lightning_runs", help="Save directory")
def train(data: str, epochs: int, batch_size: int, img_size: int, model: str, device: str, save_dir: str):
    """Train YOLO model with PyTorch Lightning"""
    click.echo(f"Training YOLO model with Lightning...")
    click.echo(f"Data: {data}")
    click.echo(f"Epochs: {epochs}, Batch size: {batch_size}")

    try:
        trained_model = quick_train_setup(
            data_yaml=data,
            epochs=epochs,
            batch_size=batch_size,
            imgsz=img_size,
            device=device
        )

        click.echo("✅ Training completed successfully!")
        click.echo(f"Model saved in: {save_dir}")

    except Exception as e:
        click.echo(f"❌ Training failed: {e}")


@lightning.command()
@click.option("--checkpoint", required=True, help="Path to Lightning checkpoint")
@click.option("--image", required=True, help="Test image path")
@click.option("--output", help="Output image with detections")
@click.option("--conf-threshold", default=0.25, help="Confidence threshold")
@click.option("--iou-threshold", default=0.45, help="IoU threshold for NMS")
def infer(checkpoint: str, image: str, output: Optional[str], conf_threshold: float, iou_threshold: float):
    """Run inference with Lightning trained model"""
    import cv2

    click.echo(f"Running inference with {checkpoint}...")

    try:
        # Load model
        model = LightningYOLOInference(
            checkpoint_path=checkpoint,
            confidence_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )

        # Load image
        img = cv2.imread(image)
        if img is None:
            click.echo(f"❌ Could not load image: {image}")
            return

        # Run inference
        results = model.predict(img)

        click.echo("✅ Inference completed")
        click.echo(f"Detections: {len(results['boxes'])}")

        if output and results['boxes']:
            # Visualize results
            vis_img = model.visualize_results(img, results, class_names=["background", "pet"])
            cv2.imwrite(output, vis_img)
            click.echo(f"Result saved to: {output}")

    except Exception as e:
        click.echo(f"❌ Inference failed: {e}")


@lightning.command()
@click.option("--checkpoint", required=True, help="Path to Lightning checkpoint")
@click.option("--test-images", required=True, help="Directory with test images")
@click.option("--num-runs", default=100, help="Number of benchmark runs")
def benchmark_lightning(checkpoint: str, test_images: str, num_runs: int):
    """Benchmark Lightning model performance"""
    click.echo(f"Benchmarking Lightning model...")

    try:
        # Load model
        model = LightningYOLOInference(checkpoint_path=checkpoint)

        # Get test images
        test_dir = Path(test_images)
        test_image_list = [str(p) for p in test_dir.glob("*.jpg")] + [str(p) for p in test_dir.glob("*.png")]

        if not test_image_list:
            click.echo(f"❌ No test images found in {test_images}")
            return

        # Benchmark
        results = model.benchmark(test_image_list, num_runs)

        click.echo("\n=== Lightning Model Benchmark ===")
        if "error" in results:
            click.echo(f"❌ {results['error']}")
        else:
            click.echo(f"Single image FPS: {results['single_fps']:.2f}")
            click.echo(f"Batch FPS: {results['batch_fps']:.2f}")
            click.echo(f"Batch speedup: {results['speedup']:.2f}x")
            click.echo(f"Average single time: {results['avg_single_time']:.4f}s")

    except Exception as e:
        click.echo(f"❌ Benchmark failed: {e}")


# Utility Commands
@cli.command()
@click.option("--model", required=True, help="Path to model file")
def info(model: str):
    """Show model information"""
    import torch
    from ultralytics import YOLO

    click.echo(f"Model Information: {model}")
    click.echo("=" * 50)

    try:
        # File size
        file_size = os.path.getsize(model) / (1024 * 1024)
        click.echo(f"File size: {file_size:.2f} MB")

        # Try to load as YOLO model
        try:
            yolo_model = YOLO(model)
            click.echo(f"Model type: YOLO")
            click.echo(f"Task: {yolo_model.task}")

            # Model details
            if hasattr(yolo_model.model, 'yaml'):
                click.echo(f"Architecture: {yolo_model.model.yaml.get('backbone', 'Unknown')}")

        except Exception:
            # Try as PyTorch checkpoint
            try:
                checkpoint = torch.load(model, map_location='cpu')
                if 'model' in checkpoint:
                    click.echo("Model type: PyTorch checkpoint")
                    if 'hyperparameters' in checkpoint:
                        click.echo("Contains hyperparameters")
                elif 'state_dict' in checkpoint:
                    click.echo("Model type: Lightning checkpoint")
                else:
                    click.echo("Model type: PyTorch state dict")
            except Exception:
                click.echo("Model type: Unknown")

    except Exception as e:
        click.echo(f"❌ Could not analyze model: {e}")


if __name__ == "__main__":
    cli()