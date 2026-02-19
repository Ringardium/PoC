import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

import torch
import torch.nn as nn
import numpy as np
import cv2
from ultralytics import YOLO

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import openvino as ov
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False


class ModelOptimizer:
    """Comprehensive model optimization for different deployment scenarios"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.yolo_model = YOLO(model_path)
        self.optimization_results = {}

    def optimize_for_deployment(
        self,
        target_platform: str = "cpu",
        optimization_level: str = "medium",
        output_dir: str = "./optimized_models"
    ) -> Dict[str, str]:
        """
        Optimize model for different deployment platforms

        Args:
            target_platform: 'cpu', 'gpu', 'mobile', 'edge', 'server'
            optimization_level: 'fast', 'medium', 'best'
            output_dir: Directory to save optimized models
        """
        os.makedirs(output_dir, exist_ok=True)
        optimized_models = {}

        print(f"Optimizing model for {target_platform} deployment...")

        if target_platform == "mobile":
            # TensorFlow Lite with quantization
            tflite_path = os.path.join(output_dir, "model_mobile.tflite")
            optimized_models["tflite"] = self._optimize_for_mobile(tflite_path, optimization_level)

        elif target_platform == "edge":
            # OpenVINO for Intel hardware
            if OPENVINO_AVAILABLE:
                openvino_path = os.path.join(output_dir, "model_openvino")
                optimized_models["openvino"] = self._optimize_for_openvino(openvino_path)

            # ONNX for general edge deployment
            onnx_path = os.path.join(output_dir, "model_edge.onnx")
            optimized_models["onnx"] = self._optimize_for_onnx(onnx_path, optimization_level)

        elif target_platform == "gpu":
            # TensorRT for NVIDIA GPUs
            if TENSORRT_AVAILABLE:
                tensorrt_path = os.path.join(output_dir, "model_tensorrt.engine")
                optimized_models["tensorrt"] = self._optimize_for_tensorrt(tensorrt_path, optimization_level)

            # ONNX with GPU optimizations
            onnx_path = os.path.join(output_dir, "model_gpu.onnx")
            optimized_models["onnx"] = self._optimize_for_onnx(onnx_path, optimization_level)

        elif target_platform == "server":
            # Multiple formats for server deployment
            onnx_path = os.path.join(output_dir, "model_server.onnx")
            optimized_models["onnx"] = self._optimize_for_onnx(onnx_path, optimization_level)

            if TENSORRT_AVAILABLE:
                tensorrt_path = os.path.join(output_dir, "model_server_tensorrt.engine")
                optimized_models["tensorrt"] = self._optimize_for_tensorrt(tensorrt_path, optimization_level)

        else:  # CPU
            # ONNX for CPU
            onnx_path = os.path.join(output_dir, "model_cpu.onnx")
            optimized_models["onnx"] = self._optimize_for_onnx(onnx_path, optimization_level)

            # OpenVINO for Intel CPUs
            if OPENVINO_AVAILABLE:
                openvino_path = os.path.join(output_dir, "model_cpu_openvino")
                optimized_models["openvino"] = self._optimize_for_openvino(openvino_path)

        return optimized_models

    def _optimize_for_mobile(self, output_path: str, optimization_level: str) -> str:
        """Optimize for mobile deployment"""
        try:
            from model_converter import YOLOToTFLiteConverter

            converter = YOLOToTFLiteConverter(self.model_path)

            # Choose quantization based on optimization level
            quantization_map = {
                "fast": "dynamic",
                "medium": "float16",
                "best": "int8"
            }

            quantization = quantization_map.get(optimization_level, "float16")

            # Create representative dataset for int8 quantization
            representative_dataset = None
            if quantization == "int8":
                # Generate representative data
                representative_dataset = self._generate_representative_data()

            return converter.convert_to_tflite(
                output_path,
                quantization=quantization,
                optimize_for_size=True,
                representative_dataset=representative_dataset
            )

        except Exception as e:
            print(f"Mobile optimization failed: {e}")
            return ""

    def _optimize_for_onnx(self, output_path: str, optimization_level: str) -> str:
        """Optimize for ONNX deployment"""
        try:
            # Export to ONNX with optimizations
            onnx_path = self.yolo_model.export(
                format='onnx',
                dynamic=False,
                simplify=True,
                opset=12
            )

            # Copy to desired location
            if onnx_path != output_path:
                import shutil
                shutil.copy(onnx_path, output_path)

            # Apply ONNX optimizations
            if ONNX_AVAILABLE and optimization_level in ["medium", "best"]:
                self._optimize_onnx_graph(output_path, optimization_level)

            return output_path

        except Exception as e:
            print(f"ONNX optimization failed: {e}")
            return ""

    def _optimize_for_openvino(self, output_dir: str) -> str:
        """Optimize for Intel OpenVINO"""
        try:
            if not OPENVINO_AVAILABLE:
                print("OpenVINO not available")
                return ""

            # First export to ONNX
            onnx_path = self.yolo_model.export(format='onnx')

            # Convert to OpenVINO IR format
            import subprocess
            cmd = [
                'mo',
                '--input_model', onnx_path,
                '--output_dir', output_dir,
                '--data_type', 'FP16',
                '--compress_to_fp16'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return output_dir
            else:
                print(f"OpenVINO conversion failed: {result.stderr}")
                return ""

        except Exception as e:
            print(f"OpenVINO optimization failed: {e}")
            return ""

    def _optimize_for_tensorrt(self, output_path: str, optimization_level: str) -> str:
        """Optimize for NVIDIA TensorRT"""
        try:
            if not TENSORRT_AVAILABLE:
                print("TensorRT not available")
                return ""

            # Export to ONNX first
            onnx_path = self.yolo_model.export(format='onnx')

            # Convert to TensorRT
            precision_map = {
                "fast": "fp32",
                "medium": "fp16",
                "best": "int8"
            }

            precision = precision_map.get(optimization_level, "fp16")

            # Use trtexec for conversion
            import subprocess
            cmd = [
                'trtexec',
                f'--onnx={onnx_path}',
                f'--saveEngine={output_path}',
                f'--{precision}',
                '--workspace=4096',  # 4GB workspace
                '--verbose'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return output_path
            else:
                print(f"TensorRT conversion failed: {result.stderr}")
                return ""

        except Exception as e:
            print(f"TensorRT optimization failed: {e}")
            return ""

    def _optimize_onnx_graph(self, onnx_path: str, optimization_level: str):
        """Apply ONNX graph optimizations"""
        try:
            import onnx
            from onnxoptimizer import optimize

            # Load model
            model = onnx.load(onnx_path)

            # Apply optimizations based on level
            if optimization_level == "medium":
                passes = [
                    'eliminate_deadend',
                    'eliminate_identity',
                    'eliminate_nop_transpose',
                    'eliminate_unused_initializer',
                    'extract_constant_to_initializer',
                    'fuse_add_bias_into_conv',
                    'fuse_bn_into_conv',
                    'fuse_consecutive_concats',
                    'fuse_consecutive_reduce_unsqueeze',
                    'fuse_consecutive_squeezes',
                    'fuse_consecutive_transposes',
                    'fuse_matmul_add_bias_into_gemm',
                    'fuse_pad_into_conv',
                    'fuse_transpose_into_gemm',
                ]
            else:  # "best"
                passes = 'all'

            # Optimize
            optimized_model = optimize(model, passes)

            # Save optimized model
            onnx.save(optimized_model, onnx_path)
            print(f"ONNX graph optimized with level: {optimization_level}")

        except ImportError:
            print("onnxoptimizer not available. Skipping graph optimizations.")
        except Exception as e:
            print(f"ONNX optimization failed: {e}")

    def _generate_representative_data(self, num_samples: int = 100) -> List[np.ndarray]:
        """Generate representative data for quantization"""
        # Create random data that matches typical input
        representative_data = []
        for _ in range(num_samples):
            # Generate random image data
            image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            image = image.astype(np.float32) / 255.0
            representative_data.append(image)

        return representative_data

    def benchmark_models(
        self,
        models_dict: Dict[str, str],
        test_images: List[str],
        num_runs: int = 100
    ) -> Dict[str, Dict]:
        """Benchmark different optimized models"""
        results = {}

        for model_name, model_path in models_dict.items():
            print(f"Benchmarking {model_name}...")

            try:
                if model_name == "tflite":
                    results[model_name] = self._benchmark_tflite(model_path, test_images, num_runs)
                elif model_name == "onnx":
                    results[model_name] = self._benchmark_onnx(model_path, test_images, num_runs)
                elif model_name == "openvino":
                    results[model_name] = self._benchmark_openvino(model_path, test_images, num_runs)
                elif model_name == "tensorrt":
                    results[model_name] = self._benchmark_tensorrt(model_path, test_images, num_runs)
                else:
                    # Original YOLO model
                    results[model_name] = self._benchmark_yolo(model_path, test_images, num_runs)

            except Exception as e:
                print(f"Failed to benchmark {model_name}: {e}")
                results[model_name] = {"error": str(e)}

        return results

    def _benchmark_tflite(self, model_path: str, test_images: List[str], num_runs: int) -> Dict:
        """Benchmark TFLite model"""
        from model_converter import TFLiteInference

        model = TFLiteInference(model_path)
        times = []

        for img_path in test_images[:min(len(test_images), 10)]:  # Limit test images
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Warmup
            for _ in range(5):
                model.predict(img)

            # Benchmark
            run_times = []
            for _ in range(num_runs // 10):
                start = time.time()
                model.predict(img)
                run_times.append(time.time() - start)

            times.extend(run_times)

        return self._calculate_stats(times, os.path.getsize(model_path))

    def _benchmark_onnx(self, model_path: str, test_images: List[str], num_runs: int) -> Dict:
        """Benchmark ONNX model"""
        if not ONNX_AVAILABLE:
            return {"error": "ONNX not available"}

        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        times = []

        for img_path in test_images[:min(len(test_images), 10)]:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 640))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img.transpose(2, 0, 1), 0)

            # Warmup
            for _ in range(5):
                session.run(None, {input_name: img})

            # Benchmark
            run_times = []
            for _ in range(num_runs // 10):
                start = time.time()
                session.run(None, {input_name: img})
                run_times.append(time.time() - start)

            times.extend(run_times)

        return self._calculate_stats(times, os.path.getsize(model_path))

    def _benchmark_yolo(self, model_path: str, test_images: List[str], num_runs: int) -> Dict:
        """Benchmark original YOLO model"""
        model = YOLO(model_path)
        times = []

        for img_path in test_images[:min(len(test_images), 10)]:
            img = cv2.imread(img_path)

            # Warmup
            for _ in range(5):
                model(img)

            # Benchmark
            run_times = []
            for _ in range(num_runs // 10):
                start = time.time()
                model(img)
                run_times.append(time.time() - start)

            times.extend(run_times)

        return self._calculate_stats(times, os.path.getsize(model_path))

    def _benchmark_openvino(self, model_dir: str, test_images: List[str], num_runs: int) -> Dict:
        """Benchmark OpenVINO model"""
        if not OPENVINO_AVAILABLE:
            return {"error": "OpenVINO not available"}

        # Implementation would go here
        return {"error": "OpenVINO benchmarking not implemented"}

    def _benchmark_tensorrt(self, model_path: str, test_images: List[str], num_runs: int) -> Dict:
        """Benchmark TensorRT model"""
        if not TENSORRT_AVAILABLE:
            return {"error": "TensorRT not available"}

        # Implementation would go here
        return {"error": "TensorRT benchmarking not implemented"}

    def _calculate_stats(self, times: List[float], model_size: int) -> Dict:
        """Calculate performance statistics"""
        times = np.array(times)
        return {
            "avg_time": float(np.mean(times)),
            "min_time": float(np.min(times)),
            "max_time": float(np.max(times)),
            "std_time": float(np.std(times)),
            "fps": float(1.0 / np.mean(times)),
            "model_size_mb": float(model_size / 1024 / 1024),
            "throughput": float(len(times) / np.sum(times))
        }

    def save_optimization_report(self, results: Dict, output_path: str):
        """Save optimization results to JSON report"""
        report = {
            "original_model": self.model_path,
            "optimization_results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": self._generate_summary(results)
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Optimization report saved to: {output_path}")

    def _generate_summary(self, results: Dict) -> Dict:
        """Generate optimization summary"""
        if not results:
            return {}

        # Find best performing model
        valid_results = {k: v for k, v in results.items() if "error" not in v}

        if not valid_results:
            return {"error": "No valid results"}

        best_fps = max(valid_results.items(), key=lambda x: x[1].get("fps", 0))
        smallest_size = min(valid_results.items(), key=lambda x: x[1].get("model_size_mb", float('inf')))

        return {
            "best_performance": {
                "model": best_fps[0],
                "fps": best_fps[1]["fps"],
                "avg_time": best_fps[1]["avg_time"]
            },
            "smallest_model": {
                "model": smallest_size[0],
                "size_mb": smallest_size[1]["model_size_mb"],
                "fps": smallest_size[1]["fps"]
            },
            "total_models": len(valid_results),
            "failed_optimizations": len(results) - len(valid_results)
        }


class AutoOptimizer:
    """Automatically choose best optimization strategy"""

    def __init__(self, model_path: str):
        self.optimizer = ModelOptimizer(model_path)

    def auto_optimize(
        self,
        target_fps: float = 30.0,
        max_model_size_mb: float = 100.0,
        test_images: List[str] = None,
        output_dir: str = "./auto_optimized"
    ) -> Dict:
        """
        Automatically find the best optimization for given constraints

        Args:
            target_fps: Minimum target FPS
            max_model_size_mb: Maximum model size in MB
            test_images: Images for benchmarking
            output_dir: Output directory
        """
        print("Starting automatic optimization...")

        # Try different optimization strategies
        strategies = [
            ("cpu", "fast"),
            ("cpu", "medium"),
            ("mobile", "medium"),
            ("edge", "medium"),
            ("gpu", "medium"),
        ]

        results = {}
        best_model = None
        best_score = -1

        for platform, level in strategies:
            try:
                print(f"Trying {platform} optimization with {level} level...")

                # Optimize
                models = self.optimizer.optimize_for_deployment(
                    target_platform=platform,
                    optimization_level=level,
                    output_dir=f"{output_dir}/{platform}_{level}"
                )

                # Benchmark if test images provided
                if test_images and models:
                    benchmark_results = self.optimizer.benchmark_models(models, test_images)

                    # Score models based on constraints
                    for model_name, stats in benchmark_results.items():
                        if "error" in stats:
                            continue

                        fps = stats.get("fps", 0)
                        size_mb = stats.get("model_size_mb", float('inf'))

                        # Calculate score (higher is better)
                        score = 0
                        if fps >= target_fps:
                            score += fps / target_fps  # FPS bonus
                        if size_mb <= max_model_size_mb:
                            score += (max_model_size_mb - size_mb) / max_model_size_mb  # Size bonus

                        if score > best_score:
                            best_score = score
                            best_model = {
                                "platform": platform,
                                "level": level,
                                "model_name": model_name,
                                "model_path": models[model_name],
                                "stats": stats,
                                "score": score
                            }

                    results[f"{platform}_{level}"] = benchmark_results

            except Exception as e:
                print(f"Failed optimization for {platform}_{level}: {e}")
                results[f"{platform}_{level}"] = {"error": str(e)}

        # Save results
        report_path = f"{output_dir}/auto_optimization_report.json"
        self.optimizer.save_optimization_report(results, report_path)

        return {
            "best_model": best_model,
            "all_results": results,
            "report_path": report_path
        }