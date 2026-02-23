import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List
import warnings

import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO

try:
    import tensorflow as tf
    import onnx
    import onnx2tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow/ONNX not available. Install with: pip install tensorflow onnx onnx2tf")

try:
    import pytorch_lightning as pl
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    print("PyTorch Lightning not available. Install with: pip install pytorch-lightning")


class YOLOToTFLiteConverter:
    """Convert YOLO models to TensorFlow Lite for mobile/edge deployment"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.yolo_model = YOLO(model_path)

    def convert_to_tflite(
        self,
        output_path: str,
        quantization: str = "float16",
        optimize_for_size: bool = True,
        representative_dataset: Optional[List[np.ndarray]] = None
    ) -> str:
        """
        Convert YOLO model to TensorFlow Lite

        Args:
            output_path: Path for output .tflite file
            quantization: 'float16', 'int8', 'dynamic', or 'none'
            optimize_for_size: Whether to optimize for model size
            representative_dataset: Sample data for int8 quantization
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for TFLite conversion")

        print(f"Converting {self.model_path} to TensorFlow Lite...")

        # Step 1: Export YOLO to ONNX
        onnx_path = output_path.replace('.tflite', '.onnx')
        print("Step 1: Exporting to ONNX...")
        self.yolo_model.export(format='onnx', dynamic=False, simplify=True)

        # Find the exported ONNX file
        model_dir = Path(self.model_path).parent
        onnx_files = list(model_dir.glob("*.onnx"))
        if onnx_files:
            onnx_path = str(onnx_files[-1])  # Use most recent

        # Step 2: Convert ONNX to TensorFlow
        print("Step 2: Converting ONNX to TensorFlow...")
        tf_model_dir = output_path.replace('.tflite', '_tf_model')

        try:
            # Use onnx2tf for conversion
            import subprocess
            cmd = [
                'onnx2tf',
                '-i', onnx_path,
                '-o', tf_model_dir,
                '--non_verbose'
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        except Exception as e:
            print(f"onnx2tf failed, trying alternative method: {e}")
            return self._convert_via_torch_export(output_path, quantization)

        # Step 3: Convert TensorFlow to TFLite
        print("Step 3: Converting to TensorFlow Lite...")
        return self._tf_to_tflite(tf_model_dir, output_path, quantization, optimize_for_size, representative_dataset)

    def _convert_via_torch_export(self, output_path: str, quantization: str) -> str:
        """Alternative conversion method using PyTorch direct export"""
        print("Using PyTorch direct export method...")

        # Export directly to TensorFlow Lite via PyTorch
        try:
            # Use YOLO's built-in TFLite export if available
            tflite_path = self.yolo_model.export(format='tflite', int8=quantization=='int8')

            # Copy to desired output path
            if tflite_path != output_path:
                import shutil
                shutil.copy(tflite_path, output_path)

            return output_path

        except Exception as e:
            print(f"Direct export failed: {e}")
            raise

    def _tf_to_tflite(
        self,
        tf_model_dir: str,
        output_path: str,
        quantization: str,
        optimize_for_size: bool,
        representative_dataset: Optional[List[np.ndarray]]
    ) -> str:
        """Convert TensorFlow model to TFLite"""

        # Load TensorFlow model
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)

        # Set optimization flags
        if optimize_for_size:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Configure quantization
        if quantization == "float16":
            converter.target_spec.supported_types = [tf.float16]
        elif quantization == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if representative_dataset:
                converter.representative_dataset = lambda: self._representative_data_gen(representative_dataset)
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
        elif quantization == "dynamic":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]

        # Convert
        try:
            tflite_model = converter.convert()

            # Save model
            with open(output_path, 'wb') as f:
                f.write(tflite_model)

            print(f"TFLite model saved to: {output_path}")

            # Print model info
            self._print_model_info(output_path)

            return output_path

        except Exception as e:
            print(f"TFLite conversion failed: {e}")
            # Try with relaxed settings
            print("Trying with relaxed conversion settings...")
            converter.allow_custom_ops = True
            converter.experimental_new_converter = True

            tflite_model = converter.convert()
            with open(output_path, 'wb') as f:
                f.write(tflite_model)

            return output_path

    def _representative_data_gen(self, dataset: List[np.ndarray]):
        """Generate representative data for quantization"""
        for data in dataset:
            # Ensure correct shape and type
            if len(data.shape) == 3:
                data = np.expand_dims(data, 0)
            yield [data.astype(np.float32)]

    def _print_model_info(self, tflite_path: str):
        """Print information about the converted model"""
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"\n=== TFLite Model Info ===")
        print(f"Model size: {os.path.getsize(tflite_path) / 1024 / 1024:.2f} MB")
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Input dtype: {input_details[0]['dtype']}")
        print(f"Output shapes: {[out['shape'] for out in output_details]}")
        print(f"Number of tensors: {len(interpreter.get_tensor_details())}")

    def benchmark_tflite(self, tflite_path: str, num_runs: int = 100) -> dict:
        """Benchmark TFLite model performance"""
        import time

        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Create dummy input
        input_shape = input_details[0]['shape']
        dummy_input = np.random.random(input_shape).astype(input_details[0]['dtype'])

        # Warm up
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            times.append(time.time() - start)

        return {
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times),
            'fps': 1.0 / np.mean(times)
        }


class TFLiteInference:
    """Fast inference using TensorFlow Lite models"""

    def __init__(self, tflite_path: str, num_threads: int = 4):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for TFLite inference")

        self.interpreter = tf.lite.Interpreter(
            model_path=tflite_path,
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]['shape']
        self.input_dtype = self.input_details[0]['dtype']

        print(f"TFLite model loaded: {tflite_path}")
        print(f"Input shape: {self.input_shape}")
        print(f"Input dtype: {self.input_dtype}")

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Run inference on image"""
        # Preprocess image
        input_tensor = self._preprocess_image(image)

        # Set input
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)

        # Run inference
        self.interpreter.invoke()

        # Get outputs
        outputs = []
        for output_detail in self.output_details:
            output = self.interpreter.get_tensor(output_detail['index'])
            outputs.append(output)

        return tuple(outputs)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input"""
        import cv2

        # Get target shape
        target_h, target_w = self.input_shape[1], self.input_shape[2]

        # Resize image
        resized = cv2.resize(image, (target_w, target_h))

        # Normalize if needed
        if self.input_dtype == np.float32:
            processed = resized.astype(np.float32) / 255.0
        else:
            processed = resized.astype(self.input_dtype)

        # Add batch dimension
        if len(processed.shape) == 3:
            processed = np.expand_dims(processed, 0)

        return processed


def create_representative_dataset(data_dir: str, num_samples: int = 100) -> List[np.ndarray]:
    """Create representative dataset for quantization"""
    import cv2
    from pathlib import Path

    data_path = Path(data_dir)
    image_files = list(data_path.glob("*.jpg")) + list(data_path.glob("*.png"))

    if len(image_files) == 0:
        print(f"No images found in {data_dir}")
        return []

    # Sample random images
    import random
    sampled_files = random.sample(image_files, min(num_samples, len(image_files)))

    dataset = []
    for img_path in sampled_files:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))  # YOLO input size
        img = img.astype(np.float32) / 255.0
        dataset.append(img)

    return dataset


def compare_models(original_path: str, tflite_path: str, test_images: List[str]) -> dict:
    """Compare performance between original and TFLite models"""
    import time
    import cv2

    # Load models
    yolo_model = YOLO(original_path)
    tflite_model = TFLiteInference(tflite_path)

    results = {
        'original': {'times': [], 'accuracy': []},
        'tflite': {'times': [], 'accuracy': []}
    }

    for img_path in test_images:
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Test original model
        start = time.time()
        yolo_results = yolo_model(img_rgb)
        original_time = time.time() - start
        results['original']['times'].append(original_time)

        # Test TFLite model
        start = time.time()
        tflite_outputs = tflite_model.predict(img_rgb)
        tflite_time = time.time() - start
        results['tflite']['times'].append(tflite_time)

    # Calculate statistics
    original_avg = np.mean(results['original']['times'])
    tflite_avg = np.mean(results['tflite']['times'])

    speedup = original_avg / tflite_avg

    return {
        'original_avg_time': original_avg,
        'tflite_avg_time': tflite_avg,
        'speedup': speedup,
        'original_fps': 1.0 / original_avg,
        'tflite_fps': 1.0 / tflite_avg
    }