import os
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple, List, Generator

import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

try:
    import tensorflow as tf
    import onnx
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

    # ------------------------------------------------------------------
    # NPU-friendly INT8 변환 (메인 경로)
    # ------------------------------------------------------------------

    def convert_npu_int8(
        self,
        output_path: str,
        input_size: int = 640,
        data_dir: Optional[str] = None,
        representative_dataset: Optional[List[np.ndarray]] = None,
        num_samples: int = 300,
    ) -> str:
        """
        NPU delegate-friendly INT8 TFLite 변환.

        규칙:
        - static input shape: 1 x input_size x input_size x 3 (NHWC)
        - NMS 제외: backbone + neck + head raw output만 포함
        - SELECT_TF_OPS 미사용: TFLITE_BUILTINS_INT8 전용
        - full INT8 quantization: input uint8 / output float32
        - allow_custom_ops = False (명시적 금지)
        - 전처리(resize/letterbox/normalize)는 앱 코드 담당

        Args:
            output_path: 출력 .tflite 경로
            input_size: 모델 입력 해상도 (정사각형, default 640)
            data_dir: representative dataset용 이미지 디렉토리
            representative_dataset: 외부에서 직접 전달하는 데이터셋
                                     shape: (num_samples, input_size, input_size, 3), float32 [0,1]
            num_samples: data_dir 사용 시 샘플 수 (최소 200 권장)
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required. pip install tensorflow onnx onnx2tf")

        print(f"[NPU INT8] {self.model_path} → {output_path}")

        # representative dataset 준비
        if representative_dataset is None:
            if data_dir is None:
                raise ValueError("data_dir 또는 representative_dataset 중 하나는 필수입니다.")
            representative_dataset = create_representative_dataset(
                data_dir, num_samples=num_samples, input_size=input_size
            )

        if len(representative_dataset) < 100:
            print(f"[경고] representative dataset이 {len(representative_dataset)}장으로 부족합니다. "
                  f"최소 200장 이상 권장.")

        # Step 1: YOLO → ONNX (static shape, NMS 제외 시도)
        onnx_path = self._export_onnx_no_nms(input_size)

        # Step 1.5: E2E 모델의 경우 NMS를 ONNX에서 직접 제거
        # (YOLO11 E2E는 nms=False가 무시되므로 수동 strip 필요)
        onnx_path = self._strip_nms_from_onnx(onnx_path)

        # Step 2: ONNX → TF SavedModel (explicit input shape)
        tf_model_dir = output_path.replace('.tflite', '_saved_model')
        self._onnx_to_saved_model(onnx_path, tf_model_dir, input_size)

        # Step 3: TF SavedModel → TFLite INT8
        self._saved_model_to_int8_tflite(
            tf_model_dir, output_path, representative_dataset, input_size
        )

        self._print_model_info(output_path)
        return output_path

    def _export_onnx_no_nms(self, input_size: int) -> str:
        """
        YOLO를 NMS 없는 ONNX로 export.

        ultralytics는 기본적으로 ONNX export 시 NMS 포함 여부를
        모델 구조에 따라 자동 결정한다. simplify=True + static shape로
        export하면 대부분 Conv/BN/Act 중심의 그래프가 된다.
        """
        print("[Step 1] YOLO → ONNX (static shape, no NMS)...")

        export_result = self.yolo_model.export(
            format='onnx',
            imgsz=input_size,
            dynamic=False,       # static shape 고정
            simplify=True,       # onnx-simplifier 적용
            opset=17,            # TF/TFLite 변환 호환 opset
            nms=False,           # NMS 제외 (ultralytics 최신 버전)
        )

        # ultralytics는 export 결과 경로를 문자열로 반환
        onnx_path = str(export_result) if export_result else None

        # fallback: 모델 파일 위치에서 탐색
        if not onnx_path or not Path(onnx_path).exists():
            model_dir = Path(self.model_path).parent
            candidates = sorted(model_dir.glob("*.onnx"), key=lambda p: p.stat().st_mtime)
            if not candidates:
                raise FileNotFoundError("ONNX export 파일을 찾을 수 없습니다.")
            onnx_path = str(candidates[-1])

        print(f"  ONNX 저장: {onnx_path}")

        # NMS 관련 노드 잔존 여부 경고
        self._warn_if_nms_present(onnx_path)

        return onnx_path

    def _strip_nms_from_onnx(self, onnx_path: str) -> str:
        """
        E2E YOLO 모델에서 NMS 부분을 제거하고 raw output ONNX를 반환.

        YOLO11 E2E 모델은 NMS가 아키텍처에 내장되어 nms=False가 무시된다.
        NMS 내부의 TopK/GatherElements는 INT8 quantization을 견디지 못하므로,
        NMS 직전에서 그래프를 자른다.

        INT8 per-tensor quantization에서 bbox(0~688)와 class score(0~1)가
        같은 tensor에 있으면 score가 0으로 뭉개지므로, output을 2개로 분리:
        - boxes: (1, 4, num_anchors)  — bbox 좌표
        - scores: (1, nc, num_anchors) — class scores (post-sigmoid)

        Returns:
            NMS가 제거된 ONNX 파일 경로 (원본이 NMS 없으면 원본 그대로 반환)
        """
        from onnx import helper, TensorProto, shape_inference

        model = onnx.load(onnx_path)
        nms_op_types = {'TopK', 'GatherElements', 'NonMaxSuppression', 'GatherND'}
        nms_nodes = [n for n in model.graph.node if n.op_type in nms_op_types]

        if not nms_nodes:
            print("  NMS op 없음 — 원본 ONNX 사용")
            return onnx_path

        # shape inference로 중간 텐서 shape 확인
        inferred = shape_inference.infer_shapes(model)
        shape_map = {}
        for vi in inferred.graph.value_info:
            dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            shape_map[vi.name] = dims

        # NMS 직전 Concat (boxes+scores → combined) 찾기
        # 이 Concat의 입력이 boxes와 scores 텐서
        boxes_name, scores_name = None, None
        for node in model.graph.node:
            if node.op_type == 'Concat':
                for out_name in node.output:
                    shape = shape_map.get(out_name, [])
                    if (len(shape) == 3 and shape[0] == 1
                            and shape[2] > 1000
                            and 4 < shape[1] < 100):
                        # 이 Concat의 입력 = [boxes_tensor, scores_tensor]
                        for inp in node.input:
                            inp_shape = shape_map.get(inp, [])
                            if len(inp_shape) == 3 and inp_shape[0] == 1:
                                if inp_shape[1] == 4:
                                    boxes_name = inp
                                elif inp_shape[1] > 0 and inp_shape[1] != 4:
                                    scores_name = inp

        if not boxes_name or not scores_name:
            print("  [경고] boxes/scores 텐서를 찾지 못함 — 원본 ONNX 사용")
            return onnx_path

        boxes_shape = shape_map[boxes_name]
        scores_shape = shape_map[scores_name]
        print(f"  NMS 제거: output 2개로 분리")
        print(f"    boxes:  {boxes_name} {boxes_shape}")
        print(f"    scores: {scores_name} {scores_shape}")

        # boxes와 scores에 도달 가능한 노드만 유지
        needed_outputs = {boxes_name, scores_name}
        needed_nodes = []
        for node in reversed(model.graph.node):
            if any(o in needed_outputs for o in node.output):
                needed_nodes.append(node)
                needed_outputs.update(node.input)

        needed_nodes.reverse()

        # 새 그래프: output 2개 (boxes, scores 분리)
        boxes_vi = helper.make_tensor_value_info(
            boxes_name, TensorProto.FLOAT, boxes_shape
        )
        scores_vi = helper.make_tensor_value_info(
            scores_name, TensorProto.FLOAT, scores_shape
        )
        new_graph = helper.make_graph(
            needed_nodes,
            model.graph.name,
            list(model.graph.input),
            [boxes_vi, scores_vi],
            initializer=list(model.graph.initializer),
        )
        new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
        new_model.ir_version = model.ir_version

        stripped_path = onnx_path.replace('.onnx', '_raw.onnx')
        onnx.save(new_model, stripped_path)

        # 검증
        stripped = onnx.load(stripped_path)
        remaining_nms = [n.op_type for n in stripped.graph.node if n.op_type in nms_op_types]
        print(f"  저장: {stripped_path}")
        print(f"  outputs: {len(stripped.graph.output)}개, 잔여 NMS ops: {remaining_nms if remaining_nms else '없음'}")

        return stripped_path

    def _warn_if_nms_present(self, onnx_path: str):
        """ONNX 그래프에 NMS 관련 op가 있으면 경고 출력."""
        try:
            model = onnx.load(onnx_path)
            nms_ops = {'NonMaxSuppression', 'TopK', 'GatherND'}
            found = [n.op_type for n in model.graph.node if n.op_type in nms_ops]
            if found:
                print(f"  [경고] NMS 관련 op 발견: {set(found)}")
                print(f"         이 op들은 NPU delegate에서 CPU fallback 될 수 있습니다.")
                print(f"         ultralytics 버전을 확인하거나 수동으로 NMS 노드를 제거하세요.")
        except Exception:
            pass  # onnx 파싱 실패는 무시

    def _onnx_to_saved_model(self, onnx_path: str, tf_model_dir: str, input_size: int):
        """
        onnx2tf로 ONNX → TF SavedModel 변환.

        --input_shape으로 static shape을 명시해 동적 차원을 제거한다.
        NHWC: batch=1, H=input_size, W=input_size, C=3
        """
        print("[Step 2] ONNX → TF SavedModel...")

        cmd = [
            'onnx2tf',
            '-i', onnx_path,
            '-o', tf_model_dir,
            '-ois', f'images:1,3,{input_size},{input_size}',  # NCHW (ONNX 원본 format)
            '-n',                           # non_verbose
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  [onnx2tf stdout]\n{result.stdout}")
            print(f"  [onnx2tf stderr]\n{result.stderr}")
            raise RuntimeError("onnx2tf 변환 실패. 위 오류를 확인하세요.")

        if not Path(tf_model_dir).exists():
            raise FileNotFoundError(f"SavedModel 디렉토리 생성 실패: {tf_model_dir}")

        print(f"  SavedModel 저장: {tf_model_dir}")

    def _saved_model_to_int8_tflite(
        self,
        tf_model_dir: str,
        output_path: str,
        representative_dataset: List[np.ndarray],
        input_size: int,
    ):
        """
        TF SavedModel → INT8 TFLite.

        NPU delegate 최대 호환을 위한 설정:
        - TFLITE_BUILTINS_INT8 전용 (SELECT_TF_OPS 없음)
        - input: uint8 (NPU는 uint8 입력 선호)
        - output: float32 (좌표 정밀도 유지)
        - allow_custom_ops = False
        """
        print("[Step 3] SavedModel → INT8 TFLite...")

        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)

        # Full INT8 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _make_representative_gen(
            representative_dataset, input_size
        )

        # NPU-friendly: TFLITE_BUILTINS_INT8 전용, SELECT_TF_OPS 금지
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        # input uint8: 앱에서 uint8 이미지를 그대로 전달 가능
        converter.inference_input_type = tf.uint8
        # output float32: box 좌표/score의 정밀도 유지
        converter.inference_output_type = tf.float32

        # custom op, fallback 명시적 금지
        converter.allow_custom_ops = False
        converter.experimental_new_converter = True

        try:
            tflite_model = converter.convert()
        except Exception as e:
            raise RuntimeError(
                f"INT8 변환 실패: {e}\n"
                "힌트: representative dataset이 충분한지 확인하고,\n"
                "      onnx2tf 변환 시 남은 custom op을 확인하세요.\n"
                "      SELECT_TF_OPS fallback은 의도적으로 차단됩니다."
            ) from e

        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        print(f"  TFLite 저장: {output_path} ({os.path.getsize(output_path)/1024/1024:.1f} MB)")

    # ------------------------------------------------------------------
    # 기존 범용 변환 (float16 / dynamic 용도)
    # ------------------------------------------------------------------

    def convert_to_tflite(
        self,
        output_path: str,
        quantization: str = "float16",
        optimize_for_size: bool = True,
        representative_dataset: Optional[List[np.ndarray]] = None
    ) -> str:
        """
        범용 TFLite 변환 (float16 / dynamic).

        NPU INT8이 목표라면 convert_npu_int8()을 사용하세요.
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for TFLite conversion")

        print(f"Converting {self.model_path} to TensorFlow Lite ({quantization})...")

        print("Step 1: Exporting to ONNX...")
        self.yolo_model.export(format='onnx', dynamic=False, simplify=True)

        model_dir = Path(self.model_path).parent
        onnx_files = sorted(model_dir.glob("*.onnx"), key=lambda p: p.stat().st_mtime)
        if not onnx_files:
            raise FileNotFoundError("ONNX export 파일을 찾을 수 없습니다.")
        onnx_path = str(onnx_files[-1])

        print("Step 2: Converting ONNX to TensorFlow...")
        tf_model_dir = output_path.replace('.tflite', '_tf_model')

        cmd = ['onnx2tf', '-i', onnx_path, '-o', tf_model_dir, '--non_verbose']
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"onnx2tf failed: {result.stderr}")

        print("Step 3: Converting to TensorFlow Lite...")
        return self._tf_to_tflite(tf_model_dir, output_path, quantization, optimize_for_size, representative_dataset)

    def _tf_to_tflite(
        self,
        tf_model_dir: str,
        output_path: str,
        quantization: str,
        optimize_for_size: bool,
        representative_dataset: Optional[List[np.ndarray]]
    ) -> str:
        """범용 TFLite 변환 내부 구현."""
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)

        if optimize_for_size:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if quantization == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif quantization == "int8":
            # 주의: NPU 목표라면 convert_npu_int8() 사용 권장
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if representative_dataset:
                converter.representative_dataset = lambda: (
                    [np.expand_dims(d, 0).astype(np.float32)] if d.ndim == 3
                    else [d.astype(np.float32)]
                    for d in representative_dataset
                )
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.float32  # float32로 수정
        elif quantization == "dynamic":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # dynamic은 범용 배포용 — SELECT_TF_OPS 허용
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]

        tflite_model = converter.convert()
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        print(f"TFLite model saved to: {output_path}")
        self._print_model_info(output_path)
        return output_path

    def _print_model_info(self, tflite_path: str):
        """변환된 TFLite 모델 정보 출력."""
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"\n=== TFLite Model Info ===")
        print(f"Model size: {os.path.getsize(tflite_path) / 1024 / 1024:.2f} MB")
        print(f"Input  shape: {input_details[0]['shape']}  dtype: {input_details[0]['dtype'].__name__}")
        for i, out in enumerate(output_details):
            print(f"Output[{i}] shape: {out['shape']}  dtype: {out['dtype'].__name__}")
        print(f"Tensors total: {len(interpreter.get_tensor_details())}")

    def verify_npu_coverage(self, tflite_path: str) -> dict:
        """
        TFLite 모델에서 NPU delegate로 올라가지 못할 op 목록을 분석.

        실제 delegate 적용 여부는 타겟 디바이스에서만 확인 가능하지만,
        TFLITE_BUILTINS에 없는 op가 있으면 CPU fallback 가능성이 높다.
        """
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        tensor_details = interpreter.get_tensor_details()
        all_ops = set()
        custom_ops = []

        for td in tensor_details:
            if td.get('quantization_parameters'):
                scales = td['quantization_parameters'].get('scales', [])
                if len(scales) == 0:
                    custom_ops.append(td['name'])

        print(f"\n=== NPU Coverage 분석 ===")
        print(f"전체 텐서 수: {len(tensor_details)}")
        print(f"미정량화 텐서 (CPU fallback 가능): {len(custom_ops)}")
        if custom_ops:
            print(f"  대상: {custom_ops[:10]}{'...' if len(custom_ops) > 10 else ''}")
        print("주의: 정확한 delegate coverage는 타겟 SoC에서 직접 확인하세요.")

        return {
            'total_tensors': len(tensor_details),
            'non_quantized_tensors': len(custom_ops),
            'non_quantized_names': custom_ops,
        }

    def benchmark_tflite(self, tflite_path: str, num_runs: int = 100) -> dict:
        """TFLite 모델 CPU 추론 벤치마크."""
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        dummy_input = np.zeros(input_shape, dtype=input_details[0]['dtype'])

        # warmup
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()

        times = []
        for _ in range(num_runs):
            t0 = time.time()
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            times.append(time.time() - t0)

        return {
            'avg_ms': float(np.mean(times) * 1000),
            'min_ms': float(np.min(times) * 1000),
            'max_ms': float(np.max(times) * 1000),
            'fps': float(1.0 / np.mean(times)),
        }


# ------------------------------------------------------------------
# representative dataset 생성 유틸
# ------------------------------------------------------------------

def _make_representative_gen(
    dataset: List[np.ndarray],
    input_size: int,
) -> callable:
    """
    TFLiteConverter.representative_dataset에 전달할 generator factory.

    dataset 원소: float32 [0,1], shape (H, W, 3) 또는 (1, H, W, 3)
    yield: [uint8 tensor, shape (1, input_size, input_size, 3)]

    INT8 calibration은 float32 [0,1] 입력으로 수행한다.
    (inference_input_type=uint8 설정과 별개로 calibration 자체는 float)
    """
    import cv2

    def gen():
        for data in dataset:
            if data.ndim == 3:
                data = data[np.newaxis]  # (1, H, W, 3)
            h, w = data.shape[1], data.shape[2]
            if h != input_size or w != input_size:
                resized = cv2.resize(
                    data[0], (input_size, input_size), interpolation=cv2.INTER_LINEAR
                )
                data = resized[np.newaxis]
            yield [data.astype(np.float32)]

    return gen


def create_representative_dataset(
    data_dir: str,
    num_samples: int = 300,
    input_size: int = 640,
) -> List[np.ndarray]:
    """
    INT8 calibration용 representative dataset 생성.

    - 실제 도메인 이미지를 사용해야 quantization 품질이 좋음
    - num_samples >= 200 권장 (클수록 calibration 정확도 향상)
    - augmentation으로 다양성 확보 (밝기 변화, 좌우 반전)
    - 반환 shape: List of (input_size, input_size, 3), float32 [0, 1]
    """
    import cv2

    data_path = Path(data_dir)
    image_files = (
        list(data_path.rglob("*.jpg"))
        + list(data_path.rglob("*.jpeg"))
        + list(data_path.rglob("*.png"))
    )

    if not image_files:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {data_dir}")

    # 원본 이미지에서 샘플링 후 augmentation으로 수 확장
    base_count = min(num_samples, len(image_files))
    sampled_files = random.sample(image_files, base_count)

    dataset = []
    for img_path in sampled_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        base = img.astype(np.float32) / 255.0
        dataset.append(base)

        # augmentation: 밝기 변화 ±20%
        if len(dataset) < num_samples:
            factor = random.uniform(0.8, 1.2)
            dataset.append(np.clip(base * factor, 0.0, 1.0))

        # augmentation: 좌우 반전
        if len(dataset) < num_samples:
            dataset.append(np.fliplr(base).copy())

        if len(dataset) >= num_samples:
            break

    print(f"Representative dataset: {len(dataset)}장 ({data_dir})")
    return dataset[:num_samples]


# ------------------------------------------------------------------
# TFLite 추론 클래스 (앱 코드 참고용 — 전처리 분리)
# ------------------------------------------------------------------

class TFLiteInference:
    """
    TFLite 추론 래퍼.

    NPU INT8 모델 기준:
    - 입력: uint8, (1, H, W, 3), letterbox/normalize는 앱에서 수행
    - 출력: float32 raw predictions (NMS 미포함)
    - NMS는 호출측에서 직접 처리
    """

    def __init__(self, tflite_path: str, num_threads: int = 4):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for TFLite inference")

        self.interpreter = tf.lite.Interpreter(
            model_path=tflite_path,
            num_threads=num_threads,
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]['shape']    # (1, H, W, 3)
        self.input_dtype = self.input_details[0]['dtype']    # uint8 or float32
        self.input_h = int(self.input_shape[1])
        self.input_w = int(self.input_shape[2])

        print(f"TFLite loaded: {tflite_path}")
        print(f"  input : {self.input_shape}  {self.input_dtype.__name__}")
        for i, od in enumerate(self.output_details):
            print(f"  output[{i}]: {od['shape']}  {od['dtype'].__name__}")

    def infer_raw(self, preprocessed: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        전처리가 완료된 텐서로 추론.

        Args:
            preprocessed: (1, H, W, 3), uint8 또는 float32 (모델 dtype에 맞게)
        Returns:
            raw output tensors (NMS 미포함) — 호출측에서 NMS 처리
        """
        self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed)
        self.interpreter.invoke()
        return tuple(
            self.interpreter.get_tensor(od['index']) for od in self.output_details
        )

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        BGR 이미지 → raw output.

        단순 리사이즈만 수행 (letterbox/normalize는 필요 시 앱 코드로 이전).
        NPU INT8 모델: uint8 입력으로 변환.
        """
        import cv2
        resized = cv2.resize(image, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        if self.input_dtype == np.uint8:
            tensor = rgb.astype(np.uint8)[np.newaxis]
        else:
            tensor = (rgb.astype(np.float32) / 255.0)[np.newaxis]

        return self.infer_raw(tensor)


# ------------------------------------------------------------------
# 모델 비교 유틸
# ------------------------------------------------------------------

def compare_models(original_path: str, tflite_path: str, test_images: List[str]) -> dict:
    """원본 YOLO와 TFLite 모델 속도 비교."""
    import cv2

    yolo_model = YOLO(original_path)
    tflite_model = TFLiteInference(tflite_path)

    orig_times, tflite_times = [], []

    for img_path in test_images:
        img = cv2.imread(img_path)

        t0 = time.time()
        yolo_model(img)
        orig_times.append(time.time() - t0)

        t0 = time.time()
        tflite_model.predict(img)
        tflite_times.append(time.time() - t0)

    orig_avg = float(np.mean(orig_times))
    tflite_avg = float(np.mean(tflite_times))

    return {
        'original_avg_ms': orig_avg * 1000,
        'tflite_avg_ms': tflite_avg * 1000,
        'speedup': orig_avg / tflite_avg,
        'original_fps': 1.0 / orig_avg,
        'tflite_fps': 1.0 / tflite_avg,
    }