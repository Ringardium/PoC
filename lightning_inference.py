import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

try:
    import pytorch_lightning as pl
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False

from lightning_training import YOLOLightningModule


class LightningYOLOInference:
    """Fast inference using PyTorch Lightning trained YOLO models"""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        half_precision: bool = True,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        if not LIGHTNING_AVAILABLE:
            raise ImportError("PyTorch Lightning is required")

        self.checkpoint_path = checkpoint_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.half_precision = half_precision and torch.cuda.is_available()

        # Load model
        self.model = self._load_model()
        self.model.eval()

        # Model info
        self.input_size = (640, 640)  # Default YOLO input size
        self.num_classes = self.model.num_classes

        print(f"Lightning YOLO model loaded on {self.device}")
        print(f"Input size: {self.input_size}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Half precision: {self.half_precision}")

    def _load_model(self) -> YOLOLightningModule:
        """Load Lightning model from checkpoint"""
        try:
            # Load from Lightning checkpoint
            model = YOLOLightningModule.load_from_checkpoint(self.checkpoint_path)
        except Exception:
            # Try loading as regular PyTorch checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

            if 'model' in checkpoint:
                # Custom checkpoint format
                model = YOLOLightningModule()
                model.net.load_state_dict(checkpoint['model'])
            else:
                # Direct state dict
                model = YOLOLightningModule()
                model.load_state_dict(checkpoint)

        model.to(self.device)

        if self.half_precision:
            model.half()

        return model

    def predict(
        self,
        image: np.ndarray,
        return_boxes: bool = True,
        return_masks: bool = False
    ) -> Dict[str, Any]:
        """
        Run inference on image

        Args:
            image: Input image (BGR format)
            return_boxes: Whether to return bounding boxes
            return_masks: Whether to return segmentation masks

        Returns:
            Dictionary with predictions
        """
        # Preprocess
        input_tensor, scale_factor, padding = self._preprocess_image(image)

        # Inference
        with torch.no_grad():
            predictions = self.model(input_tensor)

        # Post-process
        results = self._postprocess_predictions(
            predictions, image.shape, scale_factor, padding, return_boxes, return_masks
        )

        return results

    def predict_batch(
        self,
        images: List[np.ndarray],
        batch_size: int = 4
    ) -> List[Dict[str, Any]]:
        """Run batch inference on multiple images"""
        results = []

        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_tensors = []
            batch_info = []

            # Preprocess batch
            for img in batch_images:
                tensor, scale_factor, padding = self._preprocess_image(img)
                batch_tensors.append(tensor.squeeze(0))
                batch_info.append((img.shape, scale_factor, padding))

            # Stack tensors
            batch_tensor = torch.stack(batch_tensors)

            # Inference
            with torch.no_grad():
                batch_predictions = self.model(batch_tensor)

            # Post-process each image
            for j, (predictions, (img_shape, scale_factor, padding)) in enumerate(zip(batch_predictions, batch_info)):
                result = self._postprocess_predictions(
                    [predictions], img_shape, scale_factor, padding
                )
                results.append(result)

        return results

    def _preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, float, Tuple[int, int]]:
        """Preprocess image for model input"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Calculate scale factor and padding
        h, w = image_rgb.shape[:2]
        target_h, target_w = self.input_size

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize
        resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target size
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2

        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # Convert to tensor
        tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)

        if self.half_precision:
            tensor = tensor.half()

        return tensor, scale, (pad_left, pad_top)

    def _postprocess_predictions(
        self,
        predictions: List[torch.Tensor],
        original_shape: Tuple[int, int, int],
        scale_factor: float,
        padding: Tuple[int, int],
        return_boxes: bool = True,
        return_masks: bool = False
    ) -> Dict[str, Any]:
        """Post-process model predictions"""
        if isinstance(predictions, torch.Tensor):
            predictions = [predictions]

        results = {
            "boxes": [],
            "scores": [],
            "classes": [],
            "masks": [] if return_masks else None
        }

        for pred in predictions:
            if pred is None or pred.size(0) == 0:
                continue

            # Handle different prediction formats
            if len(pred.shape) == 3:  # [batch, detections, features]
                pred = pred.squeeze(0)  # Remove batch dimension

            if pred.size(-1) < 6:  # Not enough features for box + conf + class
                continue

            # Extract boxes, confidence, and classes
            boxes = pred[:, :4]  # [x1, y1, x2, y2] or [cx, cy, w, h]
            confidence = pred[:, 4]
            class_scores = pred[:, 5:]

            # Apply confidence threshold
            conf_mask = confidence > self.confidence_threshold
            boxes = boxes[conf_mask]
            confidence = confidence[conf_mask]
            class_scores = class_scores[conf_mask]

            if boxes.size(0) == 0:
                continue

            # Get class predictions
            class_confidence, class_ids = torch.max(class_scores, dim=1)
            final_confidence = confidence * class_confidence

            # Convert boxes to x1, y1, x2, y2 format if needed
            if self._is_center_format(boxes):
                boxes = self._center_to_corners(boxes)

            # Apply NMS
            keep_indices = self._apply_nms(boxes, final_confidence)

            if len(keep_indices) == 0:
                continue

            # Select final detections
            final_boxes = boxes[keep_indices]
            final_scores = final_confidence[keep_indices]
            final_classes = class_ids[keep_indices]

            # Scale back to original image size
            final_boxes = self._scale_boxes_to_original(
                final_boxes, original_shape, scale_factor, padding
            )

            results["boxes"].extend(final_boxes.cpu().numpy())
            results["scores"].extend(final_scores.cpu().numpy())
            results["classes"].extend(final_classes.cpu().numpy())

        return results

    def _is_center_format(self, boxes: torch.Tensor) -> bool:
        """Check if boxes are in center format (cx, cy, w, h)"""
        # Simple heuristic: if all width/height values are positive, likely center format
        if boxes.size(-1) >= 4:
            w_h = boxes[:, 2:4]
            return torch.all(w_h > 0)
        return False

    def _center_to_corners(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert center format to corner format"""
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)

    def _apply_nms(self, boxes: torch.Tensor, scores: torch.Tensor) -> List[int]:
        """Apply Non-Maximum Suppression"""
        if boxes.size(0) == 0:
            return []

        # Use torchvision NMS if available
        try:
            from torchvision.ops import nms
            keep = nms(boxes, scores, self.iou_threshold)
            return keep.tolist()
        except ImportError:
            # Fallback to custom NMS
            return self._custom_nms(boxes, scores)

    def _custom_nms(self, boxes: torch.Tensor, scores: torch.Tensor) -> List[int]:
        """Custom NMS implementation"""
        # Sort by score
        sorted_indices = torch.argsort(scores, descending=True)
        keep = []

        while len(sorted_indices) > 0:
            # Take highest scoring box
            current_idx = sorted_indices[0]
            keep.append(current_idx.item())

            if len(sorted_indices) == 1:
                break

            # Calculate IoU with remaining boxes
            current_box = boxes[current_idx].unsqueeze(0)
            remaining_boxes = boxes[sorted_indices[1:]]

            ious = self._calculate_iou(current_box, remaining_boxes)

            # Remove boxes with high IoU
            mask = ious < self.iou_threshold
            sorted_indices = sorted_indices[1:][mask]

        return keep

    def _calculate_iou(self, box1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between one box and multiple boxes"""
        # Intersection
        x1 = torch.max(box1[:, 0], boxes2[:, 0])
        y1 = torch.max(box1[:, 1], boxes2[:, 1])
        x2 = torch.min(box1[:, 2], boxes2[:, 2])
        y2 = torch.min(box1[:, 3], boxes2[:, 3])

        intersection = torch.clamp(x2 - x1, 0) * torch.clamp(y2 - y1, 0)

        # Areas
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        # Union
        union = area1 + area2 - intersection

        # IoU
        iou = intersection / (union + 1e-6)
        return iou

    def _scale_boxes_to_original(
        self,
        boxes: torch.Tensor,
        original_shape: Tuple[int, int, int],
        scale_factor: float,
        padding: Tuple[int, int]
    ) -> torch.Tensor:
        """Scale boxes back to original image coordinates"""
        if boxes.size(0) == 0:
            return boxes

        # Remove padding
        pad_left, pad_top = padding
        boxes[:, [0, 2]] -= pad_left  # x coordinates
        boxes[:, [1, 3]] -= pad_top   # y coordinates

        # Scale back
        boxes /= scale_factor

        # Clamp to image boundaries
        h, w = original_shape[:2]
        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, w)
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, h)

        return boxes

    def visualize_results(
        self,
        image: np.ndarray,
        results: Dict[str, Any],
        class_names: Optional[List[str]] = None,
        show_confidence: bool = True
    ) -> np.ndarray:
        """Visualize detection results on image"""
        vis_image = image.copy()

        boxes = results.get("boxes", [])
        scores = results.get("scores", [])
        classes = results.get("classes", [])

        if not boxes:
            return vis_image

        # Colors for different classes
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

        for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            x1, y1, x2, y2 = map(int, box)
            color = colors[int(cls) % len(colors)]

            # Draw box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            # Draw label
            if class_names and int(cls) < len(class_names):
                label = class_names[int(cls)]
            else:
                label = f"Class {int(cls)}"

            if show_confidence:
                label += f" {score:.2f}"

            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                vis_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                vis_image, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

        return vis_image

    def benchmark(self, test_images: List[str], num_runs: int = 100) -> Dict[str, float]:
        """Benchmark inference speed"""
        import time

        times = []

        # Load test images
        images = []
        for img_path in test_images[:min(10, len(test_images))]:
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)

        if not images:
            return {"error": "No valid test images"}

        # Warmup
        for _ in range(10):
            for img in images:
                self.predict(img)

        # Benchmark single image inference
        for _ in range(num_runs):
            for img in images:
                start = time.time()
                self.predict(img)
                times.append(time.time() - start)

        # Benchmark batch inference
        batch_times = []
        for _ in range(num_runs // 10):
            start = time.time()
            self.predict_batch(images)
            batch_times.append(time.time() - start)

        return {
            "avg_single_time": np.mean(times),
            "single_fps": 1.0 / np.mean(times),
            "avg_batch_time": np.mean(batch_times) / len(images),
            "batch_fps": len(images) / np.mean(batch_times),
            "speedup": (1.0 / np.mean(times)) / (len(images) / np.mean(batch_times))
        }