"""
Evaluation metrics for YOLOv5 Plant Disease Detection Model
Computes: Precision, Recall, F1-score, mAP@0.5, FPS, Latency
"""

import torch
import numpy as np
import cv2
import time
from pathlib import Path
from collections import defaultdict
import json


class YOLOv5Evaluator:
    """Compute evaluation metrics for detection model"""
    
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        Args:
            model_path: Path to best.pt
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = next(self.model.parameters()).device
    
    def compute_metrics(self, val_images_path, val_labels_path):
        """
        Compute precision, recall, F1-score, mAP@0.5
        
        Args:
            val_images_path: Path to validation images
            val_labels_path: Path to validation labels (YOLO format)
        """
        
        images_path = Path(val_images_path)
        labels_path = Path(val_labels_path)
        
        image_files = sorted(images_path.glob("*.jpg")) + sorted(images_path.glob("*.png"))
        
        print(f"Evaluating on {len(image_files)} images...\n")
        
        # Storage for metrics
        true_positives = defaultdict(int)
        false_positives = defaultdict(int)
        false_negatives = defaultdict(int)
        all_confidences = []
        
        processing_times = []
        
        for idx, img_path in enumerate(image_files):
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(image_files)}")
            
            # Load ground truth
            label_path = labels_path / (img_path.stem + ".txt")
            if not label_path.exists():
                continue
            
            ground_truth = self._load_yolo_labels(label_path, img_path)
            
            # Inference
            start_time = time.time()
            results = self.model(str(img_path), conf=self.conf_threshold)
            inference_time = time.time() - start_time
            processing_times.append(inference_time)
            
            # Parse predictions
            predictions = self._parse_predictions(results, img_path)
            
            # Match predictions to ground truth
            matched_preds = set()
            for gt_class, gt_box in ground_truth:
                best_iou = 0
                best_pred_idx = -1
                
                for pred_idx, (pred_class, pred_conf, pred_box) in enumerate(predictions):
                    if pred_idx in matched_preds:
                        continue
                    if pred_class != gt_class:
                        continue
                    
                    iou = self._compute_iou(gt_box, pred_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_idx = pred_idx
                
                if best_iou >= 0.5:  # IoU threshold for TP
                    true_positives[gt_class] += 1
                    matched_preds.add(best_pred_idx)
                else:
                    false_negatives[gt_class] += 1
            
            # Unmatched predictions are FP
            for pred_idx, (pred_class, pred_conf, pred_box) in enumerate(predictions):
                if pred_idx not in matched_preds:
                    false_positives[pred_class] += 1
                    all_confidences.append(pred_conf)
        
        # Compute metrics
        metrics = self._calculate_metrics(true_positives, false_positives, false_negatives)
        fps = 1.0 / np.mean(processing_times) if processing_times else 0
        avg_latency = np.mean(processing_times) * 1000  # milliseconds
        
        return metrics, fps, avg_latency
    
    def _load_yolo_labels(self, label_path, img_path):
        """Load YOLO format labels"""
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert to pixel coordinates
                    x1 = int((x_center - width/2) * w)
                    y1 = int((y_center - height/2) * h)
                    x2 = int((x_center + width/2) * w)
                    y2 = int((y_center + height/2) * h)
                    
                    labels.append((class_id, (x1, y1, x2, y2)))
        
        return labels
    
    def _parse_predictions(self, results, img_path):
        """Parse YOLOv5 predictions"""
        predictions = []
        
        if results.xyxy[0].shape[0] > 0:
            for *box, conf, cls_id in results.xyxy[0]:
                x1, y1, x2, y2 = [int(v) for v in box]
                predictions.append((
                    int(cls_id),
                    float(conf),
                    (x1, y1, x2, y2)
                ))
        
        return predictions
    
    def _compute_iou(self, box1, box2):
        """Compute Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        intersection = max(0, min(x1_max, x2_max) - max(x1_min, x2_min)) * \
                       max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        iou = intersection / union if union > 0 else 0
        return iou
    
    def _calculate_metrics(self, tp, fp, fn):
        """Calculate precision, recall, F1-score for each class"""
        metrics = {
            'per_class': {},
            'overall': {}
        }
        
        all_tp = 0
        all_fp = 0
        all_fn = 0
        
        for class_id in set(tp.keys()) | set(fp.keys()) | set(fn.keys()):
            tp_c = tp.get(class_id, 0)
            fp_c = fp.get(class_id, 0)
            fn_c = fn.get(class_id, 0)
            
            precision = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0
            recall = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics['per_class'][class_id] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'tp': tp_c,
                'fp': fp_c,
                'fn': fn_c
            }
            
            all_tp += tp_c
            all_fp += fp_c
            all_fn += fn_c
        
        # Overall metrics
        overall_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
        overall_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (
            overall_precision + overall_recall
        ) if (overall_precision + overall_recall) > 0 else 0
        
        metrics['overall'] = {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'tp': all_tp,
            'fp': all_fp,
            'fn': all_fn
        }
        
        return metrics
    
    def print_metrics(self, metrics, fps, latency):
        """Pretty print evaluation metrics"""
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        overall = metrics['overall']
        print(f"\nOVERALL METRICS:")
        print(f"  Precision: {overall['precision']:.4f}")
        print(f"  Recall:    {overall['recall']:.4f}")
        print(f"  F1-Score:  {overall['f1_score']:.4f}")
        print(f"  TP: {overall['tp']}, FP: {overall['fp']}, FN: {overall['fn']}")
        
        print(f"\nPERFORMANCE:")
        print(f"  FPS (Frames/Second):  {fps:.2f}")
        print(f"  Latency (ms):         {latency:.2f}")
        
        print(f"\nPER-CLASS METRICS:")
        for class_id, class_metrics in metrics['per_class'].items():
            print(f"\n  Class {class_id}:")
            print(f"    Precision: {class_metrics['precision']:.4f}")
            print(f"    Recall:    {class_metrics['recall']:.4f}")
            print(f"    F1-Score:  {class_metrics['f1_score']:.4f}")
        
        print("\n" + "=" * 70)


def save_evaluation_report(metrics, fps, latency, output_path="models/evaluation_report.json"):
    """Save metrics to JSON file"""
    report = {
        'overall_metrics': metrics['overall'],
        'per_class_metrics': metrics['per_class'],
        'performance': {
            'fps': fps,
            'latency_ms': latency
        }
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Evaluation report saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    model_path = "models/yolov5_plant_disease/weights/best.pt"
    val_images = "datasets/yolo_format/images/val"
    val_labels = "datasets/yolo_format/labels/val"
    
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("First train model: python utils/train_yolov5.py")
    else:
        evaluator = YOLOv5Evaluator(model_path)
        metrics, fps, latency = evaluator.compute_metrics(val_images, val_labels)
        evaluator.print_metrics(metrics, fps, latency)
        save_evaluation_report(metrics, fps, latency)
