"""
YOLOv5 Training Script for Plant Disease Detection
Includes transfer learning, evaluation, and model checkpoint management
"""

import torch
import yaml
from pathlib import Path
from datetime import datetime

class YOLOv5Trainer:
    """Wrapper for YOLOv5 training with disease detection"""
    
    def __init__(self, data_yaml_path, model_name="yolov5s"):
        """
        Args:
            data_yaml_path: Path to data.yaml
            model_name: YOLOv5 variant (yolov5n, yolov5s, yolov5m, yolov5l)
        """
        self.data_yaml = data_yaml_path
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Device: {self.device}")
        if self.device == "cpu":
            print("⚠ Running on CPU - training will be slower")
            print("  Recommend: GPU for faster training (NVIDIA CUDA)")
        
    def train(self, epochs=50, batch_size=16, img_size=640, patience=10):
        """
        Train YOLOv5 model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size (reduce for low-memory systems)
            img_size: Image size (640 is standard)
            patience: Early stopping patience
        """
        
        print("\n" + "=" * 70)
        print("YOLOv5 TRAINING CONFIGURATION")
        print("=" * 70)
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Batch Size: {batch_size}")
        print(f"Image Size: {img_size}")
        print(f"Patience (early stopping): {patience}")
        print("=" * 70 + "\n")
        
        try:
            from yolov5 import YOLOv5
        except ImportError:
            print("YOLOv5 not installed. Installing...")
            import subprocess
            subprocess.run(["pip", "install", "yolov5"], check=True)
            from yolov5 import YOLOv5
        
        # Load pretrained model (transfer learning)
        print(f"Loading pretrained {self.model_name}...")
        model = torch.hub.load('ultralytics/yolov5', self.model_name, pretrained=True)
        
        # Hyperparameters optimized for plant disease detection
        hyperparams = {
            'lr0': 0.01,           # Initial learning rate
            'lrf': 0.1,            # Final learning rate (relative)
            'momentum': 0.937,     # SGD momentum
            'weight_decay': 0.0005,# L2 regularization
            'warmup_epochs': 3.0,  # Warmup epochs
            'warmup_momentum': 0.8,
            'box': 0.05,           # Box loss gain
            'cls': 0.5,            # Cls loss gain
            'cls_pw': 1.0,         # Cls BCELoss positive weight
            'obj': 1.0,            # Obj loss gain
            'obj_pw': 1.0,         # Obj BCELoss positive weight
            'iou_t': 0.20,         # IoU training threshold
            'anchor_t': 4.0,       # Anchor multiple threshold
            'fl_gamma': 0.0,       # Focal loss gamma
            'hsv_h': 0.015,        # HSV-Hue augmentation
            'hsv_s': 0.7,          # HSV-Saturation augmentation
            'hsv_v': 0.4,          # HSV-Value augmentation
            'degrees': 0.0,        # Rotation
            'translate': 0.1,      # Translation
            'scale': 0.5,          # Scale
            'flipud': 0.0,         # Flip UD
            'fliplr': 0.5,         # Flip LR
            'mosaic': 1.0,         # Mosaic augmentation
            'mixup': 0.0,          # Mixup augmentation
        }
        
        print("Training hyperparameters:")
        for key, value in hyperparams.items():
            print(f"  {key}: {value}")
        
        # Train
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70 + "\n")
        
        results = model.train(
            data=str(self.data_yaml),
            epochs=epochs,
            imgsz=img_size,
            batch_size=batch_size,
            device=0 if self.device == "cuda" else "cpu",
            patience=patience,
            save=True,
            cache=False,
            workers=4 if self.device == "cuda" else 0,
            project="models",
            name="yolov5_plant_disease",
            exist_ok=True,
            verbose=True,
            # Data augmentation
            hsv_h=hyperparams['hsv_h'],
            hsv_s=hyperparams['hsv_s'],
            hsv_v=hyperparams['hsv_v'],
            degrees=hyperparams['degrees'],
            translate=hyperparams['translate'],
            scale=hyperparams['scale'],
            flipud=hyperparams['flipud'],
            fliplr=hyperparams['fliplr'],
            mosaic=hyperparams['mosaic'],
            mixup=hyperparams['mixup'],
        )
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print("Model saved at: models/yolov5_plant_disease/weights/best.pt")
        
        return model


def train_command_reference():
    """
    Reference for manual YOLOv5 training via command line
    """
    
    commands = """
    # YOLOv5 Training Command Reference
    
    1. BASIC TRAINING (CPU):
    python -m yolov5.train \\
        --data datasets/yolo_format/data.yaml \\
        --weights yolov5s.pt \\
        --epochs 50 \\
        --batch-size 16 \\
        --img 640 \\
        --device cpu \\
        --patience 10 \\
        --project models \\
        --name yolov5_plant_disease
    
    2. GPU TRAINING (recommended):
    python -m yolov5.train \\
        --data datasets/yolo_format/data.yaml \\
        --weights yolov5s.pt \\
        --epochs 50 \\
        --batch-size 32 \\
        --img 640 \\
        --device 0 \\
        --patience 10 \\
        --cache \\
        --project models \\
        --name yolov5_plant_disease
    
    3. RESUME TRAINING:
    python -m yolov5.train \\
        --resume models/yolov5_plant_disease/weights/last.pt
    
    4. VALIDATION:
    python -m yolov5.val \\
        --weights models/yolov5_plant_disease/weights/best.pt \\
        --data datasets/yolo_format/data.yaml \\
        --img 640 \\
        --batch-size 32
    
    5. INFERENCE:
    python -m yolov5.detect \\
        --weights models/yolov5_plant_disease/weights/best.pt \\
        --source path/to/image.jpg \\
        --img 640 \\
        --conf 0.25 \\
        --iou-thres 0.45
    
    HYPERPARAMETER TUNING:
    - Increase batch-size if GPU memory allows (better gradient estimates)
    - Reduce batch-size on CPU (memory constraints)
    - lr0: Start with 0.01, reduce if loss doesn't decrease
    - patience: Increase for longer training windows
    - epochs: 50-100 typically sufficient for disease detection
    """
    
    return commands


if __name__ == "__main__":
    # Simple training starter
    data_yaml_path = "datasets/yolo_format/data.yaml"
    
    if not Path(data_yaml_path).exists():
        print(f"Error: {data_yaml_path} not found!")
        print("First run: python utils/dataset_converter.py")
    else:
        trainer = YOLOv5Trainer(data_yaml_path, model_name="yolov5s")
        
        # Adjust these based on your system:
        # - CPU: epochs=20, batch_size=8
        # - GPU: epochs=50, batch_size=32
        trainer.train(epochs=50, batch_size=16, patience=10)
