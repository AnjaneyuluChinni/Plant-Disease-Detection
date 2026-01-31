"""
Simplified YOLOv5 Training Script
Uses ultralytics directly (no conflicts)
Works on CPU without GPU
"""

import torch
from pathlib import Path
from ultralytics import YOLO
import yaml

def train_model(data_yaml="datasets/yolo_format/data.yaml",
                epochs=50,
                batch_size=8,  # Reduced for CPU
                img_size=640,
                device="cpu",
                patience=10):
    """
    Train YOLOv5s model on plant disease dataset
    
    Args:
        data_yaml: Path to data.yaml file
        epochs: Number of training epochs
        batch_size: Batch size (smaller for CPU)
        img_size: Input image size
        device: Device to use (cpu or 0 for GPU)
        patience: Early stopping patience
    """
    
    print("=" * 70)
    print("YOLOv5 TRAINING - SIMPLIFIED VERSION")
    print("=" * 70)
    
    # Check data.yaml exists
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"Error: {data_yaml} not found")
        print("Run: python utils/dataset_converter.py")
        return False
    
    print(f"\n✓ Data file found: {data_yaml}")
    print(f"✓ Device: {device}")
    print(f"✓ Batch size: {batch_size}")
    print(f"✓ Epochs: {epochs}")
    print(f"✓ Image size: {img_size}")
    
    try:
        # Create models directory
        Path("models").mkdir(exist_ok=True)
        
        # Load pre-trained model
        print("\nLoading YOLOv5s model...")
        model = YOLO('yolov5s.pt')
        
        # Train
        print("\nStarting training...\n")
        results = model.train(
            data=str(data_path),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            patience=patience,
            project='models',
            name='yolov5s_plant_disease',
            save=True,
            verbose=True,
            plots=True
        )
        
        # Copy best model
        import shutil
        best_model = Path('models/yolov5s_plant_disease/weights/best.pt')
        if best_model.exists():
            shutil.copy(best_model, 'models/best.pt')
            print(f"\n✓ Best model saved to models/best.pt")
        
        print("\n" + "=" * 70)
        print("✓ Training complete!")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    
    # Parse arguments
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    
    print("\n" + "=" * 70)
    print("PLANT DISEASE DETECTION - MODEL TRAINING")
    print("=" * 70)
    
    success = train_model(
        epochs=epochs,
        batch_size=batch_size,
        device='cpu'  # CPU training
    )
    
    if success:
        print("\nNext steps:")
        print("1. Restart the backend: python backend/app.py")
        print("2. Open http://localhost:5000")
        print("3. Upload images for disease detection")
    else:
        print("\nTraining failed. Check the error messages above.")
