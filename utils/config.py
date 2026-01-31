"""
Configuration file for the Plant Disease Detection project
Centralized settings for all components
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_PATH = PROJECT_ROOT / "datasets"
RAW_DATASET_PATH = DATASETS_PATH / "raw"
YOLO_DATASET_PATH = DATASETS_PATH / "yolo_format"
MODELS_PATH = PROJECT_ROOT / "models"
BACKEND_PATH = PROJECT_ROOT / "backend"
UPLOADS_PATH = BACKEND_PATH / "uploads"

# Create directories if they don't exist
for path in [DATASETS_PATH, YOLO_DATASET_PATH, MODELS_PATH, UPLOADS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL SETTINGS
# ============================================================================

MODEL_NAME = "yolov5s"  # YOLOv5 variant: yolov5n, yolov5s, yolov5m, yolov5l
MODEL_PATH = MODELS_PATH / "best.pt"
CLASS_MAPPING_PATH = YOLO_DATASET_PATH / "class_mapping.json"
DATA_YAML_PATH = YOLO_DATASET_PATH / "data.yaml"

# Inference settings
CONFIDENCE_THRESHOLD = 0.25  # Detection confidence threshold
IOU_THRESHOLD = 0.45         # NMS IoU threshold
INPUT_SIZE = 640             # Model input size (640x640)
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# ============================================================================
# TRAINING SETTINGS
# ============================================================================

TRAINING_CONFIG = {
    'epochs': 50,
    'batch_size': 16,  # Reduce to 8 for low-memory systems
    'img_size': 640,
    'patience': 10,    # Early stopping patience
    'device': DEVICE,
    'workers': 4,      # DataLoader workers
}

HYPERPARAMETERS = {
    'lr0': 0.01,           # Initial learning rate
    'lrf': 0.1,            # Final learning rate
    'momentum': 0.937,     # SGD momentum
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'hsv_h': 0.015,        # HSV Hue augmentation
    'hsv_s': 0.7,          # HSV Saturation augmentation
    'hsv_v': 0.4,          # HSV Value augmentation
    'degrees': 0.0,        # Rotation augmentation
    'translate': 0.1,      # Translation augmentation
    'scale': 0.5,          # Scale augmentation
    'flipud': 0.0,         # Flip upside down
    'fliplr': 0.5,         # Flip left-right
    'mosaic': 1.0,         # Mosaic augmentation
    'mixup': 0.0,          # Mixup augmentation
}

# ============================================================================
# DATASET SETTINGS
# ============================================================================

TRAIN_VAL_SPLIT = 0.8      # 80% train, 20% validation
RESIZE_HEIGHT = 640        # Target image height
RESIZE_WIDTH = 640         # Target image width
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# ============================================================================
# FLASK APP SETTINGS
# ============================================================================

FLASK_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False,
    'threaded': True,
}

# API settings
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_CONTENT_TYPES = {'image/jpeg', 'image/png'}

# CORS settings
CORS_ORIGINS = ['*']  # Allow all origins. Restrict in production!

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================

EVAL_CONFIG = {
    'conf_threshold': 0.25,
    'iou_threshold': 0.45,
    'iou_match_threshold': 0.5,  # IoU threshold for TP/FP classification
}

# ============================================================================
# LOGGING SETTINGS
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================================
# DEPLOYMENT SETTINGS
# ============================================================================

# Render
RENDER_CONFIG = {
    'max_workers': 1,
    'timeout': 300,
}

# Production
PRODUCTION = os.getenv('PRODUCTION', 'False').lower() == 'true'
ENABLE_WEBCAM = not PRODUCTION  # Disable webcam in production
DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# Caching
CACHE_ENABLED = True
CACHE_TTL = 300  # 5 minutes

# Model optimization
FP16 = False  # Mixed precision training (requires GPU)
OPTIMIZE_MODEL = False  # ONNX/TorchScript optimization

# ============================================================================
# DISEASE CLASS INFORMATION
# ============================================================================

DISEASE_INFO = {
    # Format: 'disease_name': {'severity': 'high/medium/low', 'treatment': 'description'}
    # This will be populated from class_mapping.json
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_model_path():
    """Get path to best model"""
    if MODEL_PATH.exists():
        return str(MODEL_PATH)
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")


def get_data_yaml():
    """Get path to data.yaml"""
    if DATA_YAML_PATH.exists():
        return str(DATA_YAML_PATH)
    raise FileNotFoundError(f"data.yaml not found. Run dataset_converter.py first.")


def is_valid_image(filename):
    """Check if file is a valid image"""
    ext = Path(filename).suffix
    return ext in ALLOWED_EXTENSIONS


def get_device_info():
    """Get device information"""
    import torch
    return {
        'device': DEVICE,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


if __name__ == '__main__':
    # Print configuration
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"\nPaths:")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Models: {MODELS_PATH}")
    print(f"  Datasets: {YOLO_DATASET_PATH}")
    print(f"\nModel:")
    print(f"  Name: {MODEL_NAME}")
    print(f"  Input Size: {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"  Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  Device: {DEVICE}")
    print(f"\nTraining:")
    print(f"  Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"  Batch Size: {TRAINING_CONFIG['batch_size']}")
    print(f"\nFlask:")
    print(f"  Host: {FLASK_CONFIG['host']}")
    print(f"  Port: {FLASK_CONFIG['port']}")
    print(f"\nDeployment:")
    print(f"  Production: {PRODUCTION}")
    print(f"  Webcam Enabled: {ENABLE_WEBCAM}")
    print(f"\nDevice Info:")
    device_info = get_device_info()
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    print("\n" + "=" * 60)
