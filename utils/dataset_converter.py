"""
PlantVillage Dataset to YOLO Format Converter
Converts image classification structure to object detection with bounding boxes
Can download directly from Kaggle or process local dataset
"""

import os
import shutil
import json
import subprocess
import zipfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

class PlantVillageToYOLO:
    """Convert PlantVillage dataset from classification to YOLO detection format"""
    
    def __init__(self, raw_dataset_path, output_path):
        """
        Args:
            raw_dataset_path: Path to raw PlantVillage dataset (organized as class folders)
            output_path: Path to save YOLO formatted dataset
        """
        self.raw_path = Path(raw_dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Create dataset directories
        self.images_dir = self.output_path / "images"
        self.labels_dir = self.output_path / "labels"
        self.images_train = self.images_dir / "train"
        self.images_val = self.images_dir / "val"
        self.labels_train = self.labels_dir / "train"
        self.labels_val = self.labels_dir / "val"
        
        for dir_path in [self.images_train, self.images_val, self.labels_train, self.labels_val]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect actual data location
        # If raw_path is empty or contains a single "plantvillage dataset" folder, use that
        if self.raw_path.exists():
            contents = list(self.raw_path.iterdir())
            if len(contents) == 1 and contents[0].is_dir():
                single_folder = contents[0]
                # Check if it's the PlantVillage dataset folder
                if (single_folder / "color").exists() or any(
                    f.name in ["color", "grayscale", "segmented"] 
                    for f in single_folder.iterdir() if f.is_dir()
                ):
                    self.raw_path = single_folder
                    print(f"Auto-detected dataset path: {self.raw_path}")
    
    @staticmethod
    def download_from_kaggle(dataset_name="abdallahalidev/plantvillage-dataset", 
                            download_path="datasets/raw/"):
        """
        Download PlantVillage dataset from Kaggle using Python API
        
        Requirements:
        1. Install Kaggle API: pip install kaggle
        2. Get API key: https://www.kaggle.com/account
        3. Place kaggle.json in ~/.kaggle/ (or current directory)
        
        Args:
            dataset_name: Kaggle dataset identifier
            download_path: Where to save the dataset
        
        Returns:
            bool: True if successful, False otherwise
        """
        
        print("=" * 70)
        print("DOWNLOADING FROM KAGGLE")
        print("=" * 70)
        
        # Check if kaggle is installed
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            print("✓ Kaggle API installed")
        except ImportError:
            print("⚠ Kaggle API not installed")
            print("Install with: pip install kaggle")
            return False
        
        # Create download directory
        Path(download_path).mkdir(parents=True, exist_ok=True)
        
        try:
            print(f"\nDownloading {dataset_name}...")
            print("This may take 10-30 minutes depending on connection speed\n")
            
            # Initialize Kaggle API
            api = KaggleApi()
            api.authenticate()
            
            # Download and extract dataset
            api.dataset_download_files(dataset_name, path=download_path, unzip=True)
            
            print(f"\n✓ Downloaded successfully to {download_path}")
            return True
            
        except Exception as e:
            error_msg = str(e).lower()
            
            print(f"\n✗ Error: {e}")
            print("\nTroubleshooting:")
            
            if "401" in str(e) or "credential" in error_msg or "authentication" in error_msg:
                print("Authentication failed. Check your Kaggle credentials:")
                print("1. Download kaggle.json from https://www.kaggle.com/account")
                print("2. Place in:")
                print("   - Windows: C:\\Users\\YourName\\.kaggle\\kaggle.json")
                print("   - Linux/Mac: ~/.kaggle/kaggle.json")
                print("3. Run: python utils/dataset_converter.py --setup-kaggle")
            else:
                print("1. Install Kaggle API: pip install kaggle")
                print("2. Download kaggle.json from https://www.kaggle.com/account")
                print("3. Place kaggle.json in ~/.kaggle/ directory")
                print("4. Make sure kaggle.json has correct permissions")
            
            return False
    
    @staticmethod
    def setup_kaggle_api():
        """
        Helper to setup Kaggle API
        
        Returns:
            bool: True if setup successful
        """
        
        print("\n" + "=" * 70)
        print("KAGGLE API SETUP")
        print("=" * 70)
        
        print("""
STEP 1: Install Kaggle CLI
    pip install kaggle

STEP 2: Get API Credentials
    1. Visit: https://www.kaggle.com/account
    2. Scroll to "API" section
    3. Click "Create New API Token"
    4. File 'kaggle.json' will download

STEP 3: Place kaggle.json
    Windows:  C:\\Users\\<YourName>\\.kaggle\\kaggle.json
    macOS:    ~/.kaggle/kaggle.json
    Linux:    ~/.kaggle/kaggle.json
    
    OR in current directory: ./kaggle.json

STEP 4: Set Permissions (macOS/Linux)
    chmod 600 ~/.kaggle/kaggle.json

STEP 5: Verify
    kaggle datasets list

Then run: python utils/dataset_converter.py
        """)
        
        # Check if kaggle.json exists
        kaggle_paths = [
            Path.home() / '.kaggle' / 'kaggle.json',
            Path('kaggle.json'),
        ]
        
        for path in kaggle_paths:
            if path.exists():
                print(f"✓ Found kaggle.json at: {path}")
                return True
        
        print("⚠ kaggle.json not found")
        print("Please follow the steps above and try again")
        return False
    
    def get_class_mapping(self):
        """
        Get unique disease classes from dataset structure
        PlantVillage format: Plant__Disease or Plant__Healthy
        """
        classes = set()
        for class_folder in self.raw_path.iterdir():
            if class_folder.is_dir():
                classes.add(class_folder.name)
        
        class_mapping = {cls: idx for idx, cls in enumerate(sorted(classes))}
        return class_mapping
    
    def get_class_mapping_from_path(self, base_path):
        """
        Get unique disease classes from specified path
        Handles nested structures (color/disease_class)
        """
        classes = set()
        for class_folder in base_path.iterdir():
            if class_folder.is_dir():
                classes.add(class_folder.name)
        
        class_mapping = {cls: idx for idx, cls in enumerate(sorted(classes))}
        return class_mapping
    
    def create_yolo_bbox(self, img_width, img_height, disease_type):
        """
        Create normalized YOLO bounding box
        Strategy: For leaf-based disease detection, bbox covers most of the leaf
        Format: <class_id> <x_center> <y_center> <width> <height> (normalized 0-1)
        
        For this implementation, we assume the disease affects the entire leaf,
        so we create a bounding box covering ~80% of the image (typical leaf region)
        """
        # Assume leaf occupies center ~80% of image (conservative estimate)
        # In production, you'd use actual segmentation masks
        x_center = 0.5
        y_center = 0.5
        width = 0.8
        height = 0.8
        
        return x_center, y_center, width, height
    
    def convert_dataset(self, train_split=0.8, val_split=0.2):
        """
        Convert PlantVillage to YOLO format
        
        Handles both:
        - Flat structure: raw_path/Apple__Disease/image.jpg
        - Nested structure: raw_path/color/Apple___Disease/image.jpg
        
        Args:
            train_split: Fraction for training (default 0.8)
            val_split: Fraction for validation (default 0.2)
        """
        # Auto-detect nested structure
        potential_color_dir = self.raw_path / "color"
        if potential_color_dir.exists():
            print("Detected nested dataset structure (PlantVillage format)")
            print(f"Using: {potential_color_dir}")
            source_path = potential_color_dir
        else:
            source_path = self.raw_path
        
        class_mapping = self.get_class_mapping_from_path(source_path)
        
        print(f"Found {len(class_mapping)} disease classes:")
        for cls, idx in sorted(class_mapping.items(), key=lambda x: x[1]):
            print(f"  {idx}: {cls}")
        
        image_count = 0
        train_count = 0
        val_count = 0
        
        # Process each class folder
        for class_folder in sorted(source_path.iterdir()):
            if not class_folder.is_dir():
                continue
            
            class_name = class_folder.name
            class_id = class_mapping[class_name]
            
            # Get all images in this class
            image_files = list(class_folder.glob("*"))
            image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            
            print(f"\nProcessing {class_name}: {len(image_files)} images")
            
            # Shuffle and split
            np.random.seed(42)
            np.random.shuffle(image_files)
            
            split_point = int(len(image_files) * train_split)
            train_files = image_files[:split_point]
            val_files = image_files[split_point:]
            
            # Process training images
            for img_path in train_files:
                try:
                    self._process_image(img_path, class_id, self.images_train, self.labels_train)
                    train_count += 1
                    image_count += 1
                except Exception as e:
                    print(f"  Error processing {img_path}: {e}")
            
            # Process validation images
            for img_path in val_files:
                try:
                    self._process_image(img_path, class_id, self.images_val, self.labels_val)
                    val_count += 1
                    image_count += 1
                except Exception as e:
                    print(f"  Error processing {img_path}: {e}")
        
        print(f"\n✓ Conversion complete!")
        print(f"  Total images: {image_count}")
        print(f"  Training: {train_count}")
        print(f"  Validation: {val_count}")
        
        # Save class mapping
        self._save_class_mapping(class_mapping)
        self._create_data_yaml(class_mapping)
        
        return class_mapping
    
    def _process_image(self, img_path, class_id, images_dir, labels_dir):
        """Process single image and create corresponding label file"""
        # Read and validate image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        
        height, width = img.shape[:2]
        
        # Resize to standard size (faster processing)
        target_size = 640
        scale = target_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_resized = cv2.resize(img, (new_width, new_height))
        
        # Create new image with padding to 640x640
        img_padded = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
        y_offset = (target_size - new_height) // 2
        x_offset = (target_size - new_width) // 2
        img_padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = img_resized
        
        # Save image
        output_img_path = images_dir / img_path.name
        cv2.imwrite(str(output_img_path), img_padded)
        
        # Create YOLO format label
        # For leaf disease, bounding box covers the leaf region
        x_center, y_center, width_norm, height_norm = self.create_yolo_bbox(
            target_size, target_size, ""
        )
        
        label_content = f"{class_id} {x_center} {y_center} {width_norm} {height_norm}\n"
        
        # Save label
        output_label_path = labels_dir / (img_path.stem + ".txt")
        with open(output_label_path, 'w') as f:
            f.write(label_content)
    
    def _save_class_mapping(self, class_mapping):
        """Save class mapping as JSON"""
        mapping_file = self.output_path / "class_mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump(class_mapping, f, indent=2)
        print(f"✓ Class mapping saved to {mapping_file}")
    
    def _create_data_yaml(self, class_mapping):
        """Create data.yaml for YOLOv5 training"""
        yaml_content = f"""# Plant Disease Detection Dataset
path: {self.output_path}
train: images/train
val: images/val

# Number of classes
nc: {len(class_mapping)}

# Class names
names:
"""
        # Add class names
        for cls_name, cls_id in sorted(class_mapping.items(), key=lambda x: x[1]):
            yaml_content += f"  {cls_id}: {cls_name}\n"
        
        yaml_file = self.output_path / "data.yaml"
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        
        print(f"✓ data.yaml created at {yaml_file}")


def main():
    """
    USAGE INSTRUCTIONS:
    
    OPTION 1: DOWNLOAD FROM KAGGLE (Automatic)
    ============================================
    
    1. Setup Kaggle API:
       - Install: pip install kaggle
       - Get key: https://www.kaggle.com/account
       - Place kaggle.json in ~/.kaggle/ or current directory
    
    2. Run:
       python utils/dataset_converter.py --download
    
    3. Script will:
       - Download PlantVillage dataset
       - Extract to datasets/raw/
       - Convert to YOLO format
       - Save to datasets/yolo_format/
    
    
    OPTION 2: USE EXISTING LOCAL DATA
    ================================
    
    1. Manual download from:
       https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
    
    2. Extract to: datasets/raw/
       Expected structure:
       datasets/raw/
       ├── Apple__Apple_scab/
       ├── Apple__Black_rot/
       ├── Apple__Cedar_apple_rust/
       ... (all disease classes)
    
    3. Run:
       python utils/dataset_converter.py
    """
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PlantVillage Dataset Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from Kaggle and convert
  python utils/dataset_converter.py --download
  
  # Convert existing local dataset
  python utils/dataset_converter.py
  
  # Setup Kaggle API
  python utils/dataset_converter.py --setup-kaggle
        """
    )
    
    parser.add_argument('--download', action='store_true',
                       help='Download dataset from Kaggle')
    parser.add_argument('--setup-kaggle', action='store_true',
                       help='Setup Kaggle API credentials')
    parser.add_argument('--kaggle-dataset', default='abdallahalidev/plantvillage-dataset',
                       help='Kaggle dataset identifier (default: abdallahalidev/plantvillage-dataset)')
    
    args = parser.parse_args()
    
    raw_path = "datasets/raw"
    output_path = "datasets/yolo_format"
    
    print("=" * 70)
    print("PlantVillage -> YOLO Format Converter")
    print("=" * 70)
    
    # Handle Kaggle setup
    if args.setup_kaggle:
        PlantVillageToYOLO.setup_kaggle_api()
        return
    
    # Handle Kaggle download
    if args.download:
        print("\nAttempting to download from Kaggle...")
        success = PlantVillageToYOLO.download_from_kaggle(
            dataset_name=args.kaggle_dataset,
            download_path=raw_path
        )
        
        if not success:
            print("\nFailed to download. Please setup Kaggle API:")
            print("  python utils/dataset_converter.py --setup-kaggle")
            return
    
    # Check if raw dataset exists
    if not Path(raw_path).exists() or not list(Path(raw_path).iterdir()):
        print(f"\n⚠ Dataset not found at {raw_path}")
        print("\nOptions:")
        print("1. Download from Kaggle:")
        print("   python utils/dataset_converter.py --download")
        print("\n2. Download manually:")
        print("   https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        print("   Then extract to: datasets/raw/")
        print("\n3. Setup Kaggle API:")
        print("   python utils/dataset_converter.py --setup-kaggle")
        return
    
    # Convert dataset
    print("\nStarting conversion...\n")
    converter = PlantVillageToYOLO(raw_path, output_path)
    converter.convert_dataset()
    
    print("\n" + "=" * 70)
    print("✓ Dataset converted successfully!")
    print(f"Location: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
