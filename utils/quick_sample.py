"""
Quick dataset sampler - uses only subset of images for fast testing
Useful for development/testing before full training
"""

import os
import shutil
from pathlib import Path
import random

def create_sample_dataset(source_dir="datasets/raw", dest_dir="datasets/raw_sample", 
                         samples_per_class=100):
    """
    Create a smaller sample dataset for quick testing
    
    Args:
        source_dir: Original dataset location
        dest_dir: Where to save the sample
        samples_per_class: Number of images per disease class
    """
    
    source = Path(source_dir)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    
    # Find the actual data location (handle nested structure)
    if (source / "plantvillage dataset" / "color").exists():
        data_source = source / "plantvillage dataset" / "color"
    elif (source / "color").exists():
        data_source = source / "color"
    else:
        # Find first directory with subdirs
        for item in source.iterdir():
            if item.is_dir() and any(f.is_dir() for f in item.iterdir()):
                data_source = item
                break
    
    print(f"Source: {data_source}")
    print(f"Creating sample with {samples_per_class} images per class...\n")
    
    total = 0
    for class_dir in sorted(data_source.iterdir()):
        if not class_dir.is_dir():
            continue
        
        # Get all images
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        if not images:
            continue
        
        # Random sample
        sample_images = random.sample(images, min(samples_per_class, len(images)))
        
        # Create class folder in destination
        dest_class = dest / class_dir.name
        dest_class.mkdir(exist_ok=True)
        
        # Copy images
        for img in sample_images:
            shutil.copy2(img, dest_class / img.name)
        
        print(f"✓ {class_dir.name}: {len(sample_images)} images")
        total += len(sample_images)
    
    print(f"\n✓ Sample created: {total} total images")
    print(f"Location: {dest}")
    print(f"\nNow convert: python utils/dataset_converter.py")
    return dest

if __name__ == "__main__":
    import sys
    
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    create_sample_dataset(samples_per_class=samples)
