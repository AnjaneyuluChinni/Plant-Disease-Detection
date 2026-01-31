"""
Automate: create sample -> convert to YOLO -> train short run
Usage: python utils/convert_and_train_sample.py [epochs] [batch_size] [samples_per_class]
Defaults: epochs=3, batch_size=8, samples_per_class=20
"""
import sys
from pathlib import Path
import shutil
import random
import os

# Parameters
epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 8
samples_per_class = int(sys.argv[3]) if len(sys.argv) > 3 else 20

root = Path(__file__).resolve().parents[1]
print(f"Project root: {root}")

# Find source data (handle nested structure)
raw_root = root / 'datasets' / 'raw'
if (raw_root / 'plantvillage dataset' / 'color').exists():
    source_dir = raw_root / 'plantvillage dataset' / 'color'
elif (raw_root / 'color').exists():
    source_dir = raw_root / 'color'
else:
    # try first dir that contains many class folders
    cand = None
    for item in raw_root.iterdir():
        if item.is_dir() and any(f.is_dir() for f in item.iterdir()):
            cand = item
            break
    if cand is None:
        raise SystemExit('Could not locate PlantVillage source under datasets/raw')
    source_dir = cand

print(f"Using source dir: {source_dir}")

sample_raw = root / 'datasets' / 'raw_sample'
if sample_raw.exists():
    shutil.rmtree(sample_raw)
sample_raw.mkdir(parents=True)

total = 0
for class_dir in sorted(source_dir.iterdir()):
    if not class_dir.is_dir():
        continue
    images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
    if not images:
        continue
    pick = random.sample(images, min(samples_per_class, len(images)))
    dest_class = sample_raw / class_dir.name
    dest_class.mkdir(parents=True, exist_ok=True)
    for img in pick:
        shutil.copy2(img, dest_class / img.name)
    print(f"Copied {len(pick)} -> {dest_class}")
    total += len(pick)

print(f"Total sample images: {total}")

# Convert sample to YOLO format using PlantVillageToYOLO
print('\nConverting sample to YOLO format...')
import importlib.util
dataset_converter_path = root / 'utils' / 'dataset_converter.py'
if not dataset_converter_path.exists():
    raise SystemExit('Could not find utils/dataset_converter.py')
spec = importlib.util.spec_from_file_location('dataset_converter', str(dataset_converter_path))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PlantVillageToYOLO = module.PlantVillageToYOLO

yolo_out = root / 'datasets' / 'yolo_sample'
if yolo_out.exists():
    shutil.rmtree(yolo_out)
converter = PlantVillageToYOLO(sample_raw, yolo_out)
converter.convert_dataset()

# Train using ultralytics YOLO
print('\nStarting short training run...')
try:
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit('ultralytics not installed. Run: pip install ultralytics')

# create a data.yaml path to pass
import json
class_map_file = yolo_out / 'class_mapping.json'
if not class_map_file.exists():
    raise SystemExit('class_mapping.json missing after conversion')
with open(class_map_file, 'r') as f:
    class_mapping = json.load(f)

# build data yaml
data_yaml = yolo_out / 'data.yaml'
nc = len(class_mapping)
names = [None] * nc
for k, v in class_mapping.items():
    names[v] = k

data_yaml_text = f"path: {yolo_out}\ntrain: images/train\nval: images/val\nnc: {nc}\nnames: {names}\n"
with open(data_yaml, 'w') as f:
    f.write(data_yaml_text)

print(f"Data yaml created: {data_yaml}")

# Run training
model = YOLO('yolov5s.pt')
model.train(data=str(data_yaml), epochs=epochs, batch=batch_size, imgsz=640, device='cpu', project=str(root/'models'), name='sample_quick', save=True)

# copy best model
best = root / 'models' / 'sample_quick' / 'weights' / 'best.pt'
if best.exists():
    shutil.copy2(best, root / 'models' / 'best.pt')
    print('\nCopied best model to models/best.pt')

print('\nShort training run complete. Restart backend to load model.')
