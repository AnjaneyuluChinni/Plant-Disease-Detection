"""
Quick inference script for single image testing
Usage: python utils/inference.py --image path/to/image.jpg --model models/best.pt
"""

import argparse
import cv2
import json
from pathlib import Path
from ultralytics import YOLO


class PlantDiseaseDetector:
    """Simple inference wrapper using ultralytics YOLO"""
    
    def __init__(self, model_path="models/best.pt", class_mapping_path="datasets/yolo_format/class_mapping.json"):
        """Initialize detector"""
        self.model = YOLO(model_path)
        self.model.conf = 0.25
        self.model.iou = 0.45
        
        # Load classes
        self.classes = {}
        if Path(class_mapping_path).exists():
            with open(class_mapping_path, 'r') as f:
                mapping = json.load(f)
                self.classes = {v: k for k, v in mapping.items()}
        else:
            # Fallback: use model's class names
            if hasattr(self.model, 'names') and self.model.names:
                self.classes = self.model.names
    
    def predict(self, image_path, conf_threshold=0.25):
        """
        Run inference on image
        
        Args:
            image_path: Path to image
            conf_threshold: Confidence threshold
            
        Returns:
            detections: List of {class_name, confidence, bbox}
            annotated_image: Image with drawn boxes
        """
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        h, w = img.shape[:2]
        
        # Inference
        results = self.model(image_path, conf=conf_threshold, verbose=False)
        
        detections = []
        annotated_img = img.copy()
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get confidence and class
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Get class name
                    if isinstance(self.classes, dict):
                        class_name = self.classes.get(class_id, f"Class {class_id}")
                    else:
                        class_name = self.model.names.get(class_id, f"Class {class_id}") if self.model.names else f"Class {class_id}"
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2]
                    })
                    
                    # Draw box
                    color = (0, 255, 0) if confidence > 0.7 else (255, 165, 0)
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f"{class_name} ({confidence:.2f})"
                    cv2.putText(annotated_img, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return detections, annotated_img


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Plant Disease Detection - Inference")
    parser.add_argument('--image', required=True, help='Path to image')
    parser.add_argument('--model', default='models/best.pt', help='Path to model')
    parser.add_argument('--output', default=None, help='Output image path')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PLANT DISEASE DETECTION - INFERENCE")
    print("=" * 60)
    
    # Load detector
    print(f"\nLoading model: {args.model}")
    detector = PlantDiseaseDetector(args.model)
    
    # Run inference
    print(f"Processing image: {args.image}")
    detections, annotated_img = detector.predict(args.image, args.conf)
    
    # Print results
    print(f"\n✓ Found {len(detections)} disease(s):\n")
    for i, det in enumerate(detections, 1):
        print(f"  {i}. {det['class']}")
        print(f"     Confidence: {det['confidence']*100:.1f}%")
        print(f"     Box: {det['bbox']}")
    
    # Save result
    if args.output or args.output is None:
        output_path = args.output or 'detection_result.jpg'
        cv2.imwrite(output_path, annotated_img)
        print(f"\n✓ Saved to: {output_path}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
