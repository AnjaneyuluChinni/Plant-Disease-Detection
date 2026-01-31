"""
Flask Backend for Plant Disease Detection
Endpoints:
- POST /upload: Upload image and get predictions
- POST /predict: Get disease predictions from uploaded image
- GET /health: Health check
- GET /webcam: Webcam streaming (local only)
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import base64
import io
import os
from datetime import datetime

# Configure Flask to use frontend folders
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend', 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static'))

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model
model = None
class_names = {}

def load_model(model_path="models/best.pt", classes_path="datasets/yolo_format/class_mapping.json"):
    """
    Load YOLOv5 model and class names using ultralytics
    
    If trained model doesn't exist or fails, uses pre-trained YOLOv5s
    for testing/development purposes
    """
    global model, class_names
    
    try:
        # Check if trained model exists
        if Path(model_path).exists():
            print(f"Loading trained model from {model_path}...")
            try:
                model = YOLO(model_path)
                model_source = "trained"
                print(f"✓ Trained model loaded successfully")
            except Exception as e:
                print(f"⚠ Failed to load trained model: {str(e)}")
                print(f"Falling back to pre-trained YOLOv5s...")
                model = YOLO('yolov5s.pt')
                model_source = "pre-trained (fallback)"
        else:
            print(f"Trained model not found at {model_path}")
            print("Loading pre-trained YOLOv5s...")
            model = YOLO('yolov5s.pt')
            model_source = "pre-trained"
        
        # Ensure model is loaded
        if model is None:
            raise Exception("Failed to load any model")
        
        model.conf = 0.25  # Confidence threshold
        model.iou = 0.45   # NMS IoU threshold
        # Force CPU on Render (no GPU available)
        try:
            model.to("cpu")
        except Exception as e:
            print(f"⚠ Failed to force CPU mode: {str(e)}")
        
        # Load class names if available
        if Path(classes_path).exists():
            import json
            try:
                with open(classes_path, 'r') as f:
                    class_mapping = json.load(f)
                    class_names = {v: k for k, v in class_mapping.items()}
                class_info = f"{len(class_names)} plant disease classes"
            except Exception as e:
                print(f"⚠ Failed to load class mapping: {str(e)}")
                class_names = {}
                class_info = "default YOLOv5 classes"
        else:
            class_names = {}
            class_info = "default YOLOv5 classes"
        
        return True, f"Model loaded successfully ({model_source}) - {class_info}"
    except Exception as e:
        print(f"✗ Critical error loading model: {str(e)}")
        return False, f"Failed to load model: {str(e)}"


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image(image_path):
    """
    Run inference on image using ultralytics YOLO
    Returns: (detections_list, annotated_image, status_message)
    """
    try:
        # Check if model is loaded
        if model is None:
            return None, None, "Model not initialized"
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None, None, "Could not read image"
        
        # Inference using ultralytics
        results = model(image_path, conf=0.25, verbose=False, device="cpu")
        
        # Parse results
        detections = []
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
                    if class_names:
                        disease_name = class_names.get(class_id, f"Class {class_id}")
                    else:
                        # Fallback: use model's class names if available
                        disease_name = model.names.get(class_id, f"Class {class_id}") if hasattr(model, 'names') and model.names else f"Class {class_id}"
                    
                    detections.append({
                        'class_id': class_id,
                        'class_name': disease_name,
                        'confidence': float(confidence),
                        'bbox': [x1, y1, x2, y2]
                    })
        
        # Annotate image
        annotated_img = img.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            conf = det['confidence']
            
            # Draw bounding box
            color = (0, 255, 0) if conf > 0.7 else (255, 165, 0) if conf > 0.5 else (255, 0, 0)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{class_name} ({conf:.2f})"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_img, (x1, y1 - label_size[1] - baseline),
                        (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_img, label, (x1, y1 - baseline),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return detections, annotated_img, "Success"
    except Exception as e:
        return None, None, str(e)


def image_to_base64(cv_image):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', cv_image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str


# Routes

@app.route('/', methods=['GET'])
def index():
    """Serve main HTML page"""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/upload', methods=['POST'])
def upload_image():
    """
    Upload image and return predictions with annotated image
    Expected: multipart/form-data with 'file' field
    Returns: JSON with detections and base64 encoded annotated image
    """
    
    # Check if model is loaded
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Save file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"img_{timestamp}_{file.filename}"
    filepath = Path(UPLOAD_FOLDER) / filename
    
    try:
        file.save(str(filepath))
        
        # Process image
        detections, annotated_img, message = process_image(str(filepath))
        
        if annotated_img is None:
            return jsonify({'error': f'Processing failed: {message}'}), 500
        
        # Encode annotated image
        annotated_b64 = image_to_base64(annotated_img)
        
        # Return results
        return jsonify({
            'status': 'success',
            'detections': detections,
            'annotated_image': f"data:image/jpeg;base64,{annotated_b64}",
            'message': message
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up uploaded file
        if filepath.exists():
            filepath.unlink()


@app.route('/predict', methods=['POST'])
def predict():
    """
    Get disease predictions from base64 image
    Expected: JSON with 'image' field (base64 encoded)
    """
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Could not decode image'}), 400
        
        # Save temporarily
        temp_path = 'temp_prediction.jpg'
        cv2.imwrite(temp_path, img)
        
        # Process
        detections, annotated_img, message = process_image(temp_path)
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
        
        if annotated_img is None:
            return jsonify({'error': f'Processing failed: {message}'}), 500
        
        annotated_b64 = image_to_base64(annotated_img)
        
        return jsonify({
            'status': 'success',
            'detections': detections,
            'annotated_image': f"data:image/jpeg;base64,{annotated_b64}"
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/webcam-feed', methods=['GET'])
def webcam_feed():
    """
    Webcam streaming endpoint (local use only)
    Returns streaming video with real-time detection
    """
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Check if running locally
    remote_addr = request.remote_addr
    if remote_addr not in ['127.0.0.1', 'localhost']:
        return jsonify({'error': 'Webcam only available locally'}), 403
    
    def generate():
        """Generate frames from webcam"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame temporarily
            temp_path = 'temp_frame.jpg'
            cv2.imwrite(temp_path, frame)
            
            # Inference
            detections, annotated_frame, _ = process_image(temp_path)
            Path(temp_path).unlink(missing_ok=True)
            
            if annotated_frame is None:
                continue
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        cap.release()
    
    return app.response_class(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get list of disease classes"""
    return jsonify({
        'classes': class_names
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


def main():
    """Initialize and run Flask app"""
    print("=" * 70)
    print("PLANT DISEASE DETECTION BACKEND")
    print("=" * 70)
    
    # Try to load model
    print("\nLoading model...")
    success, message = load_model()
    
    if success:
        print(f"✓ {message}")
        print(f"✓ Classes loaded: {len(class_names)}")
    else:
        print(f"✗ {message}")
        print("✗ CRITICAL: Unable to load any model")
        print("  Exiting...")
        return
    
    print("\n" + "=" * 70)
    print("Starting Flask server at http://localhost:5000")
    print("=" * 70)
    print("\nEndpoints:")
    print("  GET  / (main UI)")
    print("  GET  /health (health check)")
    print("  POST /upload (upload image)")
    print("  POST /predict (get predictions)")
    print("  GET  /webcam-feed (live webcam - local only)")
    
    # Run app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )


if __name__ == '__main__':
    main()
