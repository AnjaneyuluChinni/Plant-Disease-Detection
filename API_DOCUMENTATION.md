# API Documentation - Plant Disease Detection

## Base URL
```
http://localhost:5000
```

---

## Endpoints

### 1. Health Check
**Check if backend is running**

```
GET /health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-31T10:30:00.000000"
}
```

---

### 2. Upload Image
**Upload image file and get disease predictions**

```
POST /upload
```

**Request:**
- Content-Type: `multipart/form-data`
- Field: `file` (image file, JPG/PNG, max 10MB)

**Example (curl):**
```bash
curl -X POST -F "file=@path/to/image.jpg" http://localhost:5000/upload
```

**Example (Python):**
```python
import requests

with open('image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/upload', files=files)
    result = response.json()
    print(result)
```

**Response (200 OK):**
```json
{
  "status": "success",
  "detections": [
    {
      "class_id": 5,
      "class_name": "Tomato__Early_blight",
      "confidence": 0.92,
      "bbox": [100, 150, 300, 400]
    },
    {
      "class_id": 5,
      "class_name": "Tomato__Early_blight",
      "confidence": 0.87,
      "bbox": [150, 200, 320, 450]
    }
  ],
  "annotated_image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "message": "Success"
}
```

**Response (400 Bad Request):**
```json
{
  "error": "No file provided"
}
```

**Response (400 Bad Request):**
```json
{
  "error": "Invalid file type"
}
```

**Response (500 Server Error):**
```json
{
  "error": "Model not loaded"
}
```

---

### 3. Predict (Base64)
**Send base64 encoded image for prediction**

```
POST /predict
```

**Request:**
- Content-Type: `application/json`
- Body:
  ```json
  {
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
  }
  ```

**Example (JavaScript):**
```javascript
const canvas = document.getElementById('canvas');
const base64Image = canvas.toDataURL('image/jpeg');

fetch('/predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({ image: base64Image })
})
.then(response => response.json())
.then(data => console.log(data));
```

**Response (200 OK):**
```json
{
  "status": "success",
  "detections": [...],
  "annotated_image": "data:image/jpeg;base64,..."
}
```

---

### 4. Webcam Feed
**Stream live webcam with real-time detection (localhost only)**

```
GET /webcam-feed
```

**Notes:**
- Only accessible from `localhost` (security restriction)
- Returns MJPEG stream
- Not available on cloud deployments

**Example (HTML):**
```html
<img src="http://localhost:5000/webcam-feed" />
```

**Response (200 OK):**
- MJPEG stream with annotated frames
- Content-Type: `multipart/x-mixed-replace; boundary=frame`

**Response (403 Forbidden):**
```json
{
  "error": "Webcam only available locally"
}
```

---

### 5. Get Classes
**Get list of disease classes**

```
GET /api/classes
```

**Response (200 OK):**
```json
{
  "classes": {
    "0": "Apple__Apple_scab",
    "1": "Apple__Black_rot",
    "2": "Apple__Cedar_apple_rust",
    "3": "Apple__healthy",
    "4": "Blueberry__healthy",
    ...
    "37": "Tomato__Tomato_yellow_leaf_curl_virus"
  }
}
```

---

### 6. Main Page
**Serve web UI**

```
GET /
```

**Response (200 OK):**
- HTML page with frontend UI

---

## Response Formats

### Detection Object
```json
{
  "class_id": 5,                          // Class ID (0-37)
  "class_name": "Tomato__Early_blight",  // Disease name
  "confidence": 0.92,                    // Confidence (0-1)
  "bbox": [100, 150, 300, 400]           // [x1, y1, x2, y2] in pixels
}
```

### Annotated Image Format
```
"data:image/jpeg;base64,<BASE64_ENCODED_IMAGE>"
```

---

## Error Responses

### 400 Bad Request
```json
{
  "error": "No file provided"
}
```

### 403 Forbidden
```json
{
  "error": "Webcam only available locally"
}
```

### 404 Not Found
```json
{
  "error": "Endpoint not found"
}
```

### 500 Internal Server Error
```json
{
  "error": "Internal server error"
}
```

---

## Usage Examples

### Python
```python
import requests
from PIL import Image
import io

# Upload image
with open('leaf.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/upload', files=files)

result = response.json()

print(f"Status: {result['status']}")
print(f"Detections: {len(result['detections'])}")

for detection in result['detections']:
    print(f"\n{detection['class_name']}")
    print(f"  Confidence: {detection['confidence']*100:.1f}%")
    print(f"  Box: {detection['bbox']}")

# Display annotated image
img_data = result['annotated_image']
# Remove 'data:image/jpeg;base64,' prefix
img_base64 = img_data.split(',')[1]
img = Image.open(io.BytesIO(base64.b64decode(img_base64)))
img.show()
```

### JavaScript
```javascript
// Upload image
const fileInput = document.getElementById('fileInput');
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('/upload', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Detections:', data.detections);
    
    // Display image
    const img = document.getElementById('resultImage');
    img.src = data.annotated_image;
    
    // Display results
    data.detections.forEach(det => {
        console.log(`${det.class_name}: ${(det.confidence*100).toFixed(1)}%`);
    });
})
.catch(error => console.error('Error:', error));
```

### cURL
```bash
# Upload image
curl -X POST -F "file=@image.jpg" http://localhost:5000/upload

# Health check
curl http://localhost:5000/health

# Get classes
curl http://localhost:5000/api/classes
```

---

## Rate Limiting

- No rate limiting on free tier
- Recommended: 10 requests/second per IP in production

---

## Performance Metrics

### Typical Response Times

| Operation | CPU | GPU |
|-----------|-----|-----|
| Upload & Inference | 100-200ms | 15-30ms |
| Predict (base64) | 100-200ms | 15-30ms |
| Classes Lookup | 5ms | 5ms |
| Health Check | 10ms | 10ms |

---

## Authentication

Currently, no authentication required. For production deployment:

```python
# Add to app.py
from functools import wraps
from flask import request

API_KEY = "your-secret-key"

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        key = request.headers.get('X-API-Key')
        if key != API_KEY:
            return {'error': 'Invalid API key'}, 401
        return f(*args, **kwargs)
    return decorated_function

# Use on endpoints:
@app.route('/upload', methods=['POST'])
@require_api_key
def upload_image():
    ...
```

---

## CORS

Frontend requests are allowed from any origin (`*`). For production, restrict:

```python
CORS(app, origins=['https://yourdomain.com'])
```

---

## Versioning

Current API version: **v1.0**

Future versions will maintain backward compatibility.

---

## Changelog

### v1.0 (2026-01-31)
- Initial release
- Upload image endpoint
- Base64 prediction endpoint
- Webcam streaming (local)
- Health check endpoint

---

**API Documentation Version:** 1.0
**Last Updated:** January 31, 2026
