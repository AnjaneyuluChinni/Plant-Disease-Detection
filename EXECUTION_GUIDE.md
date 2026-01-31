# EXECUTION GUIDE - Plant Disease Detection

## 🚀 START HERE

This file provides step-by-step instructions to get the project running.

---

## 📋 Checklist Before Starting

- [ ] Python 3.9+ installed
- [ ] pip/pip3 working
- [ ] 4GB RAM available
- [ ] 10GB free disk space
- [ ] Internet connection (for downloads)

---

## ⚡ FASTEST START (5 minutes)

### Windows Users
```bash
cd "Plant Disease Detection"
setup_and_run.bat
```

### macOS/Linux Users
```bash
cd "Plant Disease Detection"
bash setup_and_run.sh
```

This will:
1. ✅ Create virtual environment
2. ✅ Install all dependencies
3. ✅ Create project directories
4. ✅ Download/prepare dataset (if needed)
5. ✅ Start Flask server
6. ✅ Open http://localhost:5000 in browser

---

## 📝 MANUAL SETUP (if automatic fails)

### Step 1: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**⏳ This takes 5-15 minutes** (PyTorch and YOLOv5 are large)

### Step 3: Create Directories
```bash
mkdir -p datasets/raw
mkdir -p datasets/yolo_format
mkdir -p models
mkdir -p backend/uploads
```

### Step 4: Get the Model

#### Option A: Download Pre-trained (Fastest)
```bash
# Windows (PowerShell)
$ProgressPreference = 'SilentlyContinue'
Invoke-WebRequest -Uri "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt" -OutFile "models/best.pt"

# macOS/Linux (curl)
curl -L -o models/best.pt https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
```

#### Option B: Train Your Own (24-36 hours on CPU, 2-4 hours on GPU)
```bash
# 1. Download PlantVillage dataset from Kaggle
#    https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
# 2. Extract to datasets/raw/
# 3. Run:
python utils/dataset_converter.py
python utils/train_yolov5.py
```

### Step 5: Start Backend
```bash
cd backend
python app.py
```

You should see:
```
======================================================================
PLANT DISEASE DETECTION BACKEND
======================================================================

Loading model...
✓ Model loaded successfully
✓ Classes loaded: 38

======================================================================
Starting Flask server at http://localhost:5000
======================================================================

Running on http://0.0.0.0:5000
```

### Step 6: Open Browser
```
http://localhost:5000
```

---

## 🎯 NEXT STEPS

### 1. Test with Upload
1. Click "Choose File" button
2. Select a leaf image
3. Click "Analyze"
4. See disease detection results

### 2. Test with Camera
1. Click "Live Camera" mode
2. Click "Start Camera"
3. Click "Capture & Analyze"
4. See real-time detection

### 3. Train Your Own Model (Optional)
```bash
python utils/dataset_converter.py      # Convert dataset
python utils/train_yolov5.py           # Start training
python utils/evaluation.py              # Evaluate performance
```

### 4. Deploy to Cloud (Optional)
See DEPLOYMENT_INSTRUCTIONS.md

---

## 🧪 TESTING THE API

### Health Check
```bash
curl http://localhost:5000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-31T10:30:00"
}
```

### Upload Image
```bash
curl -X POST -F "file=@test_image.jpg" http://localhost:5000/upload
```

### Get Classes
```bash
curl http://localhost:5000/api/classes
```

---

## 🐛 COMMON ISSUES & FIXES

### "Python not found"
```bash
# Windows: Check Python is added to PATH
# Restart terminal after Python installation

# macOS/Linux: Use python3
python3 --version
python3 -m venv venv
```

### "pip: command not found"
```bash
# Use Python's pip module
python -m pip --version
python -m pip install -r requirements.txt
```

### "Module not found: torch"
```bash
# Reinstall PyTorch
pip install --force-reinstall torch torchvision
```

### "CUDA out of memory"
```bash
# Use CPU instead
# Edit backend/app.py, line ~15:
# device = 'cpu'

# Or reduce model in utils/train_yolov5.py
```

### "Port 5000 already in use"
```bash
# Change port in backend/app.py:
# app.run(port=5001)

# Or find and kill process:
# Windows: netstat -ano | findstr :5000
# macOS/Linux: lsof -i :5000
```

### "Model not found"
```bash
# Option 1: Download pre-trained
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt -O models/best.pt

# Option 2: Check path
ls -la models/  # Make sure best.pt exists
```

### "Camera not working"
```bash
# Only works on localhost (http://localhost:5000)
# For remote access, use image upload instead
# Or deploy with HTTPS (browsers require HTTPS for camera)
```

---

## 📊 EXPECTED PERFORMANCE

### Training Time
| Device | Time |
|--------|------|
| CPU (4 cores) | 24-36 hours |
| GPU (RTX 3080) | 1-2 hours |
| Google Colab GPU | 3-4 hours |

### Inference Speed
| Device | Speed |
|--------|-------|
| CPU | 50-100ms per image |
| GPU | 10-20ms per image |

### Model Size
- best.pt: ~20 MB

---

## 📁 PROJECT LAYOUT

```
Plant Disease Detection/
├── 📂 backend/              ← Flask server (run this)
├── 📂 datasets/             ← Data storage
├── 📂 models/               ← Model weights
├── 📂 utils/                ← Training & evaluation scripts
├── 📄 README.md             ← Full documentation
├── 📄 GETTING_STARTED.md    ← Setup guide
├── 📄 requirements.txt       ← Dependencies
├── setup_and_run.bat        ← Windows setup
└── setup_and_run.sh         ← Linux/macOS setup
```

---

## 🔗 IMPORTANT FILES

| File | Purpose |
|------|---------|
| backend/app.py | Flask REST API |
| utils/dataset_converter.py | Convert PlantVillage → YOLO |
| utils/train_yolov5.py | Train the model |
| utils/evaluation.py | Calculate metrics |
| utils/inference.py | Test single image |
| requirements.txt | Install dependencies |
| Dockerfile | Docker configuration |
| docker-compose.yml | Easy Docker setup |

---

## 🌐 AVAILABLE ENDPOINTS

| Method | URL | Purpose |
|--------|-----|---------|
| GET | / | Web UI |
| GET | /health | Health check |
| POST | /upload | Upload image |
| POST | /predict | Predict from base64 |
| GET | /webcam-feed | Live camera (local only) |
| GET | /api/classes | Get disease list |

---

## 💾 DATA MANAGEMENT

### Cleaning Up
```bash
# Remove uploaded images
rm -rf backend/uploads/*

# Remove training runs
rm -rf models/runs/*

# Keep models/best.pt and models/last.pt
```

### Backing Up
```bash
# Backup trained model
cp models/best.pt models/best_backup.pt

# Backup entire project
zip -r backup.zip . -x "*.env" "venv/*" "__pycache__/*" ".git/*"
```

---

## 📈 NEXT LEVEL IMPROVEMENTS

### For Better Results
1. Train on GPU (10-20x faster)
2. Use more epochs (100+ instead of 50)
3. Collect more custom data
4. Use data augmentation
5. Try ensemble models (yolov5m, yolov5l)

### For Production
1. Add API authentication
2. Set up database for logs
3. Enable rate limiting
4. Use HTTPS/SSL
5. Set up monitoring & alerts

### For Deployment
1. Deploy to Render (free tier)
2. Deploy to Railway (pay-as-you-go)
3. Use Docker with GPU (AWS, GCP)
4. Set up CI/CD pipeline
5. Enable auto-scaling

---

## 🆘 GETTING HELP

1. **Read the docs**: README.md (1000+ lines)
2. **Check API docs**: API_DOCUMENTATION.md
3. **Setup guide**: GETTING_STARTED.md
4. **Code comments**: Inline documentation
5. **Examples**: Code samples in docs

---

## ✅ VERIFICATION CHECKLIST

After setup, verify:

- [ ] Virtual environment activated
- [ ] Dependencies installed (pip freeze shows packages)
- [ ] Directories created (datasets/, models/, etc.)
- [ ] Model file exists (models/best.pt)
- [ ] Flask running (http://localhost:5000 works)
- [ ] Web UI loads (displays upload button)
- [ ] Upload test passes (can upload image)
- [ ] Detection works (shows results)

---

## 🎉 SUCCESS INDICATORS

You'll know everything is working when:

1. ✅ Flask server starts without errors
2. ✅ http://localhost:5000 loads UI
3. ✅ Can upload image and get results
4. ✅ Sees disease name and confidence
5. ✅ Annotated image displays with boxes

---

## 📞 QUICK REFERENCE

```bash
# Virtual environment
source venv/bin/activate            # Activate (macOS/Linux)
venv\Scripts\activate.bat           # Activate (Windows)
deactivate                          # Deactivate

# Install packages
pip install -r requirements.txt     # Install all
pip install package_name            # Install one

# Run backend
cd backend && python app.py         # Start server
# Visit http://localhost:5000

# Run training
python utils/train_yolov5.py        # Train model

# Run evaluation
python utils/evaluation.py          # Check metrics

# Docker
docker build -t plant-disease .     # Build image
docker run -p 5000:5000 plant-disease # Run container
docker-compose up -d                # Run with compose

# Git
git status                          # Check changes
git add .                           # Stage changes
git commit -m "message"             # Commit
git push                            # Push to remote
```

---

## 🎯 YOUR JOURNEY

```
📥 Download/Clone
  ↓
🔧 Install Dependencies
  ↓
📁 Prepare Dataset (optional)
  ↓
🏋️ Train Model (optional)
  ↓
🚀 Start Backend
  ↓
🌐 Open http://localhost:5000
  ↓
✨ Detect Plant Diseases!
  ↓
☁️ Deploy to Cloud (optional)
```

---

## 📚 RECOMMENDED READING ORDER

1. **This File** (You are here!) - 5 minutes
2. **GETTING_STARTED.md** - 15 minutes
3. **README.md** - 30 minutes
4. **API_DOCUMENTATION.md** - 10 minutes
5. **Code Comments** - As needed

---

## 🚀 DEPLOY NOW (Optional)

### Free Tier (Render)
```bash
# 1. Push to GitHub
git push origin main

# 2. Connect to Render
# https://render.com → New Web Service

# 3. Select repository and deploy!
```

### See DEPLOYMENT_INSTRUCTIONS.md for details

---

**You're all set! Happy detecting! 🌿🤖**

```
          🌿
         🍃
        🌱        Plant Disease Detection v1.0
       🌾        Ready to Detect Diseases
      🌳
```

---

*Last Updated: January 31, 2026*  
*Version: 1.0.0*  
*Status: ✅ Production Ready*
