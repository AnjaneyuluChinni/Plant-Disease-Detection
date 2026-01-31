# Plant Disease Detection - Render Deployment Guide

## Prerequisites
- Render account (https://render.com)
- GitHub repository with your project
- Git installed locally

## Step 1: Prepare Your Project for Render

### 1.1 Update Python Version (render.yaml or Dockerfile)
If using `render.yaml`, Render will use **Python 3.13** by default. Ensure compatibility by:

```bash
# Use the Python 3.13 compatible requirements
cp requirements-render.txt requirements.txt
```

### 1.2 Create Dockerfile (Recommended for better control)

```dockerfile
# Use official Python 3.13 slim image
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies for OpenCV and ML
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-render.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create uploads directory
RUN mkdir -p uploads models

# Set environment variables
ENV FLASK_APP=backend.app:app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5000

# Run with Gunicorn
CMD ["gunicorn", "--workers", "1", "--threads", "2", "--worker-class", "gthread", "--bind", "0.0.0.0:5000", "backend.app:app"]
```

Save as: `Dockerfile` in project root

### 1.3 Create .dockerignore

```
__pycache__
*.pyc
.git
.gitignore
.env
*.pt
uploads/*
runs/*
datasets/*
.venv
venv
node_modules
```

Save as: `.dockerignore` in project root

## Step 2: Push to GitHub

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Prepare for Render deployment"

# Push to GitHub
git push origin main
```

## Step 3: Deploy on Render

### Option A: Using Docker (Recommended)

1. **Create New Service**
   - Go to https://render.com/dashboard
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Select the repo branch (main)

2. **Configure Service**
   - **Name**: `plant-disease-detection`
   - **Environment**: Docker
   - **Build Command**: (Leave default)
   - **Start Command**: (Leave default - will use Dockerfile)
   - **Plan**: Free or Starter (ML models require ~500MB)

3. **Environment Variables** (Optional)
   - You can add custom env vars if needed
   - Flask will run with defaults

4. **Deploy**
   - Click "Create Web Service"
   - Wait for build and deployment (~5-10 minutes)

### Option B: Using render.yaml (YAML Config)

1. **Create `render.yaml` in project root**:

```yaml
services:
  - type: web
    name: plant-disease-detection
    env: docker
    plan: starter
    healthCheckPath: /health
    envVars:
      - key: FLASK_ENV
        value: production
```

2. **Push to GitHub**:
```bash
git add render.yaml
git commit -m "Add render config"
git push origin main
```

3. **Deploy**:
   - Connect repository to Render
   - Render automatically detects and uses render.yaml

## Step 4: Verify Deployment

1. **Check Logs**
   - Go to your service dashboard
   - View "Logs" tab
   - Look for "Running on http://..." message

2. **Test Health Endpoint**
   ```
   curl https://<your-service-name>.onrender.com/health
   ```
   
   Expected response:
   ```json
   {
     "status": "healthy",
     "model_loaded": true,
     "timestamp": "2026-01-31T..."
   }
   ```

3. **Access Web UI**
   - Open: `https://<your-service-name>.onrender.com`
   - Upload test image
   - Verify detection works

## Python 3.13 Compatibility Issues & Solutions

### Known Issues & Fixes:

| Package | Issue | Solution |
|---------|-------|----------|
| `numpy` | Old versions fail on 3.13 | Use `>=1.26.0` ✓ |
| `torch` | Needs 2.1.0+ for 3.13 | Use `>=2.1.0` ✓ |
| `scipy` | Compatibility issue | Use `>=1.11.0` ✓ |
| `Pillow` | Needs 10.0.0+ | Use `>=10.0.0` ✓ |

### If You Get Build Errors:

1. **torch wheels not found**:
   ```bash
   # Add this to Dockerfile before pip install
   RUN pip install --upgrade pip setuptools wheel
   ```

2. **Memory errors during pip install**:
   ```bash
   # Render Free plan has 512MB RAM
   # Use models: Free/Starter plan or reduce worker count
   ```

3. **Timeout during deployment**:
   - Increase timeout in Render settings
   - Or use requirements-render.txt (optimized)

## Production Optimization

### 1. Use Smaller Model (Optional)
```python
# In backend/app.py - use yolov5n (nano) instead of yolov5s
model = YOLO('yolov5n.pt')  # Faster, less memory
```

### 2. Add Gunicorn Configuration
Create `gunicorn.conf.py`:
```python
workers = 1
threads = 2
worker_class = 'gthread'
worker_connections = 100
timeout = 120
keepalive = 5
max_requests = 1000
```

### 3. Monitor Performance
- Render dashboard shows CPU, memory usage
- Adjust worker count based on metrics
- Use `/health` endpoint for monitoring

## Continuous Deployment

Every time you push to GitHub:
1. Render automatically detects changes
2. Rebuilds Docker image
3. Deploys new version
4. Old version rolled back if error

```bash
# Deploy updates
git add .
git commit -m "Update models"
git push origin main
# Render auto-deploys!
```

## Troubleshooting

### 1. "Model not loaded" Error
```
Solution: Models take time to load on first request
- Wait 2-3 minutes for cold start
- Models cached after first load
```

### 2. "Port already in use"
```
Solution: Use environment PORT variable
- Gunicorn reads PORT automatically
- No changes needed
```

### 3. "CUDA not available" (Expected for CPU)
```
Solution: Application automatically uses CPU
- No CUDA setup needed
- Performance is acceptable for demo
```

### 4. Out of Memory
```
Solutions:
- Use nano model: yolov5n.pt
- Reduce worker count to 1
- Upgrade to paid plan
```

## Cost Estimation

| Plan | Monthly Cost | Include |
|------|-------------|---------|
| **Free** | $0 | 750 hours, 512MB RAM (spins down) |
| **Starter** | $7 | Always on, 0.5 GB RAM |
| **Standard** | $25+ | 4 GB RAM, better performance |

**Recommendation**: Start with Free, upgrade to Starter if needed

## Support & Monitoring

### Render Metrics Dashboard
- Check CPU usage
- Monitor memory
- View request rates
- Track build times

### Health Checks
```bash
# Monitor from terminal
while true; do
  curl https://<service>.onrender.com/health && echo "" || echo "Down"
  sleep 60
done
```

## Rollback (If needed)

1. Go to Render dashboard
2. Click "Deployments" tab
3. Select previous version
4. Click "Redeploy"

---

## Summary Checklist

- [ ] Updated requirements-render.txt for Python 3.13
- [ ] Created Dockerfile with Python 3.13
- [ ] Created .dockerignore
- [ ] Committed changes to Git
- [ ] Pushed to GitHub
- [ ] Created Render service (Docker)
- [ ] Service deployed successfully
- [ ] /health endpoint returns 200
- [ ] Web UI loads at https://<service>.onrender.com
- [ ] Test image upload works
- [ ] Detection results appear

---

**Deployment Time**: ~10-15 minutes
**Service Ready**: After first successful health check
**Cold Start**: ~30 seconds (spins up container)
**Model Load**: ~2-3 minutes on first request
