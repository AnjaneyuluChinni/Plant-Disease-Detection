# Render Deployment Checklist

## Before Deployment

### 1. Local Testing (Verify everything works locally)
```bash
# Test with Python 3.13 requirements
pip install -r requirements-render.txt

# Run backend
python backend/app.py

# Test in browser
# http://localhost:5000
```

- [ ] Backend starts without errors
- [ ] Frontend loads
- [ ] Image upload works
- [ ] Model loads (check logs for "Model loaded")
- [ ] Detection results appear

### 2. Prepare Repository

```bash
# Make sure all changes are committed
git status

# Ensure requirements-render.txt is committed
git add requirements-render.txt
git add Dockerfile
git add .dockerignore
git add RENDER_DEPLOYMENT.md

git commit -m "Prepare for Render deployment with Python 3.13"
git push origin main
```

- [ ] All files committed to Git
- [ ] Latest code pushed to GitHub
- [ ] Repository is public or Render has access

## Deployment to Render

### Step 1: Create Render Account & Connect GitHub
1. Visit https://render.com
2. Sign up with GitHub
3. Authorize Render to access your GitHub repos
4. Dashboard shows "GitHub Connected"

- [ ] Render account created
- [ ] GitHub authorized
- [ ] Repository visible in Render

### Step 2: Create New Web Service
1. Go to Render Dashboard
2. Click "New" → "Web Service"
3. Select your GitHub repository
4. Select branch: `main`

- [ ] Repository selected
- [ ] Branch is `main`

### Step 3: Configure Service Settings

| Setting | Value |
|---------|-------|
| **Name** | `plant-disease-detection` |
| **Environment** | `Docker` |
| **Build Command** | (Empty - uses Dockerfile) |
| **Start Command** | (Empty - uses Dockerfile) |
| **Plan** | `Free` (or Starter if you want always-on) |

- [ ] Service name set
- [ ] Environment set to Docker
- [ ] Plan selected (Free or Starter)

### Step 4: Add Environment Variables (Optional)
If you need custom settings, add them:
- `FLASK_ENV`: `production`
- `PYTHONUNBUFFERED`: `1`

- [ ] Environment variables set (or skip for defaults)

### Step 5: Deploy
Click **"Create Web Service"**

Render will:
1. Clone your repository
2. Build Docker image (3-5 minutes)
3. Push to Render's registry
4. Start container
5. Run health checks
6. Show service URL

- [ ] Service created
- [ ] Build started (check "Build" log)
- [ ] Waiting for "Running on..." message

## Post-Deployment Verification

### 1. Check Logs
Go to Service Dashboard → Logs tab

Look for:
```
[2026-01-31 ...] Listening on port 5000
✓ Model loaded successfully (trained) - 38 plant disease classes
Starting Flask server at http://localhost:5000
```

- [ ] No error messages in logs
- [ ] "Model loaded" message appears
- [ ] Service shows "running" status

### 2. Test Health Endpoint
```bash
# Replace with your actual service URL
curl https://plant-disease-detection-xxxx.onrender.com/health

# Expected response (200 OK):
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-01-31T..."
}
```

- [ ] /health endpoint returns 200
- [ ] model_loaded is true
- [ ] No error messages

### 3. Test Web Interface
1. Open: `https://plant-disease-detection-xxxx.onrender.com`
2. Verify it loads (may take 30-60 seconds on first visit)
3. Upload a test image
4. Verify detection results appear

- [ ] Web page loads
- [ ] Upload interface visible
- [ ] Can select image
- [ ] Analysis completes
- [ ] Results displayed

### 4. Test Detection
1. Download test image: [Sample Soybean Leaf](datasets/raw_sample/Soybean___healthy/)
2. Upload through web interface
3. Verify it detects disease correctly

- [ ] Image uploads successfully
- [ ] Detection runs
- [ ] Results show confidence %
- [ ] Annotated image displays

## Troubleshooting

### Issue: "Build failed"

**Check Logs for**:
- Pip install errors → Update requirements-render.txt
- Missing system libs → Update Dockerfile RUN apt-get
- Docker syntax errors → Verify Dockerfile format

**Solution**:
```bash
# Fix issue locally first
pip install -r requirements-render.txt  # Test pip
docker build -t test .  # Test Docker build

# Push fix to GitHub
git add requirements-render.txt Dockerfile
git commit -m "Fix build issues"
git push origin main

# Redeploy from Render dashboard
# Click "Deployments" tab → Select failed build → "Redeploy"
```

### Issue: Service running but health check fails

**Check**:
- Is `/health` endpoint accessible?
- Test: `curl https://<service>.onrender.com/health`

**Solution**:
```bash
# Check logs for startup errors
# Wait 2-3 minutes for model to load
# Try accessing again after model loads
```

### Issue: "Out of memory" errors

**Solutions**:
1. Use smaller model (yolov5n instead of yolov5s)
2. Upgrade to Starter plan (0.5GB RAM)
3. Reduce worker count (already set to 1)

### Issue: Upload times out

**Solutions**:
1. Timeout is set to 120 seconds (Dockerfile)
2. Increase if needed:
   ```dockerfile
   "--timeout", "180",  # 3 minutes
   ```
3. Commit and redeploy

## Performance Tips

### 1. Cold Starts (First request after 15 mins of inactivity)
- Free tier spins down container
- First request takes 30-60 seconds
- Upgrade to Starter to prevent spin-down

### 2. Model Loading
- First request loads model (~2-3 minutes)
- Subsequent requests are fast
- Model stays in memory for faster inference

### 3. Optimize for Speed
- Use `/health` to keep service warm
- Add scheduled pings to prevent spin-down
- Monitor CPU/Memory in Render dashboard

## Continuous Deployment

Every push to main triggers redeploy:

```bash
# Make changes
# ... edit files ...

# Commit and push
git add .
git commit -m "Update feature"
git push origin main

# Render automatically:
# 1. Detects changes
# 2. Rebuilds Docker image
# 3. Deploys new version
# 4. Rollback if error
```

## Monitoring

### Via Render Dashboard
- **Metrics**: CPU, Memory, Network
- **Logs**: Real-time application logs
- **Deployments**: Version history
- **Alerts**: Email notifications on errors

### Via Health Checks
```bash
# Monitor continuously
watch -n 60 'curl -s https://plant-disease-detection-xxxx.onrender.com/health | jq'

# Or simple loop
while true; do
  curl https://plant-disease-detection-xxxx.onrender.com/health && echo "OK" || echo "FAIL"
  sleep 60
done
```

## Cost & Limits

| Plan | Cost | RAM | CPU | Hours/Month |
|------|------|-----|-----|------------|
| **Free** | $0 | 512MB | 0.5 CPU | 750h (spins down) |
| **Starter** | $7 | 0.5GB | 0.5 CPU | Always on |
| **Standard** | $25+ | 4GB | 2 CPU | Always on |

**Recommendation**: 
- Start with Free for testing
- Upgrade to Starter for production ($7/month)

## Support Resources

- [Render Docs](https://render.com/docs)
- [Docker Docs](https://docs.docker.com)
- [Gunicorn Docs](https://docs.gunicorn.org)
- [Flask Docs](https://flask.palletsprojects.com)

## Success Checklist

Final verification before launching:

- [ ] Local testing passes
- [ ] All code committed to GitHub
- [ ] Render service created
- [ ] Docker builds successfully
- [ ] Service starts without errors
- [ ] /health endpoint returns 200
- [ ] Web interface loads
- [ ] Can upload image
- [ ] Detection works correctly
- [ ] Results display properly
- [ ] Multiple uploads work
- [ ] No errors in logs

---

**Service Ready!** 🎉

Your Plant Disease Detection app is now live at:
```
https://plant-disease-detection-xxxx.onrender.com
```

Replace `xxxx` with your actual Render service ID.

Share the link with others to test the AI plant disease detection!
