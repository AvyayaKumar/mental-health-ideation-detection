# Deployment - Production Web Application

Production-ready FastAPI web application for suicide ideation detection, designed for teachers to analyze student essays.

## 🌐 Live Application

**URL**: https://mentalhealthideation.com

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Railway Platform                    │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌──────────────┐      ┌───────────┐                │
│  │  Web Service │──────│   Redis   │                │
│  │  (FastAPI)   │      │  (Cache)  │                │
│  │  Uvicorn     │      └─────┬─────┘                │
│  └──────┬───────┘            │                       │
│         │                    │                       │
│         │  Celery Queue      │                       │
│         └────────────────────┤                       │
│                              │                       │
│                  ┌───────────▼─────────┐             │
│                  │   Celery Worker     │             │
│                  │ (Background Tasks)  │             │
│                  └─────────────────────┘             │
│                                                       │
└─────────────────────────────────────────────────────┘
```

## 🚀 Quick Start - Local Development

### Prerequisites

- Docker Desktop installed
- Python 3.11+
- Redis (via Docker)

### Option 1: Docker Compose (Recommended)

```bash
cd deployment/

# Copy environment file
cp .env.example .env

# Edit .env with your settings
nano .env

# Start all services
docker-compose up

# Visit: http://localhost:8000
```

### Option 2: Local Python

```bash
cd deployment/backend/

# Install dependencies
pip install -r requirements.txt

# Run Redis (in separate terminal)
redis-server

# Run FastAPI app
uvicorn app:app --reload --port 8000

# Run Celery worker (in separate terminal)
celery -A worker.celery_app worker --loglevel=info
```

## 🌩️ Deploy to Railway

### Step 1: Upload Model to Google Drive

Your model is large (255MB), so we download it during build:

1. Go to https://drive.google.com
2. Upload `distilbert-model.zip` (or use existing upload)
3. Right-click → Share → "Anyone with the link" → Viewer
4. Copy FILE_ID from link:
   ```
   https://drive.google.com/file/d/FILE_ID_HERE/view
                                  ^^^^^^^^^^^^
   ```

### Step 2: Deploy to Railway

1. Go to https://railway.app
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway will detect Dockerfile and deploy

### Step 3: Add Redis Service

1. Click "+ New" → "Database" → "Redis"
2. Wait 30 seconds for deployment

### Step 4: Configure Environment Variables

Click "web" service → "Variables" → Add:

```bash
# Required
GDRIVE_FILE_ID=your_google_drive_file_id
MODEL_PATH=/app/backend/models/distilbert-seed42/final_model
SECRET_KEY=your_secret_key_here
ENV=production
LOG_LEVEL=INFO

# Optional
EMAIL_NOTIFICATIONS_ENABLED=false
ADMIN_EMAIL=your_email@example.com
```

Generate SECRET_KEY:
```bash
python -c 'import secrets; print(secrets.token_hex(32))'
```

### Step 5: Add Custom Domain

1. Railway → web service → Settings → Domains
2. "+ Custom Domain" → Enter: `mentalhealthideation.com`
3. Add CNAME records at your domain registrar (Railway shows you what to add)
4. Wait 10-30 minutes for DNS propagation

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `GDRIVE_FILE_ID` | Yes | Google Drive file ID for model | - |
| `MODEL_PATH` | Yes | Path to model directory | `/app/backend/models/...` |
| `SECRET_KEY` | Yes | Secret key for sessions | - |
| `REDIS_URL` | Auto | Redis connection URL | Set by Railway |
| `PORT` | Auto | Server port | Set by Railway |
| `ENV` | No | Environment (dev/production) | `production` |
| `LOG_LEVEL` | No | Logging level | `INFO` |
| `EMAIL_NOTIFICATIONS_ENABLED` | No | Enable email alerts | `false` |

### Railway Configuration Files

- **`.railway/Dockerfile`**: Production Dockerfile with model download
- **`railway.toml`**: Build and deploy settings
- **`Procfile`**: Process definitions (web, worker)
- **`railway.json`**: Service configuration

## 📊 Monitoring & Logs

### View Logs in Railway

1. Railway Dashboard → web service → Deployments
2. Click latest deployment → "View Logs"

### Key Log Messages

**Successful Deployment**:
```
✓ Model downloaded successfully
✓ Model verified: 255M
Application startup complete
Uvicorn running on 0.0.0.0:PORT
```

**Health Check**:
```bash
curl https://your-app.up.railway.app/health

# Should return:
{
  "status": "healthy",
  "model_loaded": true,
  "redis_connected": true
}
```

## 🧪 Testing

### Local Testing

```bash
# Run tests
cd deployment/backend/
pytest tests/

# Test prediction
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling hopeful about the future", "explain": true}'
```

### Production Testing

```bash
# Health check
curl https://mentalhealthideation.com/health

# Test prediction
curl -X POST https://mentalhealthideation.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Sample text", "explain": true}'
```

## 🐛 Troubleshooting

### Build Fails: "Model file not found"

**Cause**: Model didn't download from Google Drive

**Solutions**:
1. Verify Google Drive file is shareable (anyone with link)
2. Check `GDRIVE_FILE_ID` environment variable in Railway
3. Test download manually:
   ```bash
   curl -L "https://drive.google.com/uc?id=YOUR_FILE_ID" -o test.zip
   ```

### Runtime Error: "Model not loaded"

**Check**:
1. Build logs show successful model download
2. `MODEL_PATH` environment variable is correct
3. Model files exist in container

### Redis Connection Error

**Check**:
1. Redis service is running in Railway
2. `REDIS_URL` environment variable is set
3. Both web and worker services have `REDIS_URL`

### High Memory Usage

**Railway limits**: 512MB on hobby plan

**Solutions**:
- Upgrade to Pro plan (2GB RAM)
- Optimize model loading
- Reduce worker concurrency

## 🔄 Updating the Application

### Code Changes

```bash
# Make changes locally
git add .
git commit -m "Update: description"
git push origin main

# Railway auto-deploys in ~3-5 minutes
```

### Model Updates

1. Retrain model in research/
2. Copy to deployment:
   ```bash
   cp -r ../research/results/your-model-seed42 backend/models/
   ```
3. Zip and upload to Google Drive
4. Update `GDRIVE_FILE_ID` (or replace file contents)
5. Trigger Railway redeploy

## 🔐 Security

- ✅ HTTPS automatically enabled (Railway)
- ✅ SECRET_KEY for session security
- ✅ CORS configured properly
- ✅ Environment variables for sensitive data
- ⚠️ TODO: Add rate limiting
- ⚠️ TODO: Add admin authentication

### Adding Rate Limiting

```bash
pip install slowapi

# In app.py:
from slowapi import Limiter
limiter = Limiter(key_func=lambda: request.client.host)

@app.post("/api/predict")
@limiter.limit("10/minute")
async def predict(request: PredictionRequest):
    ...
```

## 📈 Scaling

### Increase Workers

In Railway → web service → Settings → Deploy:
```bash
# Change start command:
uvicorn app:app --host 0.0.0.0 --port $PORT --workers 2
```

### Add Celery Workers

1. Railway → "+ New" → "Empty Service" → Name: `celery-worker`
2. Connect same GitHub repo
3. Set start command: `cd backend && celery -A worker.celery_app worker --loglevel=info`
4. Copy environment variables from web service

## 💰 Cost Estimate

| Service | Cost | Notes |
|---------|------|-------|
| Web Service | $5/month | 512MB RAM |
| Redis | $5/month | Managed Redis |
| Celery Worker | $5/month | Optional |
| **Total** | **$10-15/month** | First $5 free |

Domain: ~$12/year (already owned)
SSL: Free (Let's Encrypt via Railway)

## 📚 API Documentation

Visit: https://mentalhealthideation.com/docs

Interactive API documentation powered by FastAPI.

### Key Endpoints

- `GET /health` - Health check
- `POST /api/predict` - Text prediction
- `GET /docs` - API documentation
- `GET /` - Web interface

## 🔗 Useful Links

- **Railway**: https://railway.app
- **Railway Docs**: https://docs.railway.app
- **FastAPI Docs**: https://fastapi.tiangolo.com
- **Docker Docs**: https://docs.docker.com

---

For research and model training, see [../research/README.md](../research/README.md)
