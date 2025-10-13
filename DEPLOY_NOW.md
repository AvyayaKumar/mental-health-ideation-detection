# 🚀 Deploy to Railway NOW - Fast Track Guide

## ⚡ 15-Minute Deployment

Your code is ready! Let's get it live at **mentalhealthideation.com**.

---

## 📋 Pre-Check

✅ Deployment files ready
✅ Model zip ready (236MB)
✅ GitHub repo updated
✅ Railway Dockerfile configured

**Let's deploy!**

---

## Step 1: Upload Model to Google Drive (5 min)

### 1a. Upload File

Your model zip is here:
```
~/Desktop/suicide-ideation-detection/deployment/distilbert-model.zip
```

**Upload to Google Drive**:
1. Go to https://drive.google.com
2. Click "New" → "File upload"
3. Select: `distilbert-model.zip` (236MB)
4. Wait for upload to complete (~2-3 minutes)

### 1b. Make Shareable

1. Right-click the uploaded file → "Share"
2. Click "Change to anyone with the link"
3. Set permission: **"Viewer"**
4. Click "Copy link"

### 1c. Get File ID

From the link, extract the FILE_ID:
```
https://drive.google.com/file/d/1ABC123xyz456DEF789/view?usp=sharing
                              ^^^^^^^^^^^^^^^^^^^
                              This is your FILE_ID
```

**Example**:
```
Link: https://drive.google.com/file/d/1rVmEb6WqLzNIdVbJOAcJnoN6xiM6NPlA/view
FILE_ID: 1rVmEb6WqLzNIdVbJOAcJnoN6xiM6NPlA
```

**Save this FILE_ID** - you'll need it in Step 3!

---

## Step 2: Deploy to Railway (5 min)

### 2a. Create Project

1. Go to https://railway.app
2. Click "Login" → Use GitHub
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Choose: **mental-health-ideation-detection**
6. Click "Deploy Now"

Railway will start building...

### 2b. Add Redis

While it's building:
1. Click "+ New" → "Database" → "Redis"
2. Wait 30 seconds for Redis to deploy

### 2c. Add Environment Variables

Click "web" service → "Variables" tab → Add these:

**Critical Variables**:
```bash
GDRIVE_FILE_ID=your_file_id_from_step_1c
MODEL_PATH=/app/backend/models/distilbert-seed42/final_model
SECRET_KEY=8c35387cdeff696206e0cee4f70b96a501c86e3e99c70636343ce1dc74147337
ENV=production
LOG_LEVEL=INFO
```

**Optional Variables**:
```bash
EMAIL_NOTIFICATIONS_ENABLED=false
ADMIN_EMAIL=avyaya.kumar@gmail.com
```

**After adding variables**: Railway will automatically rebuild

---

## Step 3: Monitor Build (5-10 min)

### Watch Logs

1. Railway → Click on latest deployment
2. Click "View Logs"

**Look for these success messages**:
```
✓ Downloading model from Google Drive (this may take 2-3 minutes)...
✓ Model downloaded successfully
✓ Extracting model...
✓ Model extracted: 255M
✓ Model verified: 255M
✓ Installing Python dependencies...
✓ Starting server...
```

**Build time**: ~5-10 minutes total

---

## Step 4: Test Deployment (2 min)

### Get Railway URL

Railway → web service → Settings → Domains
Copy the URL (e.g., `your-app-production.up.railway.app`)

### Test Health Endpoint

```bash
curl https://your-app-production.up.railway.app/health
```

**Should return**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "redis_connected": true
}
```

### Test in Browser

1. Open the Railway URL in browser
2. You should see the text analysis interface
3. Enter sample text: "I am feeling hopeful about the future"
4. Click "Analyze"
5. Verify prediction appears with highlighted text

---

## Step 5: Connect Custom Domain (10 min)

### 5a. Configure in Railway

1. Railway → web service → Settings → Domains
2. Click "+ Custom Domain"
3. Enter: **mentalhealthideation.com**
4. Railway will show DNS records (keep tab open!)

### 5b. Update DNS at Domain Registrar

Go to your domain registrar's DNS settings:

**Add CNAME Record 1** (root domain):
```
Type: CNAME
Name: @
Value: [shown by Railway - ends with .up.railway.app]
TTL: 3600
```

**Add CNAME Record 2** (www):
```
Type: CNAME
Name: www
Value: [same as above]
TTL: 3600
```

**Note**: Some registrars don't allow CNAME on root (@). If you get an error:
- Use A record instead
- Or use Cloudflare (free) as DNS provider

### 5c. Wait for DNS Propagation

- Takes 10-30 minutes (sometimes up to 24 hours)
- Check status: https://dnschecker.org

### 5d. Verify HTTPS

After DNS propagates:
- ✅ https://mentalhealthideation.com
- ✅ https://www.mentalhealthideation.com

Railway automatically provisions SSL certificates!

---

## 🎉 Done!

Your app is live at:
- **Production**: https://mentalhealthideation.com
- **Railway URL**: https://your-app.up.railway.app

---

## 🐛 Troubleshooting

### Build fails: "Model file not found"

**Cause**: Google Drive file not shareable or wrong FILE_ID

**Fix**:
1. Check Google Drive file is shareable (anyone with link)
2. Verify `GDRIVE_FILE_ID` in Railway matches your FILE_ID
3. Test download manually:
   ```bash
   curl -L "https://drive.google.com/uc?id=YOUR_FILE_ID" -o test.zip
   ```

### Build fails: "Model file is too small"

**Cause**: `GDRIVE_FILE_ID` not set in Railway

**Fix**: Add the environment variable in Railway

### App crashes at runtime

**Check**:
1. Railway logs for specific error
2. `MODEL_PATH` environment variable is correct
3. Redis is running

### Domain not working

**Check**:
1. DNS records are correct
2. Wait 30 minutes for propagation
3. Try with www. prefix
4. Check: https://dnschecker.org

---

## ⚡ Quick Summary

1. **Upload** model to Google Drive → Get FILE_ID
2. **Deploy** to Railway from GitHub
3. **Add** Redis service
4. **Set** environment variables (especially GDRIVE_FILE_ID)
5. **Wait** for build (~10 min)
6. **Test** Railway URL
7. **Connect** mentalhealthideation.com domain

**Total Time**: 15-20 minutes

---

## 📞 Need Help?

**Build logs**: Railway → Deployment → View Logs
**Health check**: `https://your-app.up.railway.app/health`
**Docs**: See `deployment/README.md` for detailed guide

---

**Ready?** Start with Step 1 - Upload to Google Drive!
