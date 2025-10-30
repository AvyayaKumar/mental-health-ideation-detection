# Automatic Feedback Export System

## Overview

Your feedback system is now **fully automated**. No manual checking needed!

## How It Works

### 🤖 Automatic Daily Checks

The system automatically:
1. **Checks feedback every day at 9 AM** (server time)
2. **Counts incorrect predictions** (corrections from teachers)
3. **Auto-exports when threshold reached** (50 corrections by default)
4. **Logs everything** for monitoring

### 📦 Auto-Export Process

When you reach 50 corrections:
```
✅ Threshold reached! Auto-exporting 52 corrections...
✅ Auto-export complete: /tmp/feedback_exports/corrections_52samples_20251030_090000.csv
📧 Ready to retrain! Check logs for export file location.
```

The system creates:
- `corrections_XXsamples_YYYYMMDD_HHMMSS.csv` - Training data (text, class columns)
- `corrections_XXsamples_YYYYMMDD_HHMMSS_metadata.txt` - Export info and next steps

### 📊 Admin Dashboard

Visit https://mentalhealthideation.com/admin/feedback to see:

1. **Real-time stats** (refreshes every 30 seconds):
   - Total feedback count
   - Model accuracy rate
   - Corrections collected

2. **Retraining readiness indicator**:
   - 📊 "Collecting Feedback" (orange) - Shows progress
   - ✅ "Ready to Retrain!" (green) - Threshold reached

3. **Auto-exported files section** (blue box):
   - Lists all automatically exported files
   - One-click download buttons
   - File creation dates and sizes

## Files Location

Auto-exported files are saved to:
- **Railway**: `/tmp/feedback_exports/` (ephemeral - download ASAP)
- Files persist until container restart
- Download from admin dashboard before redeployment

## Monitoring

### View Logs (Railway Dashboard)

1. Go to Railway dashboard
2. Click your web service
3. Go to "Deployments" tab
4. Click "View Logs"
5. Look for:
   ```
   ✅ Automatic feedback monitoring started
   📊 Feedback check: 15 corrections / 100 total
   📊 Status: Need 35 more corrections before auto-export
   ```

### Check Status Manually

From your terminal:
```bash
python3 deployment/scripts/check_feedback_status.py
```

Output:
```
📊 COLLECTING FEEDBACK
   Current: 15 corrections
   Needed: 35 more
   Progress: 15/50
```

## Schedule

The automatic checker runs:
- ✅ **On startup** (30 seconds after app starts)
- ✅ **Daily at 9 AM** (server timezone)
- ✅ **Manual refresh** via admin dashboard

## Configuration

### Change Threshold

Default is 50 corrections. To change:

Edit `backend/app.py`:
```python
init_monitor(SessionLocal, min_corrections=100)  # Change to 100
```

### Change Schedule

Edit `backend/scheduler.py`:
```python
# Run at different time (e.g., 3 PM)
self.scheduler.add_job(
    self.check_and_export,
    CronTrigger(hour=15, minute=0),  # 3 PM
    ...
)
```

### Change Export Directory

Set environment variable in Railway:
```
EXPORT_DIR=/app/exports
```

## Workflow

### 1. Collecting Phase (0-49 corrections)

**Dashboard shows:**
```
📊 Collecting Feedback
You have 15 corrections so far. Need 35 more before retraining is recommended.
```

**Logs show (daily at 9 AM):**
```
📊 Feedback check: 15 corrections / 100 total
📊 Status: Need 35 more corrections before auto-export
```

### 2. Ready Phase (50+ corrections)

**First time threshold is reached:**
```
✅ Threshold reached! Auto-exporting 52 corrections...
✅ Exported 52 corrections to: /tmp/feedback_exports/corrections_52samples_20251030_090000.csv
✅ Auto-export complete
📧 Ready to retrain! Check logs for export file location.
```

**Dashboard shows:**
```
✅ Ready to Retrain!
You have 52 incorrect predictions. This is enough to retrain your model and improve accuracy.
[📥 Download Corrections]
```

**Auto-exported files section appears:**
```
📦 Auto-Exported Files
Files automatically exported when threshold was reached:

corrections_52samples_20251030_090000.csv
Created: 10/30/2025 9:00 AM | Size: 0.05 MB
[Download]
```

### 3. Download and Retrain

**From Dashboard:**
1. Click download button in auto-exports section
2. Follow retraining guide (see `RETRAINING_GUIDE.md`)

**Or download via API:**
```bash
# List exports
curl https://mentalhealthideation.com/api/v1/feedback/auto-exports

# Download specific file
curl https://mentalhealthideation.com/api/v1/feedback/auto-exports/corrections_52samples_20251030_090000.csv > corrections.csv
```

### 4. Additional Corrections (50 → 60 → 70...)

System continues checking but won't re-export the same count:
```
✅ Ready to retrain (already exported 52 corrections)
```

When you hit 60 corrections, it auto-exports again:
```
✅ Threshold reached! Auto-exporting 60 corrections...
```

## API Endpoints

### Check Status
```bash
GET /api/v1/feedback/stats
```

### List Auto-Exports
```bash
GET /api/v1/feedback/auto-exports
```

Response:
```json
{
  "exports": [
    {
      "filename": "corrections_52samples_20251030_090000.csv",
      "size_bytes": 52480,
      "created_at": "2025-10-30T09:00:00",
      "download_url": "/api/v1/feedback/auto-exports/corrections_52samples_20251030_090000.csv"
    }
  ],
  "export_dir": "/tmp/feedback_exports"
}
```

### Download Export
```bash
GET /api/v1/feedback/auto-exports/{filename}
```

## Troubleshooting

### Not Auto-Exporting?

Check logs for errors:
```
❌ Error in feedback monitor: ...
```

### Can't Find Exported Files?

Railway uses ephemeral storage. Files are lost on redeploy. Download from dashboard immediately.

### Want Notifications?

Add to your cron (local machine):
```bash
# Check daily and download when ready
0 9 * * * cd /path/to/deployment && python3 scripts/check_feedback_status.py --auto-download
```

Or set up email notifications (edit `scheduler.py`).

## Summary

**You don't need to do anything!**

Just:
1. ✅ Use your website normally
2. ✅ Visit dashboard occasionally
3. ✅ Download exports when they appear
4. ✅ Retrain when ready

The system handles everything automatically. 🎉
