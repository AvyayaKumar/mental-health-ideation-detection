# Model Retraining with Feedback

This guide explains how to retrain your RoBERTa model using teacher feedback collected from production.

## Overview

The feedback system automatically stores teacher corrections in a PostgreSQL database. When you have enough feedback samples, you can retrain the model to improve accuracy on real-world edge cases.

## Setup

### 1. Add PostgreSQL Database to Railway

1. Go to your Railway dashboard
2. Click **"+ New"** → **"Database"** → **"Add PostgreSQL"**
3. Railway automatically creates a `DATABASE_URL` environment variable
4. Your web service will restart and connect to PostgreSQL

✅ The database tables are automatically created on startup.

### 2. Collect Feedback

Teachers can submit feedback directly from the website:
- After each prediction, they can mark it as correct or incorrect
- If incorrect, they specify the correct risk level (LOW, MEDIUM, HIGH)
- All feedback is automatically stored in the database

**Recommendation**: Collect at least **50-100 feedback samples** before retraining.

## Retraining Workflow

### Step 1: Export Feedback from Production

Get your `DATABASE_URL` from Railway:
1. Railway Dashboard → PostgreSQL service → Variables tab
2. Copy the `DATABASE_URL` value

Export all feedback:

```bash
cd deployment

# Export all feedback
DATABASE_URL="postgresql://..." python scripts/export_feedback.py \
  --output feedback_export.csv

# Or export only corrections (recommended)
DATABASE_URL="postgresql://..." python scripts/export_feedback.py \
  --output feedback_export.csv \
  --corrections-only
```

This creates two files:
- `feedback_export.csv` - Full feedback with metadata
- `feedback_export_simple.csv` - Just text + class (for retraining)

### Step 2: Review Feedback Data

Check the exported data to ensure quality:

```bash
# View first 10 rows
head -10 feedback_export.csv

# Check statistics
python -c "import pandas as pd; df = pd.read_csv('feedback_export_simple.csv'); print(df['class'].value_counts())"
```

### Step 3: Retrain the Model

**Requirements**:
- Python 3.8+
- PyTorch with CUDA (GPU highly recommended)
- At least 50 feedback samples
- Original training data

**Option A: Local Training (if you have GPU)**

```bash
# Install dependencies
pip install torch transformers datasets scikit-learn pandas

# Retrain model (takes 1-2 hours on GPU)
python scripts/retrain_model.py \
  --feedback-data feedback_export_simple.csv \
  --original-data ../research/data/raw/Suicide_Detection_Full.csv \
  --output-dir retrained_roberta_v2 \
  --oversample 5 \
  --epochs 3
```

**Option B: Google Colab Training (recommended if no GPU)**

1. Upload `feedback_export_simple.csv` to Google Drive
2. Upload `Suicide_Detection_Full.csv` to Google Drive
3. Use this Colab notebook: [Create one based on retrain_model.py]
4. Download the trained model from Colab

### Step 4: Deploy Retrained Model

After training completes:

```bash
# Zip the model
cd retrained_roberta_v2
zip -r ../retrained_model.zip final_model/

# Upload to Google Drive
# (Upload retrained_model.zip to Google Drive)

# Get the Google Drive File ID
# Share link format: https://drive.google.com/file/d/FILE_ID/view
# Copy the FILE_ID

# Update Railway environment variables
# Railway Dashboard → Web service → Variables
# Update: GDRIVE_FILE_ID=new_file_id

# Redeploy
# Railway Dashboard → Web service → Deployments → Redeploy
```

The new model will be downloaded during the build process and deployed.

### Step 5: Verify Improvement

After deployment:

1. Test the model with previous edge cases
2. Monitor feedback to see if accuracy improves
3. Compare new feedback statistics with old ones

## Monitoring Feedback

### View Feedback Statistics

```bash
curl https://mentalhealthideation.com/api/v1/feedback/stats
```

Returns:
```json
{
  "total_feedback": 127,
  "correct_predictions": 112,
  "incorrect_predictions": 15,
  "accuracy_rate": 88.19,
  "class_breakdown": { ... }
}
```

### Export Feedback for Analysis

```bash
# CSV format
curl "https://mentalhealthideation.com/api/v1/feedback/export?format=csv" > feedback.csv

# JSONL format (for ML pipelines)
curl "https://mentalhealthideation.com/api/v1/feedback/export?format=jsonl" > feedback.jsonl
```

## Advanced Options

### Adjust Feedback Weight

Increase `--oversample` to give feedback more weight during training:

```bash
# Make feedback 10x more important than original data
python scripts/retrain_model.py \
  --feedback-data feedback_export_simple.csv \
  --oversample 10
```

### Custom Training Parameters

```bash
python scripts/retrain_model.py \
  --feedback-data feedback_export_simple.csv \
  --epochs 5 \
  --batch-size 32 \
  --learning-rate 1e-5 \
  --model-name roberta-large  # Use larger model
```

### Incremental Retraining

Start from your current production model instead of base RoBERTa:

1. Download current model from Railway
2. Use it as the base:

```python
# Modify retrain_model.py to load from checkpoint
model_name = "./current_production_model"
```

## Best Practices

1. **Collect Diverse Feedback**: Ensure feedback covers different risk levels and writing styles
2. **Quality Over Quantity**: Review feedback for mislabeling before retraining
3. **Regular Retraining**: Retrain every 100-200 new feedback samples
4. **A/B Testing**: Deploy new model alongside old one to compare performance
5. **Backup Models**: Keep previous model versions in case new model underperforms
6. **Monitor FNR**: False Negative Rate is critical - ensure it doesn't increase

## Troubleshooting

### "Insufficient feedback data"
- Collect more feedback samples (minimum 50 recommended)
- Use `--min-feedback` flag to override (not recommended)

### "Out of memory during training"
- Reduce `--batch-size` (try 8 or 4)
- Use Google Colab with GPU
- Try `roberta-base` instead of `roberta-large`

### "Model accuracy decreased after retraining"
- Check feedback quality (remove mislabeled samples)
- Reduce `--oversample` (feedback may have bias)
- Increase `--epochs` for more training
- Revert to previous model version

## Architecture

```
┌─────────────────┐
│  Production     │
│  Website        │
│  (Railway)      │
└────────┬────────┘
         │
         │ Feedback submissions
         ▼
┌─────────────────┐
│  PostgreSQL     │
│  Database       │
│  (Railway)      │
└────────┬────────┘
         │
         │ Export (scripts/export_feedback.py)
         ▼
┌─────────────────┐
│  Local/Colab    │
│  feedback.csv   │
└────────┬────────┘
         │
         │ Combine with original data
         ▼
┌─────────────────┐
│  Retraining     │
│  (retrain_      │
│   model.py)     │
└────────┬────────┘
         │
         │ Upload to Google Drive
         ▼
┌─────────────────┐
│  Railway        │
│  Deployment     │
│  (new model)    │
└─────────────────┘
```

## Files

- `scripts/export_feedback.py` - Export feedback from production database
- `scripts/retrain_model.py` - Retrain model with feedback data
- `backend/models/database.py` - Database schema for feedback
- `backend/app.py` - Feedback submission API endpoints

## Questions?

For issues or questions, check:
- Railway build logs: `railway logs`
- Application logs: Railway Dashboard → Deployments → Logs
- Database connection: Test with `psql $DATABASE_URL`
