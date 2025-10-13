# Teacher Feedback Loop System

A human-in-the-loop system for continuously improving the mental health ideation detection model through teacher feedback.

## Overview

The feedback system allows teachers to report when the AI model makes incorrect predictions. This data is collected, analyzed, and can be used to retrain and improve the model over time.

## Architecture

```
┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│   Teacher    │────▶│  Web Interface │────▶│   Backend    │
│  Analyzes    │     │  (Feedback UI) │     │ (FastAPI +   │
│   Essay      │     └────────────────┘     │  SQLAlchemy) │
└──────────────┘                            └──────┬───────┘
                                                   │
                                                   ▼
                                            ┌──────────────┐
                                            │   SQLite DB  │
                                            │  (Feedback   │
                                            │   Storage)   │
                                            └──────┬───────┘
                                                   │
                                                   ▼
                                            ┌──────────────┐
                                            │    Export    │
                                            │ CSV / JSONL  │
                                            │  (Retraining)│
                                            └──────────────┘
```

## Features

### 1. **In-App Feedback Collection**
   - After each analysis, teachers see two buttons:
     - ✓ **Prediction Correct** - Confirms the model was accurate
     - ✗ **Prediction Incorrect** - Reports an error
   - If incorrect, teachers select the correct risk level and can add notes

### 2. **Database Storage**
   - All feedback is stored in SQLite database (`feedback.db`)
   - Schema tracks:
     - Original text and prediction
     - Teacher's correction
     - Model confidence scores
     - Optional teacher notes
     - Timestamps and status

### 3. **Admin Dashboard**
   - View all feedback submissions at `/admin/feedback`
   - Statistics dashboard showing:
     - Overall model accuracy
     - Per-class accuracy (LOW, MEDIUM, HIGH)
     - Correct vs incorrect predictions
   - Filter by status and correctness
   - Click any row to view full details

### 4. **Data Export**
   - **CSV Export**: For analysis in Excel/Google Sheets
   - **JSONL Export**: Ready for machine learning retraining
   - Both formats available via admin dashboard or API

## Usage Guide

### For Teachers

#### Providing Feedback

1. **Analyze an Essay**
   - Paste text into the analyzer
   - Click "Analyze" and wait for results

2. **Review the Prediction**
   - Check if the risk level (LOW/MEDIUM/HIGH) is accurate
   - Read the highlighted problematic sentences

3. **Submit Feedback**
   - If correct: Click "✓ Prediction Correct"
   - If incorrect:
     1. Click "✗ Prediction Incorrect"
     2. Select the correct risk level from dropdown
     3. (Optional) Add notes explaining why
     4. Click "Submit Feedback"

4. **Confirmation**
   - You'll see a success message
   - Your feedback is saved and helps improve future predictions

#### Example Feedback Scenarios

**Scenario 1: False Positive**
- **Model says**: HIGH risk
- **Actual**: LOW risk
- **Teacher action**: Report incorrect, select LOW, note "Text is metaphorical, not literal"

**Scenario 2: False Negative**
- **Model says**: LOW risk
- **Actual**: HIGH risk
- **Teacher action**: Report incorrect, select HIGH, note "Subtle indicators missed by model"

**Scenario 3: Correct Prediction**
- **Model says**: MEDIUM risk
- **Actual**: MEDIUM risk
- **Teacher action**: Click "Prediction Correct"

### For Administrators

#### Viewing the Dashboard

1. Navigate to: `https://mentalhealthideation.com/admin/feedback`
2. View statistics:
   - Total feedback submissions
   - Overall accuracy rate
   - Breakdown by risk level
3. Review individual submissions in the table

#### Filtering Feedback

Use dropdown filters to focus on specific feedback:
- **Status**: Pending / Reviewed / Incorporated
- **Correctness**: All / Correct / Incorrect

Click "Refresh" to reload data.

#### Exporting Data

**For Analysis:**
- Click "Export CSV" to download all feedback for spreadsheet analysis

**For Model Retraining:**
- Click "Export JSONL" to download data formatted for ML training

## API Endpoints

### Submit Feedback
```http
POST /api/v1/feedback
Content-Type: application/json

{
  "text": "The original essay text...",
  "original_prediction": "HIGH",
  "original_score": 0.92,
  "original_confidence": 0.88,
  "teacher_correction": "LOW",
  "teacher_notes": "Text is fictional writing, not concerning",
  "teacher_id": "teacher123"  // optional
}
```

**Response:**
```json
{
  "id": 42,
  "message": "Thank you for your feedback! This will help improve the model.",
  "is_correct": false
}
```

### List All Feedback
```http
GET /api/v1/feedback?skip=0&limit=100&status=pending&is_correct=false
```

**Query Parameters:**
- `skip`: Pagination offset (default 0)
- `limit`: Max results (default 100)
- `status`: Filter by status (pending/reviewed/incorporated)
- `is_correct`: Filter by correctness (true/false)

**Response:**
```json
{
  "total": 42,
  "skip": 0,
  "limit": 100,
  "data": [
    {
      "id": 1,
      "text": "...",
      "original_prediction": "HIGH",
      "teacher_correction": "LOW",
      "is_correct": false,
      "created_at": "2025-10-12T10:30:00",
      ...
    }
  ]
}
```

### Get Statistics
```http
GET /api/v1/feedback/stats
```

**Response:**
```json
{
  "total_feedback": 150,
  "correct_predictions": 140,
  "incorrect_predictions": 10,
  "accuracy_rate": 93.33,
  "class_breakdown": {
    "LOW": {
      "correct": 50,
      "incorrect": 2,
      "total": 52,
      "accuracy": 96.15
    },
    "MEDIUM": { ... },
    "HIGH": { ... }
  }
}
```

### Export Feedback
```http
GET /api/v1/feedback/export?format=csv
GET /api/v1/feedback/export?format=jsonl&is_correct=false
```

**Query Parameters:**
- `format`: csv or jsonl (default csv)
- `status`: Filter by status
- `is_correct`: Filter by correctness

**Response:** Downloadable file with feedback data

## Model Retraining Workflow

### 1. Collect Feedback

Run the system for a period (e.g., 1 month) to collect teacher feedback:
- Aim for at least 100-200 feedback submissions
- Ensure diverse examples (different risk levels, text types)

### 2. Export Training Data

```bash
# Export incorrect predictions for retraining
curl "https://mentalhealthideation.com/api/v1/feedback/export?format=jsonl&is_correct=false" -o incorrect_predictions.jsonl

# Export all feedback for analysis
curl "https://mentalhealthideation.com/api/v1/feedback/export?format=csv" -o all_feedback.csv
```

### 3. Analyze Feedback

Review the CSV to identify patterns:
- Which risk levels have most errors?
- Are there common types of texts that confuse the model?
- Any systematic biases?

### 4. Prepare Retraining Dataset

The JSONL format is ready for training:
```json
{"text": "Essay text...", "label": "LOW", "original_prediction": "HIGH", "is_correct": false, "metadata": {...}}
{"text": "Essay text...", "label": "MEDIUM", "original_prediction": "LOW", "is_correct": false, "metadata": {...}}
```

### 5. Retrain Model

**Option A: Fine-tune existing model**
```python
# In research/ directory
from transformers import AutoModelForSequenceClassification, Trainer

# Load feedback data
feedback_data = load_jsonl('incorrect_predictions.jsonl')

# Combine with original training data (weighted)
training_data = combine_datasets(original_data, feedback_data, weights=[0.9, 0.1])

# Fine-tune
trainer = Trainer(model=model, train_dataset=training_data)
trainer.train()
```

**Option B: Add to training set and retrain from scratch**
```python
# Add feedback to original dataset
augmented_data = pd.concat([original_df, feedback_df])

# Retrain model completely
python train_model.py --data augmented_data.csv --epochs 5
```

### 6. Evaluate Improvements

```python
# Test on held-out feedback examples
from sklearn.metrics import classification_report

y_true = [f['label'] for f in test_feedback]
y_pred = model.predict(test_texts)

print(classification_report(y_true, y_pred))
```

### 7. Deploy Updated Model

```bash
# Replace model files
cd deployment/backend/models/distilbert-seed42/final_model
cp ~/research/new_model/* .

# Zip and upload to Google Drive
cd deployment/
zip -r distilbert-model-v2.zip backend/models/

# Upload to Drive, update GDRIVE_FILE_ID in Railway
# Railway will rebuild and deploy automatically
```

## Database Schema

### Feedback Table

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| text | Text | Original essay/text |
| original_prediction | String | Model's prediction (LOW/MEDIUM/HIGH) |
| original_score | Float | Model's score (0-1) |
| original_confidence | Float | Model's confidence (0-1) |
| teacher_correction | String | Correct risk level |
| teacher_notes | Text | Optional explanation |
| teacher_id | String | Optional teacher identifier |
| is_correct | Boolean | Whether prediction was correct |
| status | String | pending/reviewed/incorporated |
| created_at | DateTime | Submission timestamp |
| reviewed_at | DateTime | When reviewed |
| used_for_training | Boolean | Whether used in retraining |

### FeedbackStats Table

Tracks aggregate statistics over time:
- Total feedback count
- Accuracy metrics
- Per-class performance

## Best Practices

### For Teachers

1. **Be Specific**: When reporting incorrect predictions, provide detailed notes
2. **Be Consistent**: Use the same criteria for all evaluations
3. **Report Good Predictions Too**: Positive feedback helps track overall accuracy
4. **Context Matters**: Consider the full essay context, not just flagged sentences

### For Administrators

1. **Review Regularly**: Check dashboard weekly to monitor trends
2. **Investigate Patterns**: Look for systematic errors to address
3. **Communicate Results**: Share accuracy stats with teachers
4. **Plan Retraining**: Aim to retrain model quarterly or when accuracy drops

### For Model Development

1. **Balance Dataset**: Don't only retrain on errors; include correct predictions
2. **Weight Recent Data**: Give more weight to recent feedback
3. **Validate Changes**: Always test on held-out data before deploying
4. **Version Models**: Keep track of model versions and their performance

## Privacy & Security

### Data Handling

- **Anonymization**: No student names or IDs should be in submitted text
- **Teacher IDs**: Optional and can be anonymized
- **Access Control**: Consider adding authentication to admin dashboard in production

### Recommended Security Enhancements

1. **Add Admin Authentication**
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != "admin" or credentials.password != os.getenv("ADMIN_PASSWORD"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials.username

@app.get("/admin/feedback")
def admin_page(request: Request, username: str = Depends(verify_admin)):
    return templates.TemplateResponse("admin_feedback.html", {"request": request})
```

2. **Rate Limiting**
```python
from slowapi import Limiter

limiter = Limiter(key_func=lambda: request.client.host)

@app.post("/api/v1/feedback")
@limiter.limit("10/minute")
def submit_feedback(...):
    ...
```

3. **HTTPS Only**: Ensure Railway HTTPS is enforced (it is by default)

## Troubleshooting

### Database Not Initializing

**Symptom**: "Table does not exist" errors

**Solution**:
```bash
# SSH into Railway deployment or run locally
python -c "from models.database import init_db; init_db()"
```

### Feedback Not Saving

**Symptom**: No error but feedback doesn't appear in dashboard

**Check**:
1. Database file permissions
2. Railway logs for SQLAlchemy errors
3. Network connectivity between frontend and API

### Export Files Empty

**Symptom**: Downloaded CSV/JSONL has no data

**Check**:
1. Feedback actually exists in database
2. Filters aren't too restrictive
3. Database connection is working

## Future Enhancements

### Planned Features

- [ ] **Active Learning**: Prioritize uncertain predictions for teacher review
- [ ] **Batch Upload**: Upload multiple essays for batch feedback
- [ ] **Email Notifications**: Alert admin when accuracy drops below threshold
- [ ] **A/B Testing**: Test new model versions against current production
- [ ] **Automated Retraining**: Schedule automatic retraining when enough feedback collected
- [ ] **Teacher Analytics**: Track which teachers provide most helpful feedback
- [ ] **Consensus Building**: Require multiple teachers to agree on corrections

### Integration Ideas

- **Learning Management Systems (LMS)**: Canvas, Blackboard, Moodle integration
- **Google Classroom**: Direct integration with Google Classroom assignments
- **Student Information Systems**: Link to student IDs (with privacy controls)
- **Microsoft Teams / Slack**: Notifications for high-risk detections

## Support

For questions or issues:
1. Check Railway deployment logs
2. Review API documentation at `/docs`
3. Test endpoints manually with curl or Postman
4. Check GitHub issues (if public repo)

---

**Version**: 1.0
**Last Updated**: October 2025
**Maintained By**: Mental Health Ideation Detection Project
