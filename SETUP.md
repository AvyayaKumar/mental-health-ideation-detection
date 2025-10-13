# Quick Setup Guide

## 🎯 Choose Your Path

### Path 1: Research (Training Models)

If you want to train models and conduct experiments:

```bash
cd research/
pip install -r requirements-research.txt
python src/train.py --config config/distilbert_base.yaml
```

See [research/README.md](research/README.md) for details.

---

### Path 2: Deployment (Run Web App)

If you want to deploy the production application:

#### Local Development

```bash
cd deployment/
cp .env.example .env
docker-compose up
# Visit: http://localhost:8000
```

#### Production Deployment

See [deployment/README.md](deployment/README.md) for Railway deployment guide.

---

## 📁 Project Structure

```
suicide-ideation-detection/
├── research/          # ML training & experiments
├── deployment/        # Production web app
├── docs/             # Research documentation
└── shared/           # Shared utilities
```

---

## 🔧 Prerequisites

### For Research
- Python 3.11+
- GPU (recommended) or CPU
- ~10GB disk space
- wandb account (for experiment tracking)

### For Deployment
- Docker Desktop (for local)
- Railway account (for production)
- Google Drive (for model hosting)

---

## 🚀 Quick Commands

### Research
```bash
# Train DistilBERT
cd research/
python src/train.py --config config/distilbert_base.yaml

# Evaluate model
python scripts/evaluate.py --model_path results/distilbert-seed42/final_model

# Run interpretability analysis
python scripts/interpret.py --model_path results/distilbert-seed42/final_model
```

### Deployment
```bash
# Local development
cd deployment/
docker-compose up

# Deploy to Railway
# See deployment/README.md
```

---

## 📊 Data Setup

### Research Data

Place your dataset in:
```
research/data/raw/Suicide_Detection.csv
```

Or modify `research/src/dataset.py` to point to your data location.

### Model Files

Model files are stored in:
```
research/results/model-name-seed/final_model/
```

For deployment, models are linked from research/ to deployment/.

---

## 🐛 Common Issues

### "Dataset not found"
- Ensure CSV is in `research/data/raw/`
- Check filename matches in `src/dataset.py`

### "Model not found"
- Check symlink: `deployment/backend/models/` should link to `research/results/`
- Or copy model manually to deployment

### "CUDA out of memory"
- Reduce batch size: `--batch_size 8`
- Use CPU: `--device cpu`

### "Port already in use"
- Change port in docker-compose.yml
- Or kill existing process: `lsof -ti:8000 | xargs kill`

---

## 📚 Documentation

- **[Main README](README.md)**: Project overview
- **[Research README](research/README.md)**: Training guide
- **[Deployment README](deployment/README.md)**: Deployment guide
- **[Research Plan](docs/RESEARCH_PLAN.md)**: Publication plan

---

## 🤝 Need Help?

1. Check relevant README in research/ or deployment/
2. Review troubleshooting sections
3. Check GitHub issues
4. Contact: avyaya.kumar@gmail.com

---

**Quick Links**:
- Research: `cd research/`
- Deployment: `cd deployment/`
- Docs: `cd docs/`
