# Quick Reference Card

## 📍 Location
```
~/Desktop/suicide-ideation-detection/
```

## 🎯 Two Main Paths

### Path 1: Research (Train Models)
```bash
cd ~/Desktop/suicide-ideation-detection/research/

# Train a model
python src/train.py --config config/distilbert_base.yaml

# Evaluate
python scripts/evaluate.py --model_path results/distilbert-seed42/final_model
```

**See**: `research/README.md`

---

### Path 2: Deployment (Web App)
```bash
cd ~/Desktop/suicide-ideation-detection/deployment/

# Local dev
docker-compose up

# Deploy to Railway
# See deployment/README.md
```

**See**: `deployment/README.md`

---

## 📚 Documentation Map

| File | Purpose |
|------|---------|
| `README.md` | Project overview |
| `SETUP.md` | Quick start guide |
| `CONSOLIDATION_SUMMARY.md` | What I did, what's next |
| `ARCHIVE_OLD_PROJECTS.md` | How to archive old folders |
| `research/README.md` | Research/training guide |
| `deployment/README.md` | Deployment guide |
| `docs/RESEARCH_PLAN.md` | Publication plan |

---

## ⚡ Quick Commands

### Research
```bash
# Train DistilBERT
cd research && python src/train.py --config config/distilbert_base.yaml

# Train BERT
cd research && python src/train.py --config config/bert_base.yaml
```

### Deployment
```bash
# Local test
cd deployment && docker-compose up

# Check health
curl http://localhost:8000/health
```

### Git
```bash
# Initialize
git init && git add . && git commit -m "Initial commit"

# Connect to GitHub
git remote add origin https://github.com/AvyayaKumar/your-repo.git
git push -u origin main
```

---

## 🆘 Common Issues

**"Module not found"**
```bash
cd research && pip install -r requirements-research.txt
# or
cd deployment && pip install -r backend/requirements.txt
```

**"Model not found"**
```bash
# Check model exists
ls research/results/distilbert-seed42/final_model/

# Check symlink
ls -la deployment/backend/models/
```

**"Port in use"**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill
```

---

## 🎯 Next Actions

1. Archive old folders (see `ARCHIVE_OLD_PROJECTS.md`)
2. Test research code works
3. Test deployment works locally
4. Deploy to Railway (see `deployment/README.md`)

---

**Main Guide**: See `CONSOLIDATION_SUMMARY.md`
