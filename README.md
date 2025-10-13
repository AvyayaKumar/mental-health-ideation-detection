# Suicide Ideation Detection - Research & Production System

A complete machine learning system for detecting suicide ideation in text, designed for both research publication and production deployment for teachers.

## 🎯 Project Goals

1. **Research**: Benchmark transformer models for suicide ideation detection with rigorous experimental methodology suitable for academic publication
2. **Deployment**: Provide a production-ready web application for teachers to analyze student essays with real-time feedback

## 📁 Project Structure

```
suicide-ideation-detection/
├── research/                      # Research & Experimentation
│   ├── src/                      # Training code, models, metrics
│   ├── config/                   # Experiment configurations
│   ├── scripts/                  # Training & evaluation scripts
│   ├── results/                  # Model checkpoints & metrics
│   ├── data/                     # Datasets (not in git)
│   ├── requirements-research.txt # Research dependencies
│   └── README.md                 # Research guide
│
├── deployment/                    # Production Web Application
│   ├── backend/                  # FastAPI application
│   ├── .railway/                 # Railway deployment configs
│   ├── Procfile                  # Process definitions
│   ├── railway.toml              # Railway configuration
│   └── README.md                 # Deployment guide
│
├── docs/                          # Documentation
│   ├── RESEARCH_PLAN.md          # Research methodology
│   ├── EDA_SUMMARY.md            # Data analysis
│   └── PUBLICATION_QUERY.md      # Publication planning
│
└── shared/                        # Shared utilities (if needed)
```

## 🚀 Quick Start

### For Research

```bash
cd research/
pip install -r requirements-research.txt
python src/train.py --config config/distilbert_base.yaml
```

See [research/README.md](research/README.md) for detailed instructions.

### For Deployment

```bash
cd deployment/
cp .env.example .env
# Edit .env with your settings
docker-compose up
```

Visit: http://localhost:8000

See [deployment/README.md](deployment/README.md) for production deployment to Railway.

## 🔬 Research Overview

**Objective**: Benchmark transformer models (DistilBERT, BERT, RoBERTa, ELECTRA, DeBERTa, XLNet) for suicide ideation detection.

**Key Metrics**:
- False Negative Rate (most critical)
- F1 Score
- Recall/Precision
- Accuracy

**Methodology**:
- Stratified 80/10/10 train/val/test split
- 3 random seeds per model
- Comprehensive error analysis
- Statistical validation (McNemar's test)
- SHAP/LIME interpretability

**Timeline**: 7-12 weeks for complete research study

## 🌐 Production System

**Tech Stack**:
- **Backend**: FastAPI (Python 3.11)
- **ML Framework**: PyTorch + Transformers
- **Deployment**: Railway + Docker
- **Background Tasks**: Celery + Redis
- **Model**: DistilBERT (255MB)

**Features**:
- Real-time text analysis
- Integrated Gradients text highlighting
- Teacher feedback collection
- Admin dashboard
- Production monitoring

**Domain**: mentalhealthideation.com

## 📊 Current Status

- ✅ Trained DistilBERT baseline model
- ✅ Production web app deployed
- 🔄 Additional model training in progress
- 🔄 Error analysis & interpretability
- ⏳ Statistical validation pending
- ⏳ Publication preparation pending

## 🤝 Contributing

This is a research & production project. For questions or contributions:
- Research: Check [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md)
- Deployment: Check [deployment/README.md](deployment/README.md)

## 📄 License

[Add your license here]

## ⚠️ Important Notes

- **Research Data**: Not included in repository (large CSV files). Contact maintainer for access.
- **Model Files**: Large model files (255MB) handled via Git LFS or external hosting
- **Sensitive Data**: No actual student essays or personal data in repository
- **Ethics**: This tool is designed to help teachers identify at-risk students, not for surveillance

## 📞 Contact

- Email: avyaya.kumar@gmail.com
- GitHub: @AvyayaKumar

---

**Last Updated**: 2025-10-12
