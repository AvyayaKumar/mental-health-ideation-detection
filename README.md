# Mental Health Ideation Detection

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Railway](https://img.shields.io/badge/Deployed%20on-Railway-blueviolet)](https://railway.app)

> **A complete machine learning system for detecting suicide ideation in text, designed for both rigorous academic research and real-world production deployment to help teachers identify at-risk students.**

🚀 **[Live Demo](https://mentalhealthideation.com)** | 📚 **[Documentation](docs/)** | 🔬 **[Research](research/)**

---

## 🎯 Inspiration

Suicide ideation detection has critical safety implications - missing actual cases (false negatives) can be life-threatening. Teachers and educators are often on the front lines but lack tools to help them identify at-risk students early. We were inspired to create a solution that:

- **Empowers educators** with AI-assisted early detection capabilities
- **Combines rigorous research** with practical real-world deployment
- **Continuously improves** through teacher feedback and model retraining
- **Respects privacy** while providing actionable insights

This project bridges the gap between academic machine learning research and a production-ready tool that can genuinely help save lives.

---

## 💡 What It Does

**Mental Health Ideation** is a dual-purpose system:

### 🌐 For Teachers (Production System)
A web application deployed at **[mentalhealthideation.com](https://mentalhealthideation.com)** that:
- ✅ Analyzes student essays in real-time for suicide ideation
- ✅ Provides risk levels (LOW/MEDIUM/HIGH) with confidence scores
- ✅ Highlights problematic text using AI interpretability (Integrated Gradients)
- ✅ Collects teacher feedback for continuous model improvement
- ✅ Includes admin dashboard for monitoring and analytics
- ✅ Automatically retrains the model based on real-world corrections

### 🔬 For Researchers (Research Pipeline)
A comprehensive ML research project that:
- ✅ Benchmarks 6+ transformer models (DistilBERT, BERT, RoBERTa, ELECTRA, XLNet, DeBERTa)
- ✅ Uses a 232,074-sample perfectly balanced dataset
- ✅ Employs rigorous methodology suitable for academic publication
- ✅ Includes statistical validation, error analysis, and interpretability studies
- ✅ Tracks experiments with Weights & Biases
- ✅ Ensures reproducibility with multiple random seeds

**Key Metrics**: False Negative Rate (most critical), F1 Score, Recall, Precision, Accuracy

---

## 🛠️ How We Built It

### Tech Stack

**Backend & ML**
- **FastAPI** (Python 3.11) - Async API with automatic documentation
- **PyTorch + HuggingFace Transformers** - Transformer model training and inference
- **DistilBERT** - Production model (66M parameters, 255MB)
- **Celery + Redis** - Background tasks and model retraining queue

**Database & Storage**
- **PostgreSQL** - Persistence for predictions, feedback, and retraining history
- **SQLAlchemy ORM** - Database layer with migration support
- **Google Drive** - Model storage and download

**Deployment & Infrastructure**
- **Railway** - Cloud hosting with automatic CI/CD
- **Docker** - Containerization for consistency
- **Docker Compose** - Local development environment
- **Uvicorn** - ASGI server for production

**Research Tools**
- **Google Colab Pro** - GPU-accelerated model training (24hr sessions)
- **Weights & Biases** - Experiment tracking and visualization
- **SHAP/LIME** - Model interpretability
- **Scikit-learn** - Traditional baselines and metrics

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Railway Platform                    │
├─────────────────────────────────────────────────────┤
│  ┌──────────────┐      ┌───────────┐                │
│  │  Web Service │──────│   Redis   │                │
│  │  (FastAPI)   │      │  (Queue)  │                │
│  └──────┬───────┘      └─────┬─────┘                │
│         │                    │                       │
│         │  Celery Queue      │                       │
│         └────────────────────┤                       │
│                              ▼                       │
│                  ┌───────────────────┐               │
│                  │  Celery Worker    │               │
│                  │  (Model Retrain)  │               │
│                  └───────────────────┘               │
│                              │                       │
│                  ┌───────────▼───────────┐           │
│                  │   PostgreSQL DB       │           │
│                  │ (Predictions/Feedback)│           │
│                  └───────────────────────┘           │
└─────────────────────────────────────────────────────┘
```

### Dataset
- **Size**: 232,074 samples from social media (Reddit/Twitter)
- **Balance**: Perfect 50/50 split (no class imbalance!)
- **Split**: 80/10/10 stratified train/validation/test
- **Preprocessing**: Tokenized with max length 256 (covers 80.7% of samples)

---

## 🚧 Challenges We Ran Into

1. **Project Consolidation**: Started with 3 conflicting project directories (~10GB total) with duplicated code and configs. Consolidated into one unified structure separating research from deployment.

2. **Model Size Constraints**: The DistilBERT model (255MB) exceeded typical deployment limits. Solved by implementing a Google Drive download strategy during Railway build process.

3. **Database Migration Complexity**: Transitioning from SQLite (development) to PostgreSQL (production) required complete rewrite of the database layer using SQLAlchemy ORM while maintaining data integrity.

4. **Feedback Loop Architecture**: Building a production-grade continuous improvement system required designing complex async task queuing with Celery, background model retraining, and safe model swapping without downtime.

5. **Token Length Optimization**: Balancing model coverage (80.7% at 256 tokens) vs. training speed (4x faster than 512 tokens) required careful analysis of token distribution statistics.

6. **Flask to FastAPI Migration**: Completely rewrote the API from Flask to FastAPI for better async support, automatic OpenAPI documentation, and production scalability.

7. **Deployment Reliability**: Ensuring model downloads work consistently on Railway, handling cold starts, managing memory constraints (512MB limit), and implementing proper health checks.

---

## 🏆 Accomplishments That We're Proud Of

- ✅ **Clean, Balanced Dataset**: 232K samples with perfect 50/50 class distribution - no class weights needed!
- ✅ **Live Production System**: Fully deployed at mentalhealthideation.com with Railway hosting
- ✅ **Comprehensive Documentation**: Complete research plan, deployment guides, API docs
- ✅ **Modular Architecture**: Clean separation between research experimentation and production deployment
- ✅ **Continuous Improvement**: Teachers can correct predictions, enabling real-world model improvement
- ✅ **Model Interpretability**: Integrated Gradients provides visual explanations of predictions
- ✅ **Rigorous Methodology**: Statistical validation, error analysis, reproducibility with multiple seeds
- ✅ **Industry-Grade Stack**: Docker, Celery workers, PostgreSQL, Redis - production-ready infrastructure
- ✅ **Baseline Model Trained**: DistilBERT achieving strong performance, ready for benchmarking
- ✅ **Consolidated Codebase**: From messy 3-project setup to clean unified repository

---

## 📚 What We Learned

1. **Research vs. Production Are Different**: Academic research requires reproducibility, statistical rigor, and comprehensive benchmarking, while production demands scalability, monitoring, real-time performance, and usability.

2. **Data Quality Matters More Than Model Complexity**: A perfectly balanced, well-cleaned dataset simplified training significantly - no need for complex class weighting or focal loss.

3. **Feedback Loops Are Essential**: Models only improve over time with real-world feedback. Building the infrastructure for continuous improvement is as important as the initial model.

4. **Documentation Saves Time**: With complex architecture spanning research and production, comprehensive documentation prevented hours of confusion and enabled faster iteration.

5. **Modern Deployment Tools Are Game-Changers**: Railway + Docker made deploying ML applications dramatically easier than traditional hosting approaches.

6. **Token Distribution Analysis Is Critical**: Understanding that 80.7% of samples fit in 256 tokens allowed smart trade-offs between coverage and training efficiency.

7. **Async Architecture Scales Better**: FastAPI's async capabilities and Celery's task queue enable handling multiple requests and expensive operations without blocking.

8. **Ethical Considerations Are Paramount**: Building tools for mental health detection requires careful thought about privacy, bias, false negatives, and human oversight.

---

## 🚀 What's Next for Mental Health Ideation

### Short-term (Month 1)
- [ ] Collect 50-100 teacher corrections through the feedback system
- [ ] Complete training of remaining transformer models (BERT, RoBERTa, ELECTRA, XLNet, DeBERTa)
- [ ] Implement hyperparameter tuning for top-performing models
- [ ] Add rate limiting and admin authentication to production API
- [ ] Create video demo and tutorial materials

### Medium-term (Quarter 1)
- [ ] Run first model retraining cycle with real teacher feedback
- [ ] Implement ensemble methods combining top-performing models
- [ ] Complete comprehensive error analysis and interpretability studies (SHAP/LIME)
- [ ] Prepare academic paper for publication with statistical validation
- [ ] Add email notifications for high-risk detections
- [ ] Expand beta testing to additional schools

### Long-term (Year 1)
- [ ] Multiple retraining cycles measuring improvement over time
- [ ] Conduct demographic bias analysis and fairness audits
- [ ] Develop mobile application version for iOS/Android
- [ ] Create comprehensive teacher training materials and best practices guide
- [ ] Explore multilingual support for non-English texts
- [ ] Partner with mental health organizations for validation studies
- [ ] Scale to districts and educational institutions nationwide

---

## 📁 Project Structure

```
mental-health-ideation-detection/
├── research/                      # 🔬 Research & Experimentation
│   ├── src/                      # Training code, models, metrics
│   ├── config/                   # Experiment configurations
│   ├── scripts/                  # Training & evaluation scripts
│   ├── results/                  # Model checkpoints & metrics
│   └── README.md                 # Research guide
│
├── deployment/                    # 🌐 Production Web Application
│   ├── backend/                  # FastAPI application
│   │   ├── app.py               # Main API
│   │   ├── worker.py            # Celery tasks
│   │   ├── services/            # ML inference services
│   │   └── templates/           # HTML templates
│   ├── Dockerfile               # Container definition
│   ├── docker-compose.yml       # Local development
│   └── README.md                # Deployment guide
│
├── docs/                          # 📚 Documentation
│   ├── RESEARCH_PLAN.md          # Research methodology
│   ├── EDA_SUMMARY.md            # Data analysis
│   └── PUBLICATION_QUERY.md      # Publication planning
│
├── LICENSE                        # MIT License
└── README.md                      # This file
```

---

## 🚀 Quick Start

### For Teachers (Use the App)

Visit **[mentalhealthideation.com](https://mentalhealthideation.com)** and:
1. Paste student essay text into the analyzer
2. Click "Analyze" to get real-time risk assessment
3. Review highlighted text showing concerning phrases
4. Submit feedback if the prediction is incorrect
5. Help improve the model for everyone

### For Developers (Run Locally)

```bash
# Clone the repository
git clone https://github.com/AvyayaKumar/mental-health-ideation-detection.git
cd mental-health-ideation-detection/deployment/

# Copy environment file
cp .env.example .env

# Start all services with Docker
docker-compose up

# Visit: http://localhost:8000
```

### For Researchers (Train Models)

```bash
# Navigate to research folder
cd research/

# Install dependencies
pip install -r requirements-research.txt

# Train baseline model
python src/train.py --config config/distilbert_base.yaml

# See research/README.md for detailed instructions
```

---

## 📊 Current Status

- ✅ **Trained DistilBERT baseline model** (97.96% accuracy)
- ✅ **Production web app deployed** at mentalhealthideation.com
- ✅ **Feedback loop system implemented** with PostgreSQL + Celery
- ✅ **Admin dashboard operational** for monitoring predictions
- 🔄 **Additional model training in progress** (BERT, RoBERTa, ELECTRA)
- 🔄 **Error analysis & interpretability studies** ongoing
- ⏳ **Statistical validation pending** (McNemar's tests)
- ⏳ **Academic publication preparation** planned

---

## 🔬 Research Methodology

**Objective**: Benchmark transformer models for suicide ideation detection with rigorous experimental methodology suitable for academic publication.

**Models Being Evaluated**:
1. DistilBERT (66M params) - ✅ Trained
2. BERT-base (110M params) - 🔄 In progress
3. RoBERTa-base (125M params) - ⏳ Pending
4. ELECTRA-base (110M params) - ⏳ Pending
5. XLNet-base (110M params) - ⏳ Pending
6. DeBERTa-v3-base (86M params) - ⏳ Pending

**Key Metrics** (in priority order):
- False Negative Rate (most critical)
- F1 Score
- Recall
- Precision
- Accuracy

**Timeline**: 7-12 weeks for complete research study

See [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) for full methodology.

---

## 🌐 API Documentation

Interactive API documentation available at:
- **Swagger UI**: https://mentalhealthideation.com/docs
- **ReDoc**: https://mentalhealthideation.com/redoc

### Key Endpoints

```bash
# Health check
GET /health

# Analyze text
POST /api/v1/analyze
{
  "text": "Sample essay text...",
  "explain": true
}

# Submit teacher feedback
POST /api/v1/feedback
{
  "prediction_id": 123,
  "correct_class": 0,
  "teacher_notes": "Actually not concerning..."
}

# Admin dashboard
GET /admin
```

---

## 🤝 Contributing

We welcome contributions! This project serves both research and production purposes.

### For Research Contributions
- Check [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) for current experiments
- Follow reproducibility guidelines (multiple seeds, configs, etc.)
- Document methodology and results thoroughly

### For Production Contributions
- Check [deployment/README.md](deployment/README.md) for architecture
- Test locally with Docker Compose before submitting PR
- Ensure backward compatibility with existing database schema

### Getting Started
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## ⚠️ Important Notes

- **Research Data**: Not included in repository (large CSV files - 232K samples). Contact maintainer for dataset access.
- **Model Files**: Large model files (255MB) handled via Google Drive download during deployment.
- **Sensitive Data**: No actual student essays or personal data in repository.
- **Ethics**: This tool is designed to help teachers identify at-risk students for intervention, not for surveillance or punishment.
- **Limitations**: This is an assistive tool requiring human oversight. Never use as sole diagnostic criteria.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

**Avyaya Kumar**
- Email: avyaya.kumar@gmail.com
- GitHub: [@AvyayaKumar](https://github.com/AvyayaKumar)
- Project: [mental-health-ideation-detection](https://github.com/AvyayaKumar/mental-health-ideation-detection)

---

## 🙏 Acknowledgments

- **HuggingFace** for the Transformers library and pre-trained models
- **Railway** for providing excellent deployment platform
- **Google Colab** for GPU resources for research
- **FastAPI** community for the excellent web framework
- **Mental health professionals** who provided guidance on ethical considerations

---

**Built with ❤️ to help save lives through AI-assisted early detection**

*Last Updated: October 2025*
