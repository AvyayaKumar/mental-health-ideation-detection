# Contributing to Mental Health Ideation Detection

Thank you for your interest in contributing to Mental Health Ideation Detection! This project aims to help save lives through AI-assisted early detection of suicide ideation in text.

## 🌟 Ways to Contribute

We welcome contributions in several areas:

### 1. Research Contributions
- Train and benchmark additional transformer models
- Improve model performance and reduce false negative rates
- Conduct error analysis and interpretability studies
- Contribute to academic publication preparation

### 2. Production/Engineering Contributions
- Improve the web application interface and user experience
- Optimize inference performance and reduce latency
- Enhance the feedback loop and retraining pipeline
- Add new features (rate limiting, authentication, notifications)
- Improve documentation and deployment guides

### 3. Documentation Contributions
- Improve code documentation and docstrings
- Create tutorials and educational materials
- Write blog posts or case studies
- Translate documentation to other languages

### 4. Testing and Bug Reports
- Report bugs and issues
- Test the application with real-world scenarios
- Improve test coverage
- Conduct security audits

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Docker Desktop (for local deployment testing)
- Git
- (Optional) Google Colab account for research/training

### Fork and Clone

```bash
# Fork the repository on GitHub first, then:
git clone https://github.com/YOUR_USERNAME/mental-health-ideation-detection.git
cd mental-health-ideation-detection

# Add upstream remote
git remote add upstream https://github.com/AvyayaKumar/mental-health-ideation-detection.git
```

### Set Up Development Environment

#### For Research Work

```bash
cd research/

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-research.txt

# Set up experiment tracking (optional)
wandb login
```

#### For Production/Engineering Work

```bash
cd deployment/

# Copy environment template
cp .env.example .env
# Edit .env with your local settings

# Start services with Docker
docker-compose up -d

# Or run locally:
pip install -r backend/requirements.txt
redis-server  # In separate terminal
cd backend && uvicorn app:app --reload
```

### Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
# or
git checkout -b docs/documentation-improvement
```

## 📋 Contribution Guidelines

### Code Style

We follow PEP 8 for Python code with some specific conventions:

```python
# Use descriptive variable names
model_predictions = model.predict(text)  # Good
preds = model.predict(text)              # Avoid

# Document functions with docstrings
def analyze_text(text: str, explain: bool = True) -> Dict:
    """
    Analyze text for suicide ideation.

    Args:
        text: Input text to analyze
        explain: Whether to include interpretability explanations

    Returns:
        Dictionary containing predictions and explanations
    """
    pass

# Use type hints
def train_model(config: Dict[str, Any], seed: int = 42) -> nn.Module:
    pass
```

### Research Contributions

When contributing research code:

1. **Reproducibility is critical**:
   - Use configuration files for all hyperparameters
   - Set all random seeds (Python, NumPy, PyTorch)
   - Document exact library versions used
   - Save experiment configs with results

2. **Follow the project structure**:
   ```
   research/
   ├── config/
   │   └── your_model.yaml      # Model configuration
   ├── src/
   │   └── your_module.py       # New model/metric code
   ├── scripts/
   │   └── train_your_model.py  # Training script
   └── results/
       └── your_model-seed42/   # Results and checkpoints
   ```

3. **Use experiment tracking**:
   - Log all runs to Weights & Biases
   - Include model name, hyperparameters, metrics
   - Save confusion matrices and error analysis

4. **Run multiple seeds**:
   - Always test with at least 3 random seeds (42, 123, 456)
   - Report mean ± standard deviation for all metrics

5. **Document findings**:
   - Create Jupyter notebooks for analysis
   - Write clear markdown summaries
   - Compare with baseline models

### Production Contributions

When contributing to the web application:

1. **Test locally first**:
   ```bash
   cd deployment/
   docker-compose up
   # Test your changes at http://localhost:8000
   ```

2. **Maintain backward compatibility**:
   - Don't break existing API endpoints
   - Use database migrations for schema changes
   - Version new API endpoints (e.g., /api/v2/)

3. **Add tests**:
   ```python
   # In deployment/backend/tests/
   def test_prediction_endpoint():
       response = client.post("/api/v1/analyze", json={
           "text": "Sample text",
           "explain": True
       })
       assert response.status_code == 200
       assert "prediction" in response.json()
   ```

4. **Update documentation**:
   - Update API documentation in README
   - Add docstrings to new functions
   - Update deployment guides if needed

### Commit Messages

Use clear, descriptive commit messages following conventional commits:

```bash
# Format: <type>(<scope>): <description>

# Examples:
git commit -m "feat(api): Add rate limiting to prediction endpoint"
git commit -m "fix(model): Fix tokenization bug for long texts"
git commit -m "docs(readme): Update installation instructions"
git commit -m "test(backend): Add tests for feedback submission"
git commit -m "refactor(worker): Improve Celery task error handling"
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

### Pull Request Process

1. **Update your fork**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Ensure tests pass**:
   ```bash
   # For research code:
   python -m pytest research/tests/

   # For production code:
   cd deployment/backend/
   pytest tests/
   ```

3. **Create pull request**:
   - Write a clear title and description
   - Reference any related issues (#123)
   - Include screenshots for UI changes
   - List what you've tested

4. **Pull Request Template**:
   ```markdown
   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update

   ## Testing
   - [ ] Tested locally
   - [ ] Added/updated tests
   - [ ] Passes existing tests

   ## Checklist
   - [ ] Code follows project style guidelines
   - [ ] Self-reviewed my code
   - [ ] Commented complex sections
   - [ ] Updated documentation
   - [ ] No new warnings

   ## Screenshots (if applicable)
   ```

5. **Code review**:
   - Address reviewer feedback promptly
   - Push updates to the same branch
   - Be open to suggestions and improvements

## 🐛 Reporting Bugs

### Before Submitting a Bug Report

1. Check existing issues to avoid duplicates
2. Test with the latest version
3. Collect relevant information:
   - Operating system and version
   - Python version
   - Error messages and stack traces
   - Steps to reproduce

### Bug Report Template

```markdown
**Describe the bug**
Clear description of what the bug is

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '...'
3. See error

**Expected behavior**
What you expected to happen

**Screenshots**
If applicable, add screenshots

**Environment:**
 - OS: [e.g., macOS 13.0]
 - Python: [e.g., 3.11.2]
 - Browser: [e.g., Chrome 108]

**Additional context**
Any other relevant information
```

## 💡 Suggesting Features

We love feature suggestions! To suggest a new feature:

1. Check existing feature requests to avoid duplicates
2. Create a new issue with the "enhancement" label
3. Describe:
   - What problem does it solve?
   - Who would benefit?
   - Proposed solution or implementation ideas
   - Alternatives you've considered

## 🧪 Testing

### Running Tests

```bash
# Research tests
cd research/
pytest tests/ -v

# Production tests
cd deployment/backend/
pytest tests/ -v --cov=. --cov-report=html

# Integration tests
cd deployment/
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

### Writing Tests

```python
# Test file: test_model_inference.py
import pytest
from services.predictor import SuicideIdeationPredictor

@pytest.fixture
def predictor():
    return SuicideIdeationPredictor(model_path="path/to/model")

def test_prediction_format(predictor):
    """Test that predictions have correct format."""
    result = predictor.predict("Sample text")
    assert "class" in result
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 1

def test_empty_text_handling(predictor):
    """Test handling of empty input."""
    with pytest.raises(ValueError):
        predictor.predict("")
```

## 🔐 Security

If you discover a security vulnerability:

1. **DO NOT** create a public issue
2. Email: avyaya.kumar@gmail.com with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)
3. We'll respond within 48 hours
4. We'll work with you to fix and disclose responsibly

## ⚖️ Ethics and Responsible AI

This project deals with sensitive mental health data. When contributing:

### Ethical Guidelines

1. **Privacy First**:
   - Never commit real user data
   - Use synthetic/anonymized data for testing
   - Respect data protection regulations (GDPR, HIPAA)

2. **Bias and Fairness**:
   - Test for demographic biases
   - Document limitations clearly
   - Don't overstate model capabilities

3. **Safety**:
   - Prioritize minimizing false negatives
   - Include clear disclaimers about model limitations
   - Ensure human oversight in all decisions

4. **Transparency**:
   - Document model decisions and interpretability
   - Be clear about uncertainty and confidence levels
   - Provide explanations for predictions

5. **Appropriate Use**:
   - This tool is for early detection and intervention
   - NOT for surveillance or punishment
   - NOT a replacement for professional mental health evaluation
   - Requires human judgment and follow-up

### Required Considerations

Before contributing features that:
- Change prediction behavior
- Add new data collection
- Modify feedback mechanisms
- Alter model training

Please consider:
- Could this harm vulnerable individuals?
- Does this respect user privacy?
- Is this transparent and explainable?
- Does this reinforce or reduce biases?

## 📚 Resources

### Documentation
- [Research Plan](docs/RESEARCH_PLAN.md)
- [Deployment Guide](deployment/README.md)
- [API Documentation](https://mentalhealthideation.com/docs)

### Learning Resources
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Railway Docs](https://docs.railway.app)

### Community
- GitHub Discussions: For questions and discussions
- Issues: For bug reports and feature requests
- Email: avyaya.kumar@gmail.com for direct contact

## 🙏 Recognition

Contributors will be:
- Listed in project acknowledgments
- Mentioned in release notes
- Credited in any academic publications (for significant research contributions)

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for helping make mental health support more accessible through technology! Every contribution, no matter how small, makes a difference.**

Questions? Feel free to reach out:
- Email: avyaya.kumar@gmail.com
- GitHub: [@AvyayaKumar](https://github.com/AvyayaKumar)

*Last Updated: October 2025*
