# 🏦 Advanced Loan Default Prediction System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.3.0-green.svg)](https://flask.palletsprojects.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45.1-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

**SBA-LoanPredict-AI** is a cutting-edge machine learning system designed for predicting Small Business Administration (SBA) loan default risk. Built with enterprise-grade architecture and regulatory compliance in mind, this system provides comprehensive loan risk assessment, business intelligence, and profit optimization capabilities.

### 🎯 Key Features

- **🤖 Advanced ML Pipeline**: 8 sophisticated algorithms with cost-sensitive learning
- **💰 Business Optimization**: Profit maximization with 5:1 cost ratio compliance
- **📊 Interactive Dashboard**: Modern Streamlit interface with real-time analytics
- **🔗 RESTful API**: Flask-based backend with comprehensive endpoints
- **📈 Business Intelligence**: Economic indicators, gains/lift charts, SHAP explainability
- **⚖️ Regulatory Compliance**: Basel III guidelines and model interpretability

## 🏆 Performance Metrics

| Model | Accuracy | AUC | Precision | Recall | F1-Score |
|-------|----------|-----|-----------|--------|----------|
| **Bagging** ⭐ | **93.5%** | **96.2%** | **82.2%** | **80.1%** | **81.1%** |
| Random Forest | 93.1% | 95.9% | 84.2% | 74.9% | 79.3% |
| Neural Network | 92.3% | 95.6% | 81.8% | 72.2% | 76.7% |
| Decision Tree | 90.8% | 84.2% | 73.7% | 73.9% | 73.8% |
| AdaBoost | 90.7% | 94.2% | 77.6% | 66.4% | 71.5% |

**💸 Business Impact**: $1.33 billion expected profit with 76.5% approval rate

## 🗃️ Dataset

- **Source**: SBA National Dataset (1987-2014)
- **Size**: 899,164 loan observations
- **Features**: 27 variables processed to 12 key predictors
- **Default Rate**: 17.52%
- **Total Portfolio**: $100+ billion in loan amounts

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.11+
pip (package installer)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/pranavrevee/SBA-LoanPredict-AI.git
cd SBA-LoanPredict-AI
```

2. **Install dependencies**
```bash
pip install -r requirements_basic.txt
```

3. **Train models** (optional - pre-trained models included)
```bash
cd backend
python train_model.py
```

4. **Start the API server**
```bash
cd backend
python flask_app.py
```

5. **Launch the dashboard**
```bash
streamlit run streamlit_advanced/business_dashboard.py --server.port 8501
```

### 🌐 Access Points

- **Dashboard**: http://localhost:8501
- **API**: http://127.0.0.1:5001
- **API Documentation**: http://127.0.0.1:5001/api/features

## 📁 Project Structure

```
SBA-LoanPredict-AI/
├── 📊 data/
│   └── sba_national_data.csv          # SBA national dataset
├── 🌐 backend/
│   ├── flask_app.py                   # Main API server
│   ├── train_model.py                 # Model training pipeline
│   ├── trained_model.pkl              # Best model (523MB)
│   ├── model_summary.json             # Performance metrics
│   └── features.pkl                   # Feature specifications
├── 🎨 streamlit_advanced/
│   └── business_dashboard.py          # Interactive BI dashboard
├── 🧠 src/
│   ├── models/
│   │   ├── advanced_models.py         # Quantum-inspired algorithms
│   │   └── business_optimizer.py      # Profit optimization
│   ├── features/
│   │   └── advanced_engineering.py    # Feature engineering
│   └── evaluation/
│       ├── advanced_metrics.py        # Business metrics
│       └── explainability.py          # SHAP interpretability
├── 📋 requirements_basic.txt          # Core dependencies
├── 📋 requirements_advanced.txt       # Extended ML libraries
├── 🎯 .cursorrules                    # AI coding guidelines
└── 📖 README.md                       # Project documentation
```

## 🔧 API Endpoints

### Core Endpoints
- `GET /` - Health check and system status
- `GET /api/models` - Available models information
- `POST /api/predict` - Single loan prediction
- `POST /api/predict/batch` - Batch loan processing
- `GET /api/analytics` - Model performance metrics
- `GET /api/features` - Required input features

### Example Usage

```python
import requests

# Single prediction
loan_data = {
    "Term": 120,
    "NoEmp": 10,
    "DisbursementGross": 250000,
    "GrAppv": 275000,
    "SBA_Appv": 200000,
    "NewExist": 1,
    "UrbanRural": 1,
    "RevLineCr": 0,
    "LowDoc": 1,
    "NAICS": 561
}

response = requests.post(
    "http://127.0.0.1:5001/api/predict",
    json=loan_data
)
result = response.json()
print(f"Default Risk: {result['prediction']['default_probability']:.2%}")
```

## 📊 Dashboard Features

### 🏠 Overview
- Real-time system metrics
- Performance gauges (AUC, Accuracy, Precision)
- Business impact visualization

### 🎯 Single Prediction
- Interactive loan application form
- Real-time risk assessment
- Business recommendation engine

### 📈 Batch Analysis
- CSV file upload and processing
- Risk distribution charts
- Portfolio profit analysis

### 📊 Model Analytics
- Comprehensive performance metrics
- Business impact calculations
- Model comparison tools

### ⚙️ Business Intelligence
- Economic indicator dashboard
- Threshold optimization tools
- AI-powered recommendations

## 🤖 Machine Learning Pipeline

### Algorithms Implemented
1. **Bagging Classifier** ⭐ (Best performing)
2. **Random Forest**
3. **Neural Network (MLP)**
4. **Decision Tree**
5. **AdaBoost**
6. **k-Nearest Neighbors**
7. **Logistic Regression**
8. **Linear Discriminant Analysis**

### Advanced Features
- **Cost-Sensitive Learning**: 5:1 misclassification ratio
- **SHAP Explainability**: Model interpretation for regulatory compliance
- **Threshold Optimization**: Profit-maximizing decision boundaries
- **Cross-Validation**: Robust model evaluation
- **Feature Engineering**: Business-relevant transformations

## 💼 Business Applications

### Financial Institutions
- **Risk Assessment**: Automated loan approval decisions
- **Portfolio Management**: Risk-based loan pricing
- **Regulatory Compliance**: Basel III capital requirements

### Government Agencies
- **Policy Analysis**: SBA program effectiveness
- **Economic Impact**: Job creation assessment
- **Risk Management**: Default rate monitoring

### Fintech Companies
- **Credit Scoring**: Alternative lending decisions
- **Product Development**: Risk-based loan products
- **Market Analysis**: Industry trend identification

## 🔬 Technical Specifications

### Performance
- **Training Time**: ~20 minutes on standard hardware
- **Prediction Speed**: <100ms per loan
- **Memory Usage**: ~2GB during training
- **Model Size**: 523MB (production-ready)

### Scalability
- **Concurrent Users**: 100+ simultaneous dashboard users
- **API Throughput**: 1000+ predictions/minute
- **Batch Processing**: 10,000+ loans per batch

### Security
- **Data Anonymization**: PII protection
- **API Authentication**: Configurable security layers
- **Audit Trails**: Complete prediction logging

## 📚 Documentation

- [API Documentation](docs/api.md)
- [Model Training Guide](docs/training.md)
- [Dashboard User Manual](docs/dashboard.md)
- [Deployment Guide](docs/deployment.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone and setup development environment
git clone https://github.com/pranavrevee/SBA-LoanPredict-AI.git
cd SBA-LoanPredict-AI
pip install -r requirements_advanced.txt

# Run tests
python -m pytest tests/

# Start development servers
python backend/flask_app.py &
streamlit run streamlit_advanced/business_dashboard.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SBA**: For providing the national dataset
- **CSU Analytics Competition**: Project inspiration
- **Open Source Community**: Essential libraries and frameworks

## 📞 Contact

**Pranav Revee**
- GitHub: [@pranavrevee](https://github.com/pranavrevee)
- LinkedIn: [Pranav Revee](https://linkedin.com/in/pranavrevee)
- Email: pranav.revee@example.com

---

⭐ **Star this repo if you find it helpful!** ⭐

*Built with ❤️ for the financial technology community* 
