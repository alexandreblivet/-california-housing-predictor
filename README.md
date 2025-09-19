![CI/CD Pipeline](https://github.com/alexandreblivet/-california-housing-predictor/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-red)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange)
![Docker](https://img.shields.io/badge/docker-ready-blue)

# California Housing Price Predictor

A complete machine learning project that predicts California housing prices using the classic California housing dataset. Features a Streamlit web application and full Docker containerization.


## 🚀 Quick Start

### Run with Docker (Recommended)
```bash
# Clone the repository
git clone https://github.com/alexandreblivet/-california-housing-predictor.git
cd california-housing-predictor

# Build and run with Docker
docker build -t california-housing-predictor .
docker run -p 8501:8501 california-housing-predictor

# Access at http://localhost:8501
```

### Local Development
```bash
# Clone the repository
git clone https://github.com/alexandreblivet/-california-housing-predictor.git
cd california-housing-predictor

# Set up environment
pip install -r requirements.txt
python src/data_processing.py
python src/model_training.py
streamlit run app/streamlit_app.py
```
