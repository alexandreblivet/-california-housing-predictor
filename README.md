![CI/CD Pipeline](https://github.com/alexandreblivet/-california-housing-predictor/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-red)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange)
![Docker](https://img.shields.io/badge/docker-ready-blue)

# California Housing Price Predictor

A complete machine learning project that predicts California housing prices using the classic California housing dataset. Features a Streamlit web application and full Docker containerization.

## Project Notes

This project uses the California Housing dataset from scikit-learn - a standard dataset that's commonly used for regression tutorials and examples. I chose it because it's reliable and well-documented, which meant I could focus on the containerization and web app parts rather than dealing with messy data.

The main goal here was getting comfortable with Docker and Streamlit, not building the most accurate housing price model. I simplified the ML pipeline to just use Linear and Ridge regression instead of more complex models like Random Forest or XGBoost. This cuts the Docker build time way down (from over 30 minutes to about 6 minutes) and still demonstrates all the core concepts.

Basically, this is more about showing I can build a complete ML application that actually works, with some testing, containerization, and CI/CD, using a straightforward dataset.

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


