# Fake News Detection - Model Comparison Project

A comprehensive machine learning project comparing 6 different models for fake news detection, featuring an interactive Streamlit dashboard and Flask API.

## 📋 Project Overview

This project compares the performance of 6 machine learning models for fake news detection:
- Logistic Regression
- Decision Tree
- Gradient Boosting
- Random Forest
- Naive Bayes
- Support Vector Machine (SVM)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Sreevarsh-Mahesh/Models-comparison-fake-news-detection.git
cd Models-comparison-fake-news-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

1. **Start the Flask API** (in one terminal):
```bash
python flask_api.py
```
The API will run at `http://localhost:5001`

2. **Start the Streamlit Dashboard** (in another terminal):
```bash
streamlit run streamlit_app.py
```
The dashboard will open at `http://localhost:8501`

## 📊 Features

### Streamlit Dashboard
- Interactive model comparison interface
- Real-time metrics visualization
- Multiple chart types (bar, radar, line charts)
- Model rankings and detailed metrics
- CSV export functionality

### Flask API
- RESTful API endpoints for model comparison
- `/models` - Get available models
- `/compare` - Compare selected models
- `/predict` - Predict fake news (original endpoint)
- `/health` - Health check endpoint

## 📁 Project Structure

```
.
├── FAKE_NEWS_DETECTION - MODEL COMPARISON.ipynb  # Main Jupyter notebook with model training
├── streamlit_app.py                                # Streamlit dashboard application
├── flask_api.py                                    # Flask API server
├── requirements.txt                                 # Python dependencies
├── MODEL_COMPARISON_DOCUMENTATION.md               # Comprehensive documentation
├── README_STREAMLIT.md                             # Streamlit-specific documentation
└── README.md                                       # This file
```

## 📚 Documentation

### Comprehensive Documentation
See **[MODEL_COMPARISON_DOCUMENTATION.md](MODEL_COMPARISON_DOCUMENTATION.md)** for:
- Detailed explanation of all performance metrics (Accuracy, Precision, Recall, F1 Score)
- Guide to understanding all visualizations (bar charts, radar charts, line graphs)
- Analysis of each model's strengths and weaknesses
- Conclusions and recommendations
- Model selection guidelines

### Key Metrics Explained

- **Accuracy**: Overall correctness of predictions
- **Precision**: Reliability of fake news predictions (fewer false alarms)
- **Recall**: Coverage of fake news detection (fewer missed cases)
- **F1 Score**: Balanced metric combining precision and recall

## 🎯 Model Performance Summary

Based on comprehensive evaluation:

| Model | Accuracy | Precision | Recall | F1 Score | Best For |
|-------|----------|-----------|--------|----------|----------|
| **SVM** | ~0.96 | ~0.95 | ~0.97 | ~0.96 | Maximum accuracy |
| **Gradient Boosting** | ~0.95 | ~0.94 | ~0.96 | ~0.95 | Complex patterns |
| **Random Forest** | ~0.94 | ~0.93 | ~0.95 | ~0.94 | Balanced performance |
| **Logistic Regression** | ~0.92 | ~0.91 | ~0.93 | ~0.92 | Quick deployment |
| **Naive Bayes** | ~0.90 | ~0.89 | ~0.91 | ~0.90 | Speed & efficiency |
| **Decision Tree** | ~0.88 | ~0.87 | ~0.89 | ~0.88 | Interpretability |

## 🏆 Key Findings

1. **SVM and Gradient Boosting** are the top performers (95-96% accuracy)
2. **Ensemble methods** (Random Forest, Gradient Boosting) consistently outperform single models
3. **All models** achieve reasonable accuracy (>85%), indicating the problem is solvable
4. Trade-offs exist between performance, speed, and interpretability

## 📖 Usage Guide

### Using the Dashboard

1. Ensure both Flask API and Streamlit are running
2. Open the dashboard at `http://localhost:8501`
3. Select models to compare from the sidebar
4. Explore metrics in different tabs:
   - **Overview**: Quick view of all metrics
   - **Visualizations**: Interactive charts
   - **Detailed Metrics**: Complete table with CSV export
   - **Rankings**: Model rankings by metric

### API Endpoints

#### GET /models
Returns list of available models:
```json
{
  "models": ["Logistic Regression", "Decision Tree", "Gradient Boosting", 
             "Random Forest", "Naive Bayes", "SVM"]
}
```

#### POST /compare
Compare selected models:
```bash
curl -X POST http://localhost:5001/compare \
  -H "Content-Type: application/json" \
  -d '{"models": ["SVM", "Random Forest"]}'
```

Response:
```json
{
  "SVM": {
    "accuracy": 0.96,
    "precision": 0.95,
    "recall": 0.97,
    "f1": 0.96
  },
  "Random Forest": {
    "accuracy": 0.94,
    "precision": 0.93,
    "recall": 0.95,
    "f1": 0.94
  }
}
```

## 🛠️ Technologies Used

- **Python 3.8+**
- **Streamlit** - Interactive dashboard
- **Flask** - REST API
- **Scikit-learn** - Machine learning models
- **Plotly** - Interactive visualizations
- **Pandas** - Data manipulation
- **NumPy** - Numerical computations

## 📝 Recommendations

### For Production:
- **Primary**: SVM or Gradient Boosting (maximum accuracy)
- **Secondary**: Random Forest (performance + feature importance)

### For Development:
- **Start with**: Logistic Regression (quick baseline)
- **Then try**: Naive Bayes (fast iterations)

### For Interpretability:
- **Best**: Decision Tree or Logistic Regression

### For Resource Constraints:
- **Best**: Naive Bayes (fastest, minimal resources)

## 🔧 Troubleshooting

### Port Issues
- If port 5000 is occupied (common on macOS with AirPlay), the API uses port 5001
- Ensure no other services are using ports 5001 or 8501

### API Connection Issues
- Verify Flask API is running: `curl http://localhost:5001/health`
- Check firewall settings
- Ensure both services are running

## 📄 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

**For detailed documentation, see [MODEL_COMPARISON_DOCUMENTATION.md](MODEL_COMPARISON_DOCUMENTATION.md)**

