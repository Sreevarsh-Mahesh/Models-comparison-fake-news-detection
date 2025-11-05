# Fake News Detection - Model Comparison Dashboard

A professional Streamlit dashboard for comparing multiple fake news detection models using metrics from a Flask API.

## Features

- 📊 **Model Comparison**: Compare multiple ML models side-by-side
- 📈 **Interactive Visualizations**: Bar charts, radar charts, and line graphs
- 📋 **Detailed Metrics**: View accuracy, precision, recall, and F1 scores
- 🏆 **Model Rankings**: See which models perform best for each metric
- 📥 **Export Data**: Download comparison metrics as CSV

## Prerequisites

- Python 3.8+
- Flask API running at `http://localhost:5000` with the following endpoints:
  - `GET /models` - Returns available models
  - `POST /compare` - Returns comparison metrics for selected models

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Ensure your Flask API is running at `http://localhost:5000`

2. Start the Streamlit dashboard:
```bash
streamlit run streamlit_app.py
```

3. The dashboard will open in your browser at `http://localhost:8501`

## API Endpoints Required

### GET /models
Returns list of available models:
```json
{
  "models": ["Naive Bayes", "SVM", "Logistic Regression", "Random Forest", "Gradient Boosting", "Decision Tree"]
}
```

### POST /compare
Request body:
```json
{
  "models": ["SVM", "Random Forest"]
}
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

## Dashboard Features

1. **Sidebar**: Select models to compare
2. **Overview Tab**: Quick view of all metrics for selected models
3. **Visualizations Tab**: Interactive charts and graphs
4. **Detailed Metrics Tab**: Complete metrics table with CSV export
5. **Rankings Tab**: Model rankings by metric and overall performance

## Technologies Used

- Streamlit - Dashboard framework
- Plotly - Interactive visualizations
- Pandas - Data manipulation
- Requests - API communication

