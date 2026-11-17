# Telco Customer Churn Prediction System

An end-to-end Machine Learning application that predicts customer churn, analyzes customer behavior, and explains individual predictions using SHAP.


## Live Demo

🚀 [Click here to explore the deployed application](https://telco-customer-churn-prediction-as76.onrender.com/)

## Overview

Customer churn is a major challenge for telecommunications companies. Losing existing customers can affect revenue and long-term business growth.

This project aims to identify customers who are likely to discontinue their services. It combines machine learning, data analysis, model explainability, and an interactive web application to make churn predictions more accessible and interpretable.

## Key Features

- **Customer Churn Prediction:** Predicts whether a customer is likely to leave the service.
- **Churn Probability:** Displays the estimated probability of customer churn.
- **Risk Classification:** Categorizes customers into Low, Medium, and High risk groups.
- **Interactive Analytics Dashboard:** Presents customer and business insights through visualizations.
- **Model Explainability:** Uses SHAP to explain the factors influencing individual predictions.
- **Model Evaluation:** Displays accuracy, precision, recall, F1-score, and a confusion matrix.

## Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib, Chart.js |
| Machine Learning | Scikit-learn |
| Explainable AI | SHAP |
| Backend | Flask, Flask-CORS |
| Frontend | HTML, CSS, JavaScript |
| Model Persistence | Joblib |

## Machine Learning Workflow

1. **Data Collection:** Used a telecommunications customer churn dataset.
2. **Exploratory Data Analysis:** Examined customer demographics, subscribed services, billing information, and churn patterns.
3. **Data Preprocessing:** Handled missing values, converted data types, removed the customer identifier, and encoded categorical variables.
4. **Feature Scaling:** Applied StandardScaler to numerical features.
5. **Model Training:** Compared multiple classification algorithms.
6. **Model Selection:** Selected Logistic Regression after comparing model performance.
7. **Threshold Optimization:** Evaluated different classification thresholds and selected 0.66 to balance churn detection and other evaluation metrics.
8. **Explainability:** Integrated SHAP to understand individual model predictions.
9. **Deployment:** Built a Flask-based application with an interactive frontend.

## Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The classification threshold is set to **0.66**.

The threshold was selected to improve churn detection while considering the trade-off between precision and recall.

The project prioritizes identifying potential churners while recognizing that false positives and false negatives have different business consequences.

## Project Structure

```text
FinalProject/
│
├── backend/
│   ├── app.py
│   ├── analytics.py
│   ├── explainability.py
│   └── evaluation.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── charts.js
│
├── model/
│   ├── churn_model.pkl
│   ├── feature_columns.pkl
│   ├── numerical_columns.pkl
│   └── scaler.pkl
│
├── data/
│   └── archive.csv
│
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Aniruddhapal-01/telco-customer-churn-prediction.git
```

### 2. Navigate to the project directory

```bash
cd telco-customer-churn-prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python backend/app.py
```

Open the local application in your browser at:

```text
http://127.0.0.1:5000
```

## Project Objectives

This project demonstrates practical implementation of:

- End-to-end machine learning workflows.
- Classification and model comparison.
- Data preprocessing and feature engineering.
- Model evaluation and threshold selection.
- Explainable Artificial Intelligence.
- Flask-based application development.
- Integration of machine learning models with a web interface.

## Future Improvements

- Deploy the application on a cloud platform.
- Improve model performance through ensemble learning.
- Introduce customer segmentation.
- Add more detailed business recommendations.
- Explore model monitoring and prediction drift.

## Limitations

- Predictions are based on the available historical dataset.
- Model performance may vary on real-world customer data.
- SHAP explanations describe model behavior and do not establish causal relationships.
- Risk categories are application-defined classifications, not guaranteed probabilities of actual customer departure.

## Author

**Aniruddha Pal**

B.Tech Computer Science and Engineering

[GitHub](https://github.com/AniruddhaPal-01)

---

*This project was developed as a practical application of Machine Learning, Data Analysis, Explainable AI, and Python-based web development.*
