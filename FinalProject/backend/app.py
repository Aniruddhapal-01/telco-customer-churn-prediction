
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import pandas as pd
import joblib
import os

from explainability import create_explainer, explain_customer
from evaluation import evaluate_model
from analytics import build_analytics


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

MODEL_DIR = os.path.join(PROJECT_DIR, "model")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")

DATA_PATH = os.path.join(DATA_DIR, "archive.csv")


# ==========================================
# 2. FLASK CONFIGURATION
# ==========================================

app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path="/frontend"
)

CORS(app)


# ==========================================
# 3. CHECK FILES
# ==========================================

if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

if not os.path.isdir(FRONTEND_DIR):
    raise FileNotFoundError(f"Frontend folder not found: {FRONTEND_DIR}")

print("Dataset:", DATA_PATH)
print("Dataset exists:", os.path.isfile(DATA_PATH))


# ==========================================
# 4. LOAD SAVED MODEL FILES
# ==========================================

model = joblib.load(os.path.join(MODEL_DIR, "churn_model.pkl"))

feature_columns = joblib.load(
    os.path.join(MODEL_DIR, "feature_columns.pkl")
)

numerical_columns = joblib.load(
    os.path.join(MODEL_DIR, "numerical_columns.pkl")
)

scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))


# ==========================================
# 5. LOAD DATASET
# ==========================================

df = pd.read_csv(DATA_PATH)


# ==========================================
# 6. INITIALISE SHAP EXPLAINER
# ==========================================

explainer = create_explainer(
    model,
    df,
    feature_columns,
    numerical_columns,
    scaler
)


# ==========================================
# 7. FRONTEND ROUTE
# ==========================================

@app.route("/", methods=["GET"])
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


# ==========================================
# 8. HEALTH CHECK
# ==========================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "message": "Customer Churn Prediction API is running!"
    })


# ==========================================
# 9. PREDICTION ROUTE
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No customer data received"
            }), 400

        customer_df = pd.DataFrame([data])

        for col in numerical_columns:
            customer_df[col] = pd.to_numeric(
                customer_df[col],
                errors="raise"
            )

        customer_encoded = pd.get_dummies(customer_df)

        customer_encoded = customer_encoded.reindex(
            columns=feature_columns,
            fill_value=0
        )

        customer_encoded[numerical_columns] = scaler.transform(
            customer_encoded[numerical_columns]
        )

        probability = model.predict_proba(customer_encoded)[0][1]

        prediction = 1 if probability >= 0.66 else 0

        result = "Yes" if prediction == 1 else "No"

        risk = (
            "High" if probability >= 0.7
            else "Medium" if probability >= 0.4
            else "Low"
        )

        return jsonify({
            "prediction": result,
            "churn_probability": round(float(probability) * 100, 2),
            "risk_level": risk
        })

    except Exception as e:
        app.logger.exception("Prediction error")

        return jsonify({
            "error": str(e)
        }), 400


# ==========================================
# 10. DATASET ANALYTICS
# ==========================================

@app.route("/analytics", methods=["GET"])
def analytics():

    try:
        results = build_analytics(df)
        return jsonify(results)

    except Exception as e:
        app.logger.exception("Analytics error")

        return jsonify({
            "error": str(e)
        }), 500


# ==========================================
# 11. SHAP EXPLANATION
# ==========================================

@app.route("/explain", methods=["POST"])
def explain():

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No customer data received"
            }), 400

        customer_df = pd.DataFrame([data])

        explanation = explain_customer(
            explainer,
            customer_df,
            feature_columns,
            numerical_columns,
            scaler
        )

        return jsonify(explanation)

    except Exception as e:
        app.logger.exception("SHAP explanation error")

        return jsonify({
            "error": str(e)
        }), 400


# ==========================================
# 12. MODEL PERFORMANCE
# ==========================================

@app.route("/model-performance", methods=["GET"])
def model_performance():

    try:
        results = evaluate_model(
            model,
            df,
            feature_columns,
            numerical_columns,
            scaler,
            threshold=0.66
        )

        return jsonify(results)

    except Exception as e:
        app.logger.exception("Model evaluation error")

        return jsonify({
            "error": str(e)
        }), 500


# ==========================================
# 13. RUN APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(debug=True, port=5000)
