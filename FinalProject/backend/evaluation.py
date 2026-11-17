
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_model(
    model,
    df,
    feature_columns,
    numerical_columns,
    scaler,
    threshold=0.66
):

    # Separate features and target
    X = df.drop(
        columns=["Churn", "customerID"],
        errors="ignore"
    ).copy()

    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    # Ensure numeric values are properly converted
    X["TotalCharges"] = pd.to_numeric(
        X["TotalCharges"],
        errors="coerce"
    ).fillna(0)

    # Encode categorical features
    X = pd.get_dummies(X, dtype=int)

    # Match the saved model's feature order
    X = X.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Recreate the original held-out test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Apply the already-fitted scaler to test data
    X_test = X_test.copy()

    X_test[numerical_columns] = scaler.transform(
        X_test[numerical_columns]
    )

    # Get churn probabilities
    probabilities = model.predict_proba(X_test)[:, 1]

    # Apply the selected threshold
    predictions = (
        probabilities >= threshold
    ).astype(int)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    return {
        "accuracy": round(
            float(accuracy_score(y_test, predictions)) * 100, 2
        ),

        "precision": round(
            float(precision_score(
                y_test, predictions, zero_division=0
            )) * 100, 2
        ),

        "recall": round(
            float(recall_score(
                y_test, predictions, zero_division=0
            )) * 100, 2
        ),

        "f1_score": round(
            float(f1_score(
                y_test, predictions, zero_division=0
            )) * 100, 2
        ),

        "confusion_matrix": {
            "true_negative": int(cm[0][0]),
            "false_positive": int(cm[0][1]),
            "false_negative": int(cm[1][0]),
            "true_positive": int(cm[1][1])
        },

        "threshold": threshold,
        "test_samples": int(len(y_test))
    }
