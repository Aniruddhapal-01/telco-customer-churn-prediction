
import pandas as pd
import numpy as np
import shap


def prepare_model_input(
    raw_df,
    feature_columns,
    numerical_columns,
    scaler
):
    """
    Convert raw customer records into the exact
    feature format expected by the saved model.
    """

    X = raw_df.drop(
        columns=["Churn", "customerID"],
        errors="ignore"
    ).copy()

    for col in numerical_columns:
        X[col] = pd.to_numeric(X[col], errors="raise")

    X_encoded = pd.get_dummies(X)

    X_encoded = X_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    X_encoded[numerical_columns] = scaler.transform(
        X_encoded[numerical_columns]
    )

    return X_encoded


def create_explainer(
    model,
    df,
    feature_columns,
    numerical_columns,
    scaler
):
    """
    Create a representative background dataset
    and initialise the SHAP explainer.
    """

    background_raw = df.sample(
        n=min(100, len(df)),
        random_state=42
    )

    background = prepare_model_input(
        background_raw,
        feature_columns,
        numerical_columns,
        scaler
    )

    explainer = shap.LinearExplainer(
        model,
        background
    )

    return explainer


def explain_customer(
    explainer,
    customer_df,
    feature_columns,
    numerical_columns,
    scaler
):
    """
    Calculate feature contributions for one customer.
    """

    customer_encoded = prepare_model_input(
        customer_df,
        feature_columns,
        numerical_columns,
        scaler
    )

    explanation = explainer(customer_encoded)

    values = np.asarray(explanation.values)

    # Handle binary classifier output shapes
    if values.ndim == 3:
        values = values[0, :, 1]
    elif values.ndim == 2:
        values = values[0]

    # Match contributions with feature names
    contributions = pd.DataFrame({
        "feature": feature_columns,
        "impact": values
    })

    # Select the 8 strongest contributions
    contributions["absolute_impact"] = (
        contributions["impact"].abs()
    )

    contributions = contributions.sort_values(
        "absolute_impact",
        ascending=False
    ).head(8)

    # Sort for a horizontal bar chart
    contributions = contributions.sort_values("impact")

    features = []

    for _, row in contributions.iterrows():

        impact = float(row["impact"])

        features.append({
            "feature": str(row["feature"]),
            "impact": round(impact, 5),
            "direction": (
                "toward_churn"
                if impact > 0
                else "toward_stay"
            )
        })

    return {
        "features": features,
        "explanation_type": "SHAP feature contributions"
    }
