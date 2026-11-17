
import pandas as pd
from pathlib import Path


def build_analytics(df):

    # 1. Overall churn distribution
    churn_counts = df["Churn"].value_counts()

    churn_distribution = {
        "labels": churn_counts.index.tolist(),
        "values": churn_counts.values.tolist()
    }

    # 2. Churn percentage by contract
    contract_churn = pd.crosstab(
        df["Contract"],
        df["Churn"],
        normalize="index"
    ) * 100

    contract_data = {
        "labels": contract_churn.index.tolist(),
        "churn": contract_churn["Yes"].round(2).tolist(),
        "retained": contract_churn["No"].round(2).tolist()
    }

    # 3. Churn percentage by tenure
    df = df.copy()

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, 72],
        labels=[
            "0-12 Months",
            "13-24 Months",
            "25-48 Months",
            "49-60 Months",
            "61-72 Months"
        ]
    )

    tenure_churn = pd.crosstab(
        df["TenureGroup"],
        df["Churn"],
        normalize="index"
    ) * 100

    tenure_data = {
        "labels": tenure_churn.index.astype(str).tolist(),
        "values": tenure_churn["Yes"].round(2).tolist()
    }

    # 4. Average monthly charges by churn status
    monthly_charges = df.groupby("Churn")[
        "MonthlyCharges"
    ].mean()

    monthly_data = {
        "labels": monthly_charges.index.tolist(),
        "values": monthly_charges.round(2).tolist()
    }

    return {
        "churn_distribution": churn_distribution,
        "contract_churn": contract_data,
        "tenure_churn": tenure_data,
        "monthly_charges": monthly_data
    }
