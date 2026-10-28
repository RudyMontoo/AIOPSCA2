import json
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


def _ensure_model():
    """Tests run against a tiny synthetic model if a real trained one isn't
    present, so CI doesn't depend on the Kaggle dataset being committed."""
    if os.path.exists("models/model.joblib"):
        return
    os.makedirs("models", exist_ok=True)
    rows = [
        {**{n: i for n, i in zip(NUMERIC, [10, 50.0, 500.0, 0])},
         **{c: "Yes" if i % 2 == 0 else "No" for i, c in enumerate(CATEGORICAL)},
         "Churn": i % 2}
        for i in range(10)
    ]
    df = pd.DataFrame(rows)
    model = Pipeline([
        ("preprocess", ColumnTransformer([
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ])),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(df[NUMERIC + CATEGORICAL], df["Churn"])
    joblib.dump(model, "models/model.joblib")
    json.dump({"features": NUMERIC + CATEGORICAL, "model": "LogisticRegression"},
              open("models/model_metadata.json", "w"))


_ensure_model()
