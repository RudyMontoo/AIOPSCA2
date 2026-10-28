import joblib
import pandas as pd
from fastapi import FastAPI
from app.schemas import ChurnRequest, ChurnResponse

app = FastAPI(title="Churn Prediction API")
model = joblib.load("models/model.joblib")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest):
    row = pd.DataFrame([request.model_dump()])
    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0][1]
    return ChurnResponse(churn=bool(pred), churn_probability=round(float(prob), 4))
