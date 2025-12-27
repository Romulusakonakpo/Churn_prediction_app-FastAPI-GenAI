from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import joblib

# --- Chargement du modèle (pipeline complet) ---
model = joblib.load("models/churn_model.pkl")
features = joblib.load("models/features.pkl")  # ["InternetService","Contract","tenure","MonthlyCharges","TotalCharges"]

app = FastAPI(title="Churn Prediction API")

# --- Payload minimal pour les 5 features ---
class CustomerInput(BaseModel):
    tenure: int
    TotalCharges: float
    MonthlyCharges: float
    InternetService: str
    Contract: str

def prepare_input(data: dict):
    df = pd.DataFrame([data])
    for col in features:
        if col not in df.columns:
            df[col] = 0 if col in ["tenure", "MonthlyCharges", "TotalCharges"] else "Unknown"
    return df[features]

@app.post("/predict")
def predict_churn(customer: CustomerInput):
    df = prepare_input(customer.dict())
    
    proba = model.predict_proba(df)[:, 1][0]
    prediction = int(proba >= 0.5)

    # --- Explication ---
    from src.llm_client import explain_churn

    explanation = explain_churn(
        churn_proba=proba,
        features=df.iloc[0].to_dict()  # seulement les 5 features
    )

    return {
        "prediction": prediction,
        "probability": round(proba, 4),
        "explanation": explanation
    }
