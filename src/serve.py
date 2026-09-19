from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Enterprise MLOps Model Serving API", version="1.0.0")

class PredictRequest(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        model = joblib.load("models/latest_model.pkl")
        data = np.array(request.features).reshape(1, -1)
        prediction = model.predict(data)
        probability = model.predict_proba(data).max()
        return {
            "prediction": int(prediction[0]),
            "confidence": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
