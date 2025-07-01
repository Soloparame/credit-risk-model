from fastapi import FastAPI
import mlflow.pyfunc
from pydantic import BaseModel
from src.api.pydantic_models import PredictionRequest, PredictionResponse
import pandas as pd

app = FastAPI()
model_name = "random_forest_model"  # Change if needed
model_uri = f"models:/{model_name}/Production"
model = mlflow.pyfunc.load_model(model_uri)

@app.get("/")
def root():
    return {"message": "Credit Risk API is running."}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    input_df = pd.DataFrame([data.dict()])
    prob = model.predict(input_df)[0]
    return PredictionResponse(risk_probability=float(prob))
