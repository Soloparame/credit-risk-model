from pydantic import BaseModel

class PredictionRequest(BaseModel):
    transaction_amount: float
    transaction_hour: int
    transaction_day: int
    transaction_month: int
    transaction_year: int
    total_transaction_amount: float
    avg_transaction_amount: float
    transaction_count: float
    std_transaction_amount: float

class PredictionResponse(BaseModel):
    risk_probability: float
