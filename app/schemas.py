from pydantic import BaseModel

class Message(BaseModel):
    message: str

class PredictionResponse(BaseModel):
    prediction: str