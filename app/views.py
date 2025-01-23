from fastapi import APIRouter, HTTPException
from .schemas import Message, PredictionResponse
from .email_classifier import EmailClassifier

router = APIRouter()

classifier = EmailClassifier()

@router.post("/predict", response_model=PredictionResponse)
def predict(message: Message):
    try:
        prediction = classifier.predict(message.message)
        return {"prediction": "spam" if prediction == 1 else "ham"}
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))