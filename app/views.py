from fastapi import APIRouter
from .schemas import Message, PredictionResponse
from .email_classifier import EmailClassifier

router = APIRouter()

classifier = EmailClassifier()

@router.post("/predict", response_model=PredictionResponse)
def predict(message: Message):
    prediction = classifier.predict(message.message)
    return {"prediction": "spam" if prediction == 1 else "ham"}