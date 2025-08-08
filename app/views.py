"""
FastAPI Router Views for Email Classification API.

This module contains the FastAPI router and endpoint definitions for the email
classification service. It provides the main classification endpoint that accepts
email messages and returns spam/ham predictions using the trained machine learning
model.

The views handle request validation, model prediction, and error responses
with proper HTTP status codes and error messages.
"""

from fastapi import APIRouter, HTTPException

from .schemas import Message, PredictionResponse

router = APIRouter()

# Global classifier instance - will be initialized in main.py
classifier = None


@router.post("/predict", response_model=PredictionResponse)
def predict(message: Message) -> PredictionResponse:
    """
    Predict email classification (spam or ham).

    This endpoint accepts an email message and returns a classification
    prediction using the trained machine learning model. The prediction
    indicates whether the email is spam or legitimate (ham).

    Args:
        message (Message): Email message for classification including:
            - message: Email text content to be classified

    Returns:
        PredictionResponse: Classification result containing:
            - prediction: String indicating "spam" or "ham"

    Raises:
        HTTPException: If the model file is not found or prediction fails.
            - status_code: 500 (Internal Server Error)
            - detail: Error message describing the issue

    Example:
        >>> response = await predict(Message(message="Win a free iPhone now!"))
        >>> print(f"Email classified as: {response.prediction}")
        Email classified as: spam
    """
    try:
        if classifier is None:
            raise HTTPException(status_code=503, detail="Classifier not ready")
        
        prediction: int = classifier.predict(message.message)
        return PredictionResponse(
            prediction="spam" if prediction == 1 else "ham"
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")