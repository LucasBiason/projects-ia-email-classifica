"""
Pydantic Schema Models for Email Classification API.

This module defines the Pydantic models used for request and response validation
in the email classification API. It includes schemas for email message input
and classification output to ensure data consistency and type safety.

The schemas provide automatic validation, serialization, and documentation
generation for the FastAPI endpoints.
"""

from pydantic import BaseModel


class Message(BaseModel):
    """
    Email message schema for classification requests.

    This model defines the structure and validation rules for email messages
    that are used for spam/ham classification. The message field is required
    and must be a non-empty string.

    Attributes:
        message (str): Email message text to be classified. Must be a non-empty
            string containing the email content.

    Example:
        >>> message = Message(message="Win a free iPhone now!")
        >>> print(message.message)
        "Win a free iPhone now!"
    """

    message: str


class PredictionResponse(BaseModel):
    """
    Email classification response schema.

    This model defines the structure for the API response containing the
    classification result. The prediction is returned as a string indicating
    whether the email is spam or ham.

    Attributes:
        prediction (str): Classification result, either "spam" or "ham".

    Example:
        >>> response = PredictionResponse(prediction="spam")
        >>> print(f"Email classified as: {response.prediction}")
        Email classified as: spam
    """

    prediction: str