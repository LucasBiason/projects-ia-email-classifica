"""
Unit tests for app.schemas module.
"""
import pytest
from pydantic import ValidationError

from app.schemas import Message, PredictionResponse


def test_message_valid_data():
    """Test Message schema with valid data."""
    message = Message(message="Hello, this is a test email")
    
    assert message.message == "Hello, this is a test email"
    assert isinstance(message.message, str)


def test_message_empty_string():
    """Test Message schema with empty string."""
    message = Message(message="")
    
    assert message.message == ""
    assert isinstance(message.message, str)


def test_message_missing_required_field():
    """Test Message schema with missing required field."""
    with pytest.raises(ValidationError):
        Message()


def test_prediction_response_valid_data():
    """Test PredictionResponse schema with valid data."""
    response = PredictionResponse(prediction="spam")
    
    assert response.prediction == "spam"
    assert isinstance(response.prediction, str)


def test_prediction_response_ham():
    """Test PredictionResponse schema with ham prediction."""
    response = PredictionResponse(prediction="ham")
    
    assert response.prediction == "ham"
    assert isinstance(response.prediction, str)


def test_prediction_response_missing_required_field():
    """Test PredictionResponse schema with missing required field."""
    with pytest.raises(ValidationError):
        PredictionResponse()


def test_message_with_special_characters():
    """Test Message schema with special characters."""
    message = Message(message="Hello! How are you? 😊 @user #test")
    
    assert message.message == "Hello! How are you? 😊 @user #test"
    assert isinstance(message.message, str)


def test_message_with_unicode():
    """Test Message schema with unicode characters."""
    message = Message(message="Olá, como você está? 你好")
    
    assert message.message == "Olá, como você está? 你好"
    assert isinstance(message.message, str)


def test_prediction_response_invalid_prediction():
    """Test PredictionResponse schema with invalid prediction value."""
    # This should work as any string is valid
    response = PredictionResponse(prediction="invalid")
    
    assert response.prediction == "invalid"
    assert isinstance(response.prediction, str)


def test_message_very_long():
    """Test Message schema with very long message."""
    long_message = "This is a very long message. " * 1000
    message = Message(message=long_message)
    
    assert message.message == long_message
    assert len(message.message) > 10000


def test_prediction_response_case_sensitive():
    """Test PredictionResponse schema with different cases."""
    response1 = PredictionResponse(prediction="SPAM")
    response2 = PredictionResponse(prediction="spam")
    response3 = PredictionResponse(prediction="HAM")
    response4 = PredictionResponse(prediction="ham")
    
    assert response1.prediction == "SPAM"
    assert response2.prediction == "spam"
    assert response3.prediction == "HAM"
    assert response4.prediction == "ham" 