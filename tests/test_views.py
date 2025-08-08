"""
Unit tests for app.views module.
"""
import pytest
from unittest.mock import patch
from fastapi import HTTPException

from app.views import router, predict
from app.schemas import Message


def test_router_initialization():
    """Test that router is properly initialized."""
    assert router is not None
    assert hasattr(router, "routes")


def test_predict_success():
    """Test predict function with valid message returns correct prediction."""
    message = Message(message="Win a free iPhone now!")
    
    with patch("app.views.classifier") as mock_classifier:
        mock_classifier.predict.return_value = 1  # spam
        
        result = predict(message)
        
        assert result.prediction == "spam"
        mock_classifier.predict.assert_called_once_with("Win a free iPhone now!")


def test_predict_ham():
    """Test predict function with ham prediction."""
    message = Message(message="Hello, how are you?")
    
    with patch("app.views.classifier") as mock_classifier:
        mock_classifier.predict.return_value = 0  # ham
        
        result = predict(message)
        
        assert result.prediction == "ham"
        mock_classifier.predict.assert_called_once_with("Hello, how are you?")


def test_predict_file_not_found_error():
    """Test predict function raises HTTPException when model file not found."""
    message = Message(message="Test message")
    
    with patch("app.views.classifier") as mock_classifier:
        mock_classifier.predict.side_effect = FileNotFoundError("Model file not found")
        
        with pytest.raises(HTTPException) as excinfo:
            predict(message)
        
        assert excinfo.value.status_code == 500
        assert "Model file not found" in str(excinfo.value.detail)


def test_predict_general_exception():
    """Test predict function raises Exception for general exceptions."""
    message = Message(message="Test message")
    
    with patch("app.views.classifier") as mock_classifier:
        mock_classifier.predict.side_effect = Exception("General error")
        
        with pytest.raises(Exception) as excinfo:
            predict(message)
        
        assert "General error" in str(excinfo.value)


def test_predict_empty_message():
    """Test predict function with empty message."""
    message = Message(message="")
    
    with patch("app.views.classifier") as mock_classifier:
        mock_classifier.predict.return_value = 0  # ham
        
        result = predict(message)
        
        assert result.prediction == "ham"
        mock_classifier.predict.assert_called_once_with("")


def test_predict_long_message():
    """Test predict function with long message."""
    long_message = "This is a very long message. " * 100
    message = Message(message=long_message)
    
    with patch("app.views.classifier") as mock_classifier:
        mock_classifier.predict.return_value = 1  # spam
        
        result = predict(message)
        
        assert result.prediction == "spam"
        mock_classifier.predict.assert_called_once_with(long_message)


def test_router_routes():
    """Test that router has the expected routes."""
    routes = [route.path for route in router.routes]
    assert "/predict" in routes 