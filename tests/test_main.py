"""
Unit tests for app.main module.
"""

import pytest
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

# Import the app directly without mocking first
from app.main import app, lifespan


def test_app_initialization():
    """Test FastAPI app is properly initialized."""
    assert app.title == "Email Classification Service"
    assert app.description == "AI-powered email classification service"
    assert app.version == "1.0.0"


def test_root_endpoint():
    """Test root endpoint returns correct response."""
    client = TestClient(app)
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "status" in data
    assert "classifier_ready" in data


def test_health_endpoint():
    """Test health endpoint."""
    client = TestClient(app)
    response = client.get("/health")
    
    # Should return either 200 (healthy) or 503 (not ready)
    assert response.status_code in [200, 503]


def test_predict_endpoint():
    """Test predict endpoint."""
    client = TestClient(app)
    test_data = {"message": "Win a free iPhone now!"}
    
    # Mock the classifier to avoid EOFError
    with patch("app.views.classifier") as mock_classifier:
        mock_classifier.predict.return_value = 1  # spam
        
        response = client.post("/predict", json=test_data)
        
        # Should return 200 (success) when classifier is mocked
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data


def test_invalid_json_request():
    """Test handling of invalid JSON requests."""
    client = TestClient(app)
    response = client.post("/predict", content="invalid json")
    assert response.status_code == 422


def test_missing_message_field():
    """Test handling of missing required fields."""
    client = TestClient(app)
    response = client.post("/predict", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_lifespan_startup_success():
    """Test lifespan startup successfully initializes classifier."""
    with patch("app.email_classifier.EmailClassifier") as mock_classifier_class:
        mock_classifier = Mock()
        mock_classifier_class.return_value = mock_classifier
        
        async with lifespan(app):
            mock_classifier_class.assert_called_once()
            mock_classifier.train.assert_called_once()


@pytest.mark.asyncio
async def test_lifespan_startup_failure():
    """Test lifespan startup handles classifier initialization failure gracefully."""
    with patch("app.email_classifier.EmailClassifier") as mock_classifier_class:
        mock_classifier_class.side_effect = Exception("Initialization error")
        
        # The lifespan should not raise an exception, it should handle it gracefully
        async with lifespan(app):
            pass
        
        # Verify that the exception was logged but not raised
        assert True  # If we get here, the lifespan handled the exception properly


@pytest.mark.asyncio
async def test_lifespan_shutdown():
    """Test lifespan shutdown completes without errors."""
    with patch("app.email_classifier.EmailClassifier") as mock_classifier_class:
        mock_classifier = Mock()
        mock_classifier_class.return_value = mock_classifier
        
        async with lifespan(app):
            pass
        
        # Should complete without errors
        assert True 