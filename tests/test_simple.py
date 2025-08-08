"""
Testes simples para verificar se o ambiente está funcionando.
"""

import pytest
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

# Mock the EmailClassifier before importing app.main
with patch('app.email_classifier.EmailClassifier') as mock_classifier:
    mock_classifier.return_value = Mock()
    from app.main import app


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


def test_missing_message_field():
    """Test handling of missing required fields."""
    client = TestClient(app)
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_simple_assertion():
    """Test simple assertion to verify pytest is working."""
    assert 1 + 1 == 2
    assert "hello" == "hello"
    assert True is True 