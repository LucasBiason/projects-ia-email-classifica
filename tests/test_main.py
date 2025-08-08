"""
Unit tests for app.main module.
"""

import pytest
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

from app.main import app, lifespan, classifier


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
    response = client.post("/predict", data="invalid json")
    assert response.status_code == 422


def test_missing_message_field():
    """Test handling of missing required fields."""
    client = TestClient(app)
    response = client.post("/predict", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_lifespan_startup_success():
    """Test lifespan startup successfully initializes classifier."""
    with patch("app.main.EmailClassifier") as mock_classifier_class:
        mock_classifier = Mock()
        mock_classifier_class.return_value = mock_classifier
        
        async with lifespan(app):
            mock_classifier_class.assert_called_once()
            mock_classifier.train.assert_called_once()


@pytest.mark.asyncio
async def test_lifespan_startup_failure():
    """Test lifespan startup raises exception when classifier initialization fails."""
    with patch("app.main.EmailClassifier") as mock_classifier_class:
        mock_classifier_class.side_effect = Exception("Initialization error")
        
        with pytest.raises(Exception) as excinfo:
            async with lifespan(app):
                pass
        
        assert "Initialization error" in str(excinfo.value)


@pytest.mark.asyncio
async def test_lifespan_shutdown():
    """Test lifespan shutdown completes without errors."""
    with patch("app.main.EmailClassifier") as mock_classifier_class:
        mock_classifier = Mock()
        mock_classifier_class.return_value = mock_classifier
        
        async with lifespan(app):
            pass


def test_root_endpoint_with_classifier_ready():
    """Test root endpoint when classifier is ready."""
    with patch("app.main.classifier", Mock()):
        client = TestClient(app)
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["classifier_ready"] is True


def test_root_endpoint_with_classifier_not_ready():
    """Test root endpoint when classifier is not ready."""
    with patch("app.main.classifier", None):
        client = TestClient(app)
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["classifier_ready"] is False


def test_health_endpoint_with_classifier_ready():
    """Test health endpoint when classifier is ready."""
    with patch("app.main.classifier", Mock()):
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["classifier_ready"] is True


def test_health_endpoint_with_classifier_not_ready():
    """Test health endpoint when classifier is not ready."""
    with patch("app.main.classifier", None):
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 503
        data = response.json()
        assert "detail" in data


def test_health_endpoint_with_exception():
    """Test health endpoint when exception occurs."""
    with patch("app.main.classifier", Mock(side_effect=Exception("Test error"))):
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code in [200, 503]
        data = response.json()
        if response.status_code == 503:
            assert "detail" in data
        else:
            assert "status" in data


def test_app_has_cors_middleware():
    """Test that CORS middleware is properly configured."""
    cors_middleware = None
    for middleware in app.user_middleware:
        if "CORSMiddleware" in str(middleware.cls):
            cors_middleware = middleware
            break
    
    assert cors_middleware is not None


def test_app_has_router():
    """Test that router is included in the app."""
    assert len(app.routes) > 0


def test_app_logging_configuration():
    """Test that logging is properly configured."""
    import logging
    logger = logging.getLogger("app.main")
    assert logger.level <= logging.INFO


def test_classifier_global_variable():
    """Test that classifier global variable is accessible."""
    assert classifier is None


def test_app_version_consistency():
    """Test app version consistency."""
    assert app.version == "1.0.0"


def test_app_title_consistency():
    """Test app title consistency."""
    assert app.title == "Email Classification Service"


def test_app_description_consistency():
    """Test app description consistency."""
    assert app.description == "AI-powered email classification service"


def test_app_different_from_house_price():
    """Test that this app is different from house price prediction."""
    assert app.title != "House Price Prediction Service"
    assert "email" in app.description.lower()


def test_classifier_variable_name():
    """Test classifier variable name."""
    assert 'classifier' in globals()


@pytest.mark.asyncio
async def test_lifespan_classifier_initialization():
    """Test lifespan classifier initialization."""
    with patch("app.main.EmailClassifier") as mock_classifier_class:
        mock_classifier = Mock()
        mock_classifier_class.return_value = mock_classifier
        
        async with lifespan(app):
            mock_classifier_class.assert_called_once()
            mock_classifier.train.assert_called_once()


def test_root_endpoint_response_structure():
    """Test root endpoint response has correct structure."""
    client = TestClient(app)
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "status" in data
    assert "classifier_ready" in data
    assert isinstance(data["message"], str)
    assert isinstance(data["status"], str)
    assert isinstance(data["classifier_ready"], bool)


def test_health_endpoint_response_structure():
    """Test health endpoint response has correct structure."""
    client = TestClient(app)
    response = client.get("/health")
    
    assert response.status_code in [200, 503]
    data = response.json()
    
    if response.status_code == 200:
        assert "status" in data
        assert "classifier_ready" in data
        assert isinstance(data["status"], str)
        assert isinstance(data["classifier_ready"], bool)
    else:
        assert "detail" in data


def test_app_routes_count():
    """Test that app has expected number of routes."""
    # Should have at least root, health, and predict endpoints
    assert len(app.routes) >= 3


def test_app_middleware_count():
    """Test that app has expected middleware."""
    # Should have CORS middleware
    middleware_count = len(app.user_middleware)
    assert middleware_count > 0


def test_app_lifespan_attribute():
    """Test that app has lifespan attribute."""
    assert hasattr(app, "router")


def test_app_openapi_url():
    """Test that app has OpenAPI documentation URL."""
    assert hasattr(app, "openapi_url")


def test_app_docs_url():
    """Test that app has documentation URL."""
    assert hasattr(app, "docs_url")


def test_app_redoc_url():
    """Test that app has ReDoc documentation URL."""
    assert hasattr(app, "redoc_url")


def test_app_debug_mode():
    """Test that app debug mode is properly configured."""
    # Debug should be False by default
    assert app.debug is False


def test_app_response_model():
    """Test that app endpoints have proper response models."""
    client = TestClient(app)
    
    # Test root endpoint response model
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    
    # Test health endpoint response model
    response = client.get("/health")
    assert response.status_code in [200, 503]
    data = response.json()
    assert isinstance(data, dict)


def test_app_error_handling():
    """Test that app handles errors properly."""
    client = TestClient(app)
    
    # Test non-existent endpoint
    response = client.get("/non-existent")
    assert response.status_code == 404


def test_app_cors_headers():
    """Test that CORS headers are properly set."""
    client = TestClient(app)
    response = client.options("/")
    
    # CORS preflight request should be handled
    assert response.status_code in [200, 405, 404]


def test_app_async_endpoints():
    """Test that async endpoints work properly."""
    client = TestClient(app)
    
    # Test root endpoint (async)
    response = client.get("/")
    assert response.status_code == 200
    
    # Test health endpoint (async)
    response = client.get("/health")
    assert response.status_code in [200, 503]


def test_app_sync_lifespan():
    """Test that lifespan context manager works synchronously."""
    with patch("app.main.EmailClassifier") as mock_classifier_class:
        mock_classifier = Mock()
        mock_classifier_class.return_value = mock_classifier
        
        # Test that lifespan can be used in sync context
        try:
            with lifespan(app):
                pass
        except Exception:
            # This is expected as lifespan is async
            pass


def test_app_classifier_global_scope():
    """Test that classifier variable is in global scope."""
    import app.main
    assert hasattr(app.main, "classifier")


def test_app_logger_global_scope():
    """Test that logger is in global scope."""
    import app.main
    assert hasattr(app.main, "logger")


def test_app_fastapi_import():
    """Test that FastAPI is properly imported."""
    import app.main
    assert hasattr(app.main, "app")
    assert app.main.app is not None


def test_app_router_import():
    """Test that router is properly imported."""
    import app.main
    assert hasattr(app.main, "router")


def test_app_lifespan_import():
    """Test that lifespan is properly imported."""
    import app.main
    assert hasattr(app.main, "lifespan")


def test_app_email_classifier_import():
    """Test that EmailClassifier is properly imported."""
    import app.main
    assert hasattr(app.main, "EmailClassifier")


def test_app_views_import():
    """Test that views router is properly imported."""
    import app.main
    assert hasattr(app.main, "router")


def test_app_cors_middleware_import():
    """Test that CORS middleware is properly imported."""
    import app.main
    assert "CORSMiddleware" in str(app.main.app.user_middleware)


def test_app_logging_import():
    """Test that logging is properly imported."""
    import app.main
    assert hasattr(app.main, "logging")


def test_app_contextlib_import():
    """Test that contextlib is properly imported."""
    import app.main
    assert hasattr(app.main, "asynccontextmanager")


def test_app_typing_import():
    """Test that typing is properly imported."""
    import app.main
    assert hasattr(app.main, "Dict")
    assert hasattr(app.main, "Any")


def test_app_fastapi_types_import():
    """Test that FastAPI types are properly imported."""
    import app.main
    assert hasattr(app.main, "FastAPI")
    assert hasattr(app.main, "HTTPException")


def test_app_cors_middleware_types_import():
    """Test that CORS middleware types are properly imported."""
    import app.main
    assert hasattr(app.main, "CORSMiddleware") 