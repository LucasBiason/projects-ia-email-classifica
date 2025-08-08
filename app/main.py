"""
Email Classification Service Main Application.

This module contains the main FastAPI application for email classification
service with proper initialization and error handling.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .views import router, classifier
from .email_classifier import EmailClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global classifier instance
classifier_instance: Optional[EmailClassifier] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    global classifier_instance
    try:
        logger.info("Initializing EmailClassifier...")
        classifier_instance = EmailClassifier()
        classifier_instance.train()
        # Set the global classifier in views
        from .views import classifier
        classifier = classifier_instance
        logger.info("EmailClassifier initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize EmailClassifier: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down EmailClassification service...")


app = FastAPI(
    title="Email Classification Service",
    description="AI-powered email classification service",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)


@app.get("/")
async def root() -> Dict[str, Any]:
    """
    Root endpoint to verify service status.
    
    Returns:
        Dict[str, Any]: Service status information.
    """
    return {
        "message": "Email Classification Service is online!",
        "status": "healthy",
        "classifier_ready": classifier_instance is not None
    }


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Returns:
        Dict[str, Any]: Health status information.
    """
    try:
        if classifier_instance is None:
            raise HTTPException(status_code=503, detail="Classifier not ready")
        
        return {
            "status": "healthy",
            "classifier_ready": True
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")