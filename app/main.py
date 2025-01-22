from fastapi import FastAPI
from .views import router
from .email_classifier import EmailClassifier

app = FastAPI()

app.include_router(router)

classifier = EmailClassifier()
classifier.train()