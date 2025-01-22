import os
import pytest
import pandas as pd
from app.email_classifier import EmailClassifier

@pytest.fixture
def email_classifier():
    classifier = EmailClassifier()
    return classifier

def test_list_emails(email_classifier):
    emails = email_classifier.list_emails()
    assert isinstance(emails, list)
    assert len(emails) > 0
    assert isinstance(emails[0], tuple)

def test_prepare_data(email_classifier):
    data = email_classifier.list_emails()
    df = email_classifier.prepare_data(data)
    assert isinstance(df, pd.DataFrame)
    assert 'label' in df.columns
    assert 'message' in df.columns
    assert df['label'].isnull().sum() == 0
    assert df['message'].isnull().sum() == 0

def test_train(email_classifier):
    email_classifier.train()
    assert os.path.exists(email_classifier.model_path)

def test_predict(email_classifier):
    email_classifier.train()
    prediction = email_classifier.predict("Win a free iPhone now!")
    assert prediction in [0, 1]  # 0 for ham, 1 for spam