"""
Email Classification Model Module.

This module contains the EmailClassifier class that handles the machine learning
model for email classification (spam vs ham). It includes data loading, model
training, and prediction functionality using scikit-learn's MultinomialNB with
CountVectorizer for text preprocessing.

The model uses a combination of CountVectorizer for text feature extraction and
MultinomialNB for classification, providing a simple yet effective approach
for email spam detection.
"""

import os
import pickle
from typing import List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


class EmailClassifier:
    """
    Email Classification Model using Multinomial Naive Bayes.

    This class provides functionality to train and use a machine learning model
    for classifying emails as spam or ham (legitimate). The model uses text
    preprocessing with CountVectorizer and classification with MultinomialNB.

    Attributes:
        vectorizer (CountVectorizer): Text feature extraction component.
        model (MultinomialNB): Naive Bayes classification model.
        model_path (str): Path to the saved model file.

    Example:
        >>> classifier = EmailClassifier()
        >>> classifier.train()
        >>> prediction = classifier.predict("Win a free iPhone now!")
        >>> print(f"Prediction: {'spam' if prediction == 1 else 'ham'}")
    """

    def __init__(self) -> None:
        """
        Initialize the EmailClassifier model.

        Sets up the CountVectorizer for text preprocessing and MultinomialNB
        for classification. The model will be trained and saved to 'model.pkl' file.
        """
        self.vectorizer: CountVectorizer = CountVectorizer()
        self.model: MultinomialNB = MultinomialNB()
        self.model_path: str = 'model.pkl'

    def list_emails(self) -> List[Tuple[str, str]]:
        """
        Load email data from CSV file.

        Reads the email dataset from the data directory. The CSV file
        should contain columns for email labels and messages.

        Returns:
            List[Tuple[str, str]]: List of email data as tuples containing
                (label, message) pairs.

        Raises:
            FileNotFoundError: If the data file 'data/emails.csv' is not found.
            pd.errors.EmptyDataError: If the CSV file is empty.
            pd.errors.ParserError: If the CSV file has invalid format.

        Example:
            >>> emails = classifier.list_emails()
            >>> print(f"Loaded {len(emails)} emails")
        """
        data: pd.DataFrame = pd.read_csv(
            'data/emails.csv', 
            sep='|', 
            header=0, 
            names=['label', 'message']
        )
        emails: List[Tuple[str, str]] = list(data.itertuples(index=False, name=None))
        return emails
    
    def prepare_data(self, data: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Prepare email data for training.

        Converts raw email data into a pandas DataFrame and maps labels
        to numerical values for machine learning.

        Args:
            data: List of email data as tuples containing (label, message) pairs.

        Returns:
            pd.DataFrame: Prepared data with columns 'label' and 'message',
                where labels are mapped to numerical values (spam=1, ham=0).

        Example:
            >>> emails = classifier.list_emails()
            >>> df = classifier.prepare_data(emails)
            >>> print(f"Prepared {len(df)} emails for training")
        """
        df: pd.DataFrame = pd.DataFrame(data, columns=['label', 'message'])
        df['label'] = df['label'].map({'spam': 1, 'ham': 0})
        return df

    def train(self) -> None:
        """
        Train the email classification model.

        This method loads the training data, prepares it for machine learning,
        trains a MultinomialNB model with CountVectorizer preprocessing,
        and saves the trained model to disk.

        The training process includes:
        - Loading email data from CSV
        - Preparing and preprocessing text data
        - Training the Naive Bayes classifier
        - Saving the trained model and vectorizer

        Raises:
            FileNotFoundError: If the data file cannot be loaded.
            ValueError: If the data contains invalid values or missing required columns.
            Exception: If model training fails due to insufficient data or other issues.

        Example:
            >>> classifier = EmailClassifier()
            >>> classifier.train()
            >>> print("Model trained and saved successfully")
        """
        data: List[Tuple[str, str]] = self.list_emails()
        df: pd.DataFrame = self.prepare_data(data)
        
        X_train = self.vectorizer.fit_transform(df['message'])
        y_train = df['label']
        self.model.fit(X_train, y_train)
        
        with open(self.model_path, 'wb') as f:
            pickle.dump((self.vectorizer, self.model), f)
            
    def predict(self, message: str) -> int:
        """
        Predict email classification (spam or ham).

        Loads the trained model from disk and makes a prediction for the given
        email message. The prediction is returned as an integer (0 for ham, 1 for spam).

        Args:
            message: Email message text to classify.

        Returns:
            int: Prediction result (0 for ham, 1 for spam).

        Raises:
            FileNotFoundError: If the trained model file is not found.
            ValueError: If message is empty or invalid.
            Exception: If prediction fails due to model loading or inference issues.

        Example:
            >>> classifier = EmailClassifier()
            >>> result = classifier.predict("Win a free iPhone now!")
            >>> print(f"Prediction: {'spam' if result == 1 else 'ham'}")
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file '{self.model_path}' not found. Please train the model first."
            )
        
        with open(self.model_path, 'rb') as f:
            self.vectorizer, self.model = pickle.load(f)
            
        message_vec = self.vectorizer.transform([message])
        return self.model.predict(message_vec)[0]