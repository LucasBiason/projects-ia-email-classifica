import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

class EmailClassifier:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()
        self.model_path = 'model.pkl'

    def list_emails(self):
        data = pd.read_csv('data/emails.csv', sep='|', header=0, names=['label', 'message'])
        emails = list(data.itertuples(index=False, name=None))
        return emails
    
    def prepare_data(self, data):
        df = pd.DataFrame(data, columns=['label', 'message'])
        df['label'] = df['label'].map({'spam': 1, 'ham': 0})
        return df

    def train(self):
        data = self.list_emails()
        df = self.prepare_data(data)
        
        X_train = self.vectorizer.fit_transform(df['message'])
        y_train = df['label']
        self.model.fit(X_train, y_train)
        with open(self.model_path, 'wb') as f:
            pickle.dump((self.vectorizer, self.model), f)
            
    def predict(self, message):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file '{self.model_path}' not found. Please train the model first.")
        
        with open(self.model_path, 'rb') as f:
            self.vectorizer, self.model = pickle.load(f)
            
        message_vec = self.vectorizer.transform([message])
        return self.model.predict(message_vec)[0]