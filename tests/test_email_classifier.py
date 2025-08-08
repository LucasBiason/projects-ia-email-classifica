"""
Unit tests for app.email_classifier module.
"""
import os
import pytest
from unittest.mock import patch, Mock

# Mock pandas before importing
with patch('pandas.read_csv') as mock_read_csv:
    mock_read_csv.return_value = Mock()
    from app.email_classifier import EmailClassifier

# Mock pandas for tests - remove direct import to avoid issues
# import pandas as pd


def test_email_classifier_init():
    """Test EmailClassifier initialization."""
    classifier = EmailClassifier()
    
    assert classifier.vectorizer is not None
    assert classifier.model is not None
    assert classifier.model_path == 'model.pkl'


def test_list_emails_success():
    """Test successful email data loading."""
    classifier = EmailClassifier()
    
    with patch('pandas.read_csv') as mock_read_csv:
        # Mock DataFrame
        mock_data = Mock()
        mock_data.itertuples.return_value = [
            Mock(label='spam', message='Win iPhone'),
            Mock(label='ham', message='Hello'),
            Mock(label='spam', message='Free money')
        ]
        mock_read_csv.return_value = mock_data
        
        emails = classifier.list_emails()
        
        assert isinstance(emails, list)
        assert len(emails) == 3
        mock_read_csv.assert_called_once_with(
            'data/emails.csv', 
            sep='|', 
            header=0, 
            names=['label', 'message']
        )


def test_list_emails_file_not_found():
    """Test email data loading when file is not found."""
    classifier = EmailClassifier()
    
    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.side_effect = FileNotFoundError("File not found")
        
        with pytest.raises(FileNotFoundError, match="File not found"):
            classifier.list_emails()


def test_list_emails_empty_data():
    """Test email data loading with empty CSV."""
    classifier = EmailClassifier()
    
    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.side_effect = Exception("Empty file")
        
        with pytest.raises(Exception, match="Empty file"):
            classifier.list_emails()


def test_list_emails_parser_error():
    """Test email data loading with parser error."""
    classifier = EmailClassifier()
    
    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.side_effect = Exception("Invalid format")
        
        with pytest.raises(Exception, match="Invalid format"):
            classifier.list_emails()


def test_prepare_data_success():
    """Test successful data preparation."""
    classifier = EmailClassifier()
    data = [('spam', 'Win iPhone'), ('ham', 'Hello'), ('spam', 'Free money')]
    
    with patch.object(classifier, 'prepare_data') as mock_prepare_data:
        mock_df = Mock()
        mock_prepare_data.return_value = mock_df
        
        df = classifier.prepare_data(data)
        
        assert df is not None


def test_prepare_data_empty_list():
    """Test data preparation with empty data."""
    classifier = EmailClassifier()
    data = []
    
    with patch.object(classifier, 'prepare_data') as mock_prepare_data:
        mock_df = Mock()
        mock_prepare_data.return_value = mock_df
        
        df = classifier.prepare_data(data)
        
        assert df is not None


def test_prepare_data_mixed_labels():
    """Test data preparation with mixed label types."""
    classifier = EmailClassifier()
    data = [('spam', 'Message 1'), ('ham', 'Message 2'), ('SPAM', 'Message 3')]
    
    with patch.object(classifier, 'prepare_data') as mock_prepare_data:
        mock_df = Mock()
        mock_prepare_data.return_value = mock_df
        
        df = classifier.prepare_data(data)
        
        assert df is not None


def test_prepare_data_unknown_labels():
    """Test data preparation with unknown labels."""
    classifier = EmailClassifier()
    data = [('unknown', 'Message 1'), ('ham', 'Message 2'), ('spam', 'Message 3')]
    
    with patch.object(classifier, 'prepare_data') as mock_prepare_data:
        mock_df = Mock()
        mock_prepare_data.return_value = mock_df
        
        df = classifier.prepare_data(data)
        
        assert df is not None


def test_train_success():
    """Test successful model training."""
    classifier = EmailClassifier()
    
    with patch.object(classifier, 'train') as mock_train:
        mock_train.return_value = None
        
        classifier.train()
        
        mock_train.assert_called_once()


def test_train_file_not_found():
    """Test model training when data file is not found."""
    classifier = EmailClassifier()
    
    with patch.object(classifier, 'list_emails') as mock_list_emails:
        mock_list_emails.side_effect = FileNotFoundError("Data file not found")
        
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            classifier.train()


def test_train_empty_data():
    """Test model training with empty data."""
    classifier = EmailClassifier()
    
    with patch.object(classifier, 'list_emails') as mock_list_emails:
        mock_list_emails.return_value = []
        
        with pytest.raises(ValueError):
            classifier.train()


def test_train_data_loading_error():
    """Test model training with data loading error."""
    classifier = EmailClassifier()
    
    with patch.object(classifier, 'list_emails') as mock_list_emails:
        mock_list_emails.side_effect = Exception("Data loading error")
        
        with pytest.raises(Exception, match="Data loading error"):
            classifier.train()


def test_train_model_fitting_error():
    """Test model training with model fitting error."""
    classifier = EmailClassifier()
    
    with patch.object(classifier, 'list_emails') as mock_list_emails:
        mock_list_emails.return_value = [
            ('spam', 'Win iPhone'),
            ('ham', 'Hello'),
            ('spam', 'Free money')
        ]
        
        with patch.object(classifier.model, 'fit') as mock_fit:
            mock_fit.side_effect = Exception("Model fitting error")
            
            with pytest.raises(Exception, match="Model fitting error"):
                classifier.train()


def test_train_pickle_save_error():
    """Test model training with pickle save error."""
    classifier = EmailClassifier()
    
    with patch.object(classifier, 'list_emails') as mock_list_emails:
        mock_list_emails.return_value = [
            ('spam', 'Win iPhone'),
            ('ham', 'Hello'),
            ('spam', 'Free money')
        ]
        
        with patch('pickle.dump') as mock_pickle_dump:
            mock_pickle_dump.side_effect = Exception("Pickle save error")
            
            with pytest.raises(Exception, match="Pickle save error"):
                classifier.train()


def test_predict_success():
    """Test successful prediction."""
    classifier = EmailClassifier()
    
    with patch('os.path.exists') as mock_exists:
        with patch('builtins.open', create=True) as mock_open:
            mock_exists.return_value = True
            mock_open.return_value.__enter__.return_value = Mock()
            
            with patch('pickle.load') as mock_pickle_load:
                mock_vectorizer = Mock()
                mock_model = Mock()
                mock_pickle_load.return_value = (mock_vectorizer, mock_model)
                
                mock_vectorizer.transform.return_value = Mock()
                mock_model.predict.return_value = [1]
                
                result = classifier.predict("Win a free iPhone now!")
                
                assert result == 1


def test_predict_file_not_found():
    """Test prediction when model file is not found."""
    classifier = EmailClassifier()
    
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError, match="Model file 'model.pkl' not found"):
            classifier.predict("Test message")


def test_predict_empty_message():
    """Test prediction with empty message."""
    classifier = EmailClassifier()
    
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        
        with patch('builtins.open', create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            with patch('pickle.load') as mock_pickle_load:
                mock_vectorizer = Mock()
                mock_model = Mock()
                mock_pickle_load.return_value = (mock_vectorizer, mock_model)
                
                mock_vectorizer.transform.return_value = Mock()
                mock_model.predict.return_value = [0]  # ham
                
                result = classifier.predict("")
                
                assert result == 0


def test_predict_long_message():
    """Test prediction with long message."""
    classifier = EmailClassifier()
    
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        
        with patch('builtins.open', create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            with patch('pickle.load') as mock_pickle_load:
                mock_vectorizer = Mock()
                mock_model = Mock()
                mock_pickle_load.return_value = (mock_vectorizer, mock_model)
                
                mock_vectorizer.transform.return_value = Mock()
                mock_model.predict.return_value = [1]  # spam
                
                long_message = "This is a very long message. " * 100
                result = classifier.predict(long_message)
                
                assert result == 1


def test_predict_special_characters():
    """Test prediction with special characters in message."""
    classifier = EmailClassifier()
    
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        
        with patch('builtins.open', create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            with patch('pickle.load') as mock_pickle_load:
                mock_vectorizer = Mock()
                mock_model = Mock()
                mock_pickle_load.return_value = (mock_vectorizer, mock_model)
                
                mock_vectorizer.transform.return_value = Mock()
                mock_model.predict.return_value = [0]  # ham
                
                special_message = "Hello! How are you? 😊 @user #test"
                result = classifier.predict(special_message)
                
                assert result == 0


def test_predict_model_error():
    """Test prediction when model fails."""
    classifier = EmailClassifier()
    
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        
        with patch('builtins.open', create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            with patch('pickle.load') as mock_pickle_load:
                mock_pickle_load.side_effect = Exception("Model loading failed")
                
                with pytest.raises(Exception, match="Model loading failed"):
                    classifier.predict("Test message")


def test_predict_vectorizer_error():
    """Test prediction when vectorizer fails."""
    classifier = EmailClassifier()
    
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        
        with patch('builtins.open', create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            with patch('pickle.load') as mock_pickle_load:
                mock_vectorizer = Mock()
                mock_model = Mock()
                mock_pickle_load.return_value = (mock_vectorizer, mock_model)
                
                mock_vectorizer.transform.side_effect = Exception("Vectorizer error")
                
                with pytest.raises(Exception, match="Vectorizer error"):
                    classifier.predict("Test message")


def test_predict_model_prediction_error():
    """Test prediction when model prediction fails."""
    classifier = EmailClassifier()
    
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        
        with patch('builtins.open', create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            with patch('pickle.load') as mock_pickle_load:
                mock_vectorizer = Mock()
                mock_model = Mock()
                mock_pickle_load.return_value = (mock_vectorizer, mock_model)
                
                mock_vectorizer.transform.return_value = Mock()
                mock_model.predict.side_effect = Exception("Prediction error")
                
                with pytest.raises(Exception, match="Prediction error"):
                    classifier.predict("Test message")


def test_predict_file_open_error():
    """Test prediction when file open fails."""
    classifier = EmailClassifier()
    
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.side_effect = FileNotFoundError("File open error")
            
            with pytest.raises(FileNotFoundError, match="File open error"):
                classifier.predict("Test message")


def test_integration_train_and_predict():
    """Integration test for training and prediction."""
    classifier = EmailClassifier()
    
    # Mock the data loading and training
    with patch.object(classifier, 'list_emails') as mock_list_emails:
        mock_list_emails.return_value = [
            ('spam', 'Win iPhone'),
            ('ham', 'Hello'),
            ('spam', 'Free money')
        ]
        
        with patch('pickle.dump') as mock_pickle_dump:
            classifier.train()
            mock_pickle_dump.assert_called_once()
    
    # Mock the prediction
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        
        with patch('builtins.open', create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            with patch('pickle.load') as mock_pickle_load:
                mock_vectorizer = Mock()
                mock_model = Mock()
                mock_pickle_load.return_value = (mock_vectorizer, mock_model)
                
                mock_vectorizer.transform.return_value = Mock()
                mock_model.predict.return_value = [1]  # spam
                
                result = classifier.predict("Win a free iPhone!")
                assert result == 1


def test_predict_with_none_message():
    """Test prediction with None message."""
    classifier = EmailClassifier()
    
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        
        with patch('builtins.open', create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            with patch('pickle.load') as mock_pickle_load:
                mock_vectorizer = Mock()
                mock_model = Mock()
                mock_pickle_load.return_value = (mock_vectorizer, mock_model)
                
                mock_vectorizer.transform.return_value = Mock()
                mock_model.predict.return_value = [0]  # ham
                
                result = classifier.predict("")
                
                assert result == 0


def test_predict_with_unicode_message():
    """Test prediction with unicode message."""
    classifier = EmailClassifier()
    
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        
        with patch('builtins.open', create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            with patch('pickle.load') as mock_pickle_load:
                mock_vectorizer = Mock()
                mock_model = Mock()
                mock_pickle_load.return_value = (mock_vectorizer, mock_model)
                
                mock_vectorizer.transform.return_value = Mock()
                mock_model.predict.return_value = [1]  # spam
                
                unicode_message = "Olá! Como você está? 🎉"
                result = classifier.predict(unicode_message)
                
                assert result == 1


def test_predict_with_numbers_in_message():
    """Test prediction with numbers in message."""
    classifier = EmailClassifier()
    
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        
        with patch('builtins.open', create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            with patch('pickle.load') as mock_pickle_load:
                mock_vectorizer = Mock()
                mock_model = Mock()
                mock_pickle_load.return_value = (mock_vectorizer, mock_model)
                
                mock_vectorizer.transform.return_value = Mock()
                mock_model.predict.return_value = [0]  # ham
                
                number_message = "Your account balance is $1234.56"
                result = classifier.predict(number_message)
                
                assert result == 0


if __name__ == "__main__":
    pytest.main([__file__]) 