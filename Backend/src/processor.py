import pandas as pd
import re
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class SwiggyPreprocessor:
    def __init__(self, max_features=5000, max_len=200):
        self.max_features = max_features
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=self.max_features)

    def clean_text(self, text):
        # Converts text to lowercase and removes non-alphanumeric characters
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.strip()

    def prepare_data(self, csv_path):
        # Reads the Swiggy dataset and drops rows with missing values
        data = pd.read_csv(csv_path)
        data = data.dropna(subset=['Review', 'Avg Rating'])
        
        # Sentiment Labeling: Positive (1) if rating > 3.5, else Negative (0)
        data['sentiment'] = data['Avg Rating'].apply(lambda x: 1 if x > 3.5 else 0)
        data['cleaned_review'] = data['Review'].apply(self.clean_text)
        
        # Tokenization: Builds the word index and converts text to sequences
        self.tokenizer.fit_on_texts(data['cleaned_review'])
        sequences = self.tokenizer.texts_to_sequences(data['cleaned_review'])
        
        # Padding: Ensures all sequences have a length of 200
        X = pad_sequences(sequences, maxlen=self.max_len)
        y = data['sentiment'].values
        
        # Save tokenizer for later inference
        with open('models/tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
            
        return X, y