import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout

def build_model(max_features=5000, max_len=200):
    model = Sequential([
        # Embedding: Maps integer word indices to 16-dimensional dense vectors
        Embedding(input_dim=max_features, output_dim=16, input_length=max_len),
        
        # Bidirectional LSTM: Captures context from sequential data from both directions
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        
        Bidirectional(LSTM(32)),
        Dense(64, activation='relu'),
        
        # Output layer with sigmoid for binary classification
        Dense(1, activation='sigmoid')
    ])
    
    # Configure with binary crossentropy loss and adam optimizer
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model