from fastapi import FastAPI
import tensorflow as tf
import pickle
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI(title="Swiggy Sentiment API")

# Load model and tokenizer on startup
# Ensure these paths are correct relative to your Docker WORKDIR
model = tf.keras.models.load_model('models/swiggy_rnn.keras')
with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

class ReviewRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict_sentiment(request: ReviewRequest):
    # 1. Preprocessing individual review
    clean_text = request.text.lower()
    seq = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(seq, maxlen=200)
    
    # 2. Predict probability (Value between 0 and 1)
    prediction = float(model.predict(padded)[0][0])
    
    # 3. FIXED LOGIC: Mapping the model output to the correct label
    # Swapped: High probability (>= 0.5) is now Negative
    if prediction >= 0.5:
        sentiment = "Negative"
        confidence = prediction
    else:
        sentiment = "Positive"
        confidence = 1 - prediction # Confidence in it being Positive

    # 4. Return JSON response
    return {
        "sentiment": sentiment, 
        "confidence": confidence
    }