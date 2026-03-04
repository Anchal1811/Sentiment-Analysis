import mlflow
import mlflow.tensorflow
from sklearn.model_selection import train_test_split
from .processor import SwiggyPreprocessor
from .architecture import build_model
import os

# Ensure models directory exists
if not os.path.exists('models'): os.makedirs('models')

def run_train():
    prep = SwiggyPreprocessor()
    X, y = prep.prepare_data('data/swiggy.csv')
    
    # Split into 80% training and 20% test sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Use MLflow for experiment tracking
    mlflow.set_experiment("Swiggy_Sentiment_Analysis")
    with mlflow.start_run():
        model = build_model()
        
        # Training for 5 epochs
        history = model.fit(
            X_train, y_train, 
            epochs=5, 
            batch_size=32, 
            validation_split=0.1
        )
        
        # Evaluate performance on test data
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        mlflow.log_metric("accuracy", acc)
        
        model.save('models/swiggy_rnn.keras')
        mlflow.tensorflow.log_model(model, "model")
        print(f"Test accuracy: {acc:.2f}")

if __name__ == "__main__":
    run_train()