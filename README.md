# 📊 Swiggy Sentiment Analysis Dashboard
application designed for restaurant managers to analyze customer reviews in real-time. This system leverages a **Bi-directional Long Short-Term Memory (Bi-LSTM)** network—a specialized type of **Recurrent Neural Network (RNN)**—to interpret customer emotions and suggest business interventions.

---

## 🚀 Key Features
* **Deep Learning Inference:** Uses a Bi-LSTM model to capture sequential context in restaurant reviews.
* **Manager Insights:** Provides real-time feedback with color-coded sentiment analysis and "Action Required" warnings.
* **Microservices Architecture:** Decoupled Frontend (Streamlit) and Backend (FastAPI).
* **Containerized Environment:** Fully orchestrated using Docker and Docker Compose for seamless deployment.

---

## 🧠 The Role of RNN (Bi-LSTM)
Standard machine learning models often fail to understand sentence context (e.g., "The food wasn't bad"). This project solves that using an **RNN (Recurrent Neural Network)**:

1.  **Sequence Awareness:** The RNN processes text as a sequence, maintaining a "hidden state" that serves as a memory of previous words.
2.  **Bi-directional Processing:** By reading the review from left-to-right and right-to-left simultaneously, the model captures nuances and "sentiment flips" in complex sentences.
3.  **LSTM Gates:** Long Short-Term Memory units utilize Input, Forget, and Output gates to remember long-term dependencies, making it highly effective for long, detailed reviews.



---

## 🛠️ Tech Stack
* **Modeling:** TensorFlow/Keras (Bi-LSTM RNN)
* **API:** FastAPI (Python)
* **Dashboard:** Streamlit
* **DevOps:** Docker & Docker Compose
* **Preprocessing:** Tokenization & Padding (Keras)

---

## 📂 Project Structure
```text
Sentiment-Analysis/
├── Backend/
│   ├── models/           # swiggy_rnn.keras & tokenizer.pkl
│   ├── src/              # app.py (Inference Logic)
│   └── Dockerfile
├── Frontend/
│   ├── ui.py             # Streamlit Dashboard
│   └── Dockerfile
└── docker-compose.yml    # Orchestration
