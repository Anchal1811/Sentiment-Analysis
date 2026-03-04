# Frontend/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install Streamlit and Requests
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit default port
EXPOSE 8501

# Run the UI
CMD ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0"]