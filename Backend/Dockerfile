FROM python:3.10-slim

WORKDIR /app

# Step 1: Upgrade pip to handle large wheels and hashes better
RUN pip install --no-cache-dir --upgrade pip

# Step 2: Install dependencies with a higher timeout for large files
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8001"]