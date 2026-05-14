FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY . .

EXPOSE 7860

# HF Spaces uses 7860, Render/Railway set $PORT
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}
