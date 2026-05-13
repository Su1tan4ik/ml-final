FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY . .

EXPOSE 8000

# PORT is set by Render/Railway; fallback to 8000 locally
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
