# V2 image: API (uvicorn) and collector (scrape -> forecast -> daily retrain).
# Code only; data/ and models_v2/ are bind-mounted (see docker-compose.yml, profile "server").
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 IN_DOCKER=1 PYTHON=python TZ=UTC
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-v2.txt .
RUN pip install -r requirements-v2.txt
COPY scraper/config.py scraper/
COPY scripts/v2/ scripts/v2/
RUN useradd -m app && mkdir -p data models_v2 && chown -R app /app
USER app
WORKDIR /app/scripts/v2
EXPOSE 8002
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8002"]
