# Serves the saved used-car price model over HTTP (src.api).
#
# Build:  docker build -t used-car-price .
# Run:    docker run --rm -p 8000:8000 used-car-price
# Docs:   http://127.0.0.1:8000/docs
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install the API dependency set first so source changes do not invalidate it.
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# The estimator unpickles into src.predict's default path: models/04_best_model.joblib
COPY src/ ./src/
COPY models/04_best_model.joblib ./models/04_best_model.joblib

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import json, urllib.request; d = json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)); raise SystemExit(0 if d['model'] != 'missing' else 1)"

# Bind on all interfaces so the published port is reachable from the host.
CMD ["python", "-m", "src.api", "--host", "0.0.0.0", "--port", "8000"]
