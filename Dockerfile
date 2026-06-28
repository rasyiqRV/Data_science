# ─────────────────────────────────────────────────────────────────
#  Dockerfile
#  Vertix – Portable container image.
#  Works on Hugging Face Spaces, Fly.io, Railway, Google Cloud Run,
#  Koyeb, Render, and any Docker-capable host.
# ─────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# Prevent Python from writing .pyc files and force unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    PORT=7860

# System deps needed by numpy / scikit-learn wheels are already bundled,
# but build-essential helps if a wheel must compile.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# The instance folder holds the SQLite DB; make sure it exists & is writable
RUN mkdir -p instance && chmod -R 777 instance

# Hugging Face Spaces & most hosts inject $PORT; default to 7860
EXPOSE 7860

# Use shell form so $PORT is expanded at runtime
CMD gunicorn run:app --bind 0.0.0.0:${PORT} --workers 2 --timeout 120
