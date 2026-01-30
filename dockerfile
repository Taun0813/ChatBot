FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
  && rm -rf /var/lib/apt/lists/*

# copy requirements trước để tận dụng cache layer
COPY requirements*.txt ./

RUN pip install --upgrade pip \
 && (test -f requirements.txt && pip install -r requirements.txt || true) \
 && (test -f requirements-dev.txt && pip install -r requirements-dev.txt || true)

COPY . .

EXPOSE 8000
CMD ["gunicorn", "app:app", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120"]
