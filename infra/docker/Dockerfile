FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

RUN apt-get update \
    && apt-get install -y --no-install-recommends git poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY config ./config
COPY prompts ./prompts
COPY src ./src
COPY tests ./tests
COPY data/reference ./data/reference
COPY data/raw ./data/raw

RUN pip install --no-cache-dir -e ".[dev,api]"

EXPOSE 8010

CMD ["python", "-m", "adip.api", "--host", "0.0.0.0", "--port", "8010"]
