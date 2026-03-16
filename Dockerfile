FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY chainlink_memory/ chainlink_memory/
COPY api.py ./

RUN pip install --no-cache-dir ".[server]"

# Pre-download the embedding model at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
