FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/models
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
# CPU-only torch keeps the image small; drop the extra index for GPU builds.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu torch \
 && pip install ".[ml]"

# Build the index first (make index MODEL=e5-base) and mount it:
#   docker run -p 8000:8000 -v $PWD/artifacts:/app/artifacts -v hf-cache:/models persian-semantic-search
ENV PSS_INDEX_DIR=/app/artifacts/index/e5-base
EXPOSE 8000
CMD ["uvicorn", "pss.api:app", "--host", "0.0.0.0", "--port", "8000"]
