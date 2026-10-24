# Ankan web app: CPU-only, with the handwriting model built into the image so the first
# upload does not have to download 1.3 GB.
#
#   docker build -t ankan .
#   docker run -p 8080:8080 -e ANKAN_PASSWORD=choose-one ankan
#
# Build without the model (smaller image; it downloads when the server starts instead):
#   docker build --build-arg PRELOAD_MODELS=false -t ankan .
# Include the embedding or LLM graders:
#   docker build --build-arg EXTRAS=trocr,embeddings,llm -t ankan .

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/models \
    ANKAN_DATA_DIR=/app/data

WORKDIR /app

# CPU build of PyTorch first: the default build bundles GPU libraries (several GB) that a
# web server without a GPU never uses.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
ARG EXTRAS=trocr
RUN pip install ".[${EXTRAS}]"

ARG PRELOAD_MODELS=true
ARG TROCR_MODEL=microsoft/trocr-base-handwritten
ENV ANKAN_TROCR_MODEL=${TROCR_MODEL}
RUN if [ "$PRELOAD_MODELS" = "true" ]; then \
      python -c "from transformers import TrOCRProcessor, VisionEncoderDecoderModel; \
TrOCRProcessor.from_pretrained('${TROCR_MODEL}'); \
VisionEncoderDecoderModel.from_pretrained('${TROCR_MODEL}')"; \
    fi

# Run as an unprivileged user that owns only the data folder.
RUN useradd --create-home --uid 1000 ankan \
    && mkdir -p /app/data /opt/models \
    && chown -R ankan /app/data /opt/models
USER ankan

# Cloud Run, Railway and similar hosts tell the app which port to use through $PORT.
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "exec ankan serve --host 0.0.0.0 --port \"$PORT\""]
