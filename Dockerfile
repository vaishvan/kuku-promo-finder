# Public demo image (Hugging Face Spaces, Render, Fly, any Docker host).
# Contains code, the model registry, gold labels and the fictional sample show. Never the confidential scripts,
# run outputs or .env (see .dockerignore). The API key is injected at runtime as the JUDGE_API_KEY secret.
FROM python:3.12-slim

WORKDIR /app
COPY promofinder/ promofinder/
COPY gold/ gold/
COPY models/ models/
COPY samples/ samples/

# Judge settings that are not secret. JUDGE_API_KEY comes from the host's secret store.
ENV PYTHONUNBUFFERED=1 \
    PROMO_HOSTED=1 \
    PORT=7860 \
    JUDGE_BASE_URL=https://integrate.api.nvidia.com/v1 \
    JUDGE_MODEL=nvidia/nemotron-3.5-super-vl-120b-a12b \
    JUDGE_NO_THINK=1 \
    JUDGE_JSON_MODE=0 \
    JUDGE_ATTEMPTS=4

# Hugging Face runs containers as uid 1000; the app writes uploads and the judge cache under /app/runs.
RUN useradd -m -u 1000 user && mkdir -p runs && chown -R user /app
USER user

EXPOSE 7860
CMD ["sh", "-c", "python -m promofinder serve --host 0.0.0.0 --port ${PORT}"]
