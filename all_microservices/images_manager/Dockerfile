FROM python:3.9-slim AS builder
WORKDIR /app
COPY requirements.txt /app/
RUN apt-get update && apt-get install -y curl && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

FROM python:3.9-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY all_microservices/images_manager/ /app/
RUN apt-get update && apt-get install -y curl

USER root

ENV PORT=5000

EXPOSE $PORT
HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/ || exit 1

CMD uvicorn endpoints:app --host 0.0.0.0 --port $PORT