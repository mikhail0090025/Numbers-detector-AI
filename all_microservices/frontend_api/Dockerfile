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
COPY all_microservices/frontend_api/ /app/
# COPY static/ /app/static/
RUN apt-get update && apt-get install -y curl

USER root

EXPOSE 5002
HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:5002/ || exit 1

CMD uvicorn endpoints:app --host 0.0.0.0 --port 5002