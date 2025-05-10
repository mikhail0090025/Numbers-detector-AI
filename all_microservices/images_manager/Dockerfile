FROM python:3.9-slim
WORKDIR /app
COPY all_microservices/images_manager/ /app/
COPY requirements.txt /app/
RUN apt-get update && apt-get install -y curl && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge
EXPOSE 5000
CMD ["uvicorn", "endpoints:app", "--host", "0.0.0.0", "--port", "5000"]