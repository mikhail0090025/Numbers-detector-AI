FROM python:3.9-slim
WORKDIR /app
COPY all_microservices/Neural_net/ /app/
COPY requirements.txt /app/
RUN apt-get update && apt-get install -y curl && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge
EXPOSE 5001
CMD ["uvicorn", "endpoints:app", "--host", "0.0.0.0", "--port", "5001"]