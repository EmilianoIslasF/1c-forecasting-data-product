FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema (psycopg2 necesita esto)
RUN apt-get update && apt-get install -y \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8501

# Variables de entorno que ECS inyectará en runtime:
# SECRET_ARN  → ARN del secret en Secrets Manager
# RDS_HOST    → Endpoint de RDS (del Output del CloudFormation stack)
ENV SECRET_ARN=""
ENV RDS_HOST=""

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
