FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Crear directorios necesarios en el contenedor
RUN mkdir -p logs database models/saved backtesting/results

# Puerto del dashboard Streamlit
EXPOSE 8501

# Variables de entorno para Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# El comando se define en docker-compose.yml por servicio
CMD ["python", "scheduler/main.py"]
