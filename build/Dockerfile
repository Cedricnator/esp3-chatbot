FROM python:3.11-slim

ENV TZ=America/Santiago

WORKDIR /app

# Copiar archivos de dependencias
COPY requirements.txt .
COPY pyproject.toml .

# Instalar dependencias
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    rm -rf ~/.cache/pip

# Copiar código fuente
COPY src/ ./src/
COPY data/ ./data/
COPY run_api.py .
COPY .env.example .env

# Crear directorios necesarios
RUN mkdir -p logs

# Exponer puerto de la API
EXPOSE 33201

# Ejecutar la API
CMD ["python", "run_api.py"]
