FROM python:3.11-slim

ENV TZ=America/Santiago

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    rm -rf ~/.cache/pip

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "src/streamlit/streamlit.py"]
