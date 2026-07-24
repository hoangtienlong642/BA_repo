FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install using pip cache mount for instant builds
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefer-binary -r requirements.txt

# Copy webapp and app modules
COPY webapp ./webapp
COPY app ./app
COPY data ./data

EXPOSE 12076

ENV PYTHONPATH=/app

CMD ["streamlit", "run", "webapp/app.py", "--server.port=12076", "--server.address=0.0.0.0"]
