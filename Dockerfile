# Use an AMD64 Python base image
FROM --platform=linux/amd64 python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libasound2-dev \
    portaudio19-dev \
    libffi-dev \
    libsndfile1 \
    build-essential \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirementFinal.txt .
RUN pip install --upgrade pip && pip install -r requirementFinal.txt

# Copy your application code
COPY . .

# Expose port for FastAPI if needed
EXPOSE 8000

# Default command — change 'main.py' to your actual entry point if different
CMD ["python", "edgetts.py"]
