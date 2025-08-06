# Use Python base image with platform compatibility
FROM --platform=linux/amd64 python:3.11-slim

# Set environment vars
ENV PYTHONUNBUFFERED=1 \
    SDL_AUDIODRIVER=dsp \
    SDL_VIDEODRIVER=dummy

# Install system dependencies for audio and pygame
RUN apt-get update && apt-get install -y \
    libasound2-dev \
    libportaudio2 \
    libsndfile1 \
    ffmpeg \
    libsdl2-mixer-2.0-0 \
    libsdl2-2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip 
# && pip install --no-cache-dir -r requirements.txt
#RUN pip install --no-cache-dir -r requirements.txt

# Expose port if needed (optional)
# EXPOSE 8000

# Run the assistant
#CMD ["python", "app/main.py"]
CMD ["python", "websocketV2.py"]
