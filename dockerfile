FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV TOKENIZERS_PARALLELISM=false

# System deps needed for audio, ffmpeg, building some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ffmpeg libsndfile1 portaudio19-dev libasound2-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel && pip install -r /app/requirements.txt

# Copy project
COPY . /app

# Expose Flask default port (alpha_app2 uses Flask)
EXPOSE 5000

# Default command (adjust if you run via gunicorn or different entry)
CMD ["python", "alpha_app2.py"]
