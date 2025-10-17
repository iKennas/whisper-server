# Python Flask Whisper Server Dockerfile
# More reliable for cloud deployment

FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV WHISPER_MODEL=base
ENV WHISPER_PORT=9000
ENV WHISPER_LANGUAGE=ar

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python server
COPY whisper_server_simple.py .

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting Simple Whisper server..."\n\
echo "Model: ${WHISPER_MODEL}"\n\
echo "Port: ${WHISPER_PORT}"\n\
echo "Language: ${WHISPER_LANGUAGE}"\n\
echo "API Endpoint: http://0.0.0.0:${WHISPER_PORT}/inference"\n\
echo ""\n\
python whisper_server_simple.py' > start.sh

RUN chmod +x start.sh

# Expose port
EXPOSE ${WHISPER_PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${WHISPER_PORT}/health || exit 1

# Start the server
CMD ["./start.sh"]