# Whisper Server Dockerfile
# Optimized for Arabic speech recognition

FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV WHISPER_MODEL=base
ENV WHISPER_PORT=9000
ENV WHISPER_LANGUAGE=ar

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    make \
    cmake \
    python3 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone whisper.cpp
RUN git clone https://github.com/ggerganov/whisper.cpp .

# Build whisper.cpp
RUN make

# Download Arabic model
RUN bash ./models/download-ggml-model.sh ${WHISPER_MODEL}

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting Whisper server..."\n\
echo "Model: ${WHISPER_MODEL}"\n\
echo "Port: ${WHISPER_PORT}"\n\
echo "Language: ${WHISPER_LANGUAGE}"\n\
echo "API Endpoint: http://0.0.0.0:${WHISPER_PORT}/inference"\n\
echo ""\n\
./build/bin/server --model models/ggml-${WHISPER_MODEL}.bin --host 0.0.0.0 --port ${WHISPER_PORT} --language ${WHISPER_LANGUAGE}' > start.sh

RUN chmod +x start.sh

# Expose port
EXPOSE ${WHISPER_PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${WHISPER_PORT}/health || exit 1

# Start the server
CMD ["./start.sh"]
