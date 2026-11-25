# Base image with system dependencies
FROM python:3.11-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Set Python path
ENV PYTHONPATH=/app/src

# Default command
CMD ["python", "-m", "document_intelligence.api.main"]


# API service
FROM base AS api

EXPOSE 8000

CMD ["uvicorn", "document_intelligence.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]


# Worker service (GPU)
FROM base AS worker

# Install GPU-specific dependencies
RUN pip install --no-cache-dir \
    paddlepaddle-gpu==2.5.2.post122 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html \
    onnxruntime-gpu==1.19.0

CMD ["python", "-m", "document_intelligence.workers.extractor"]