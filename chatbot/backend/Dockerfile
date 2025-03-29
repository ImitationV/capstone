FROM python:3.10-slim

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV HF_HOME=/app/cache \
    XDG_CACHE_HOME=/app/cache \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app
COPY . .

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install base packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy==1.23.5 \
    scipy==1.10.1

# Install compatible PyTorch and Transformers versions
RUN pip install --no-cache-dir \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    torchaudio==2.0.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements with specific versions
RUN pip install --no-cache-dir \
    transformers==4.34.1 \
    peft==0.6.2 \
    datasets==2.14.6 \
    sentencepiece==0.1.99 \
    protobuf==3.20.3 \
    scikit-learn==1.3.0 \
    fastapi==0.95.2 \
    uvicorn==0.22.0 \
    accelerate==0.24.1

# Create cache directories
RUN mkdir -p /app/cache /app/finetuned /app/logs && \
    chmod -R 777 /app/cache /app/finetuned /app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import requests; requests.get('http://localhost:7860/health', timeout=3)" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
