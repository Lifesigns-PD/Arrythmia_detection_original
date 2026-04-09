FROM python:3.13-slim

WORKDIR /app

# System deps for scipy/numpy compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Model checkpoints are baked into image (or use K8s volume mount)
# Verify checkpoints exist:
RUN python -c "from pathlib import Path; \
    ckpt_dir = Path('models_training/outputs/checkpoints'); \
    assert (ckpt_dir / 'best_model_rhythm_v2.pth').exists() or \
           (ckpt_dir / 'best_model_rhythm.pth').exists(), \
    'No rhythm model checkpoint found'; \
    assert (ckpt_dir / 'best_model_ectopy_v2.pth').exists() or \
           (ckpt_dir / 'best_model_ectopy.pth').exists(), \
    'No ectopy model checkpoint found'; \
    print('[OK] Checkpoints verified')"

# Entry point
CMD ["python", "kafka_consumer.py"]
