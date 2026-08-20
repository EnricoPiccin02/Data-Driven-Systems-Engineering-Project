FROM python:3.11-slim

WORKDIR /app

# LightGBM requires the GNU OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
    
# System dependencies are kept minimal on purpose
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

# Default command runs the complete data-pipeline end-to-end
CMD ["python3", "scripts/run_pipeline.py"]
