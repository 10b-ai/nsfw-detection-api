# Use Python slim image with CPU optimizations
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MKL_NUM_THREADS=8
ENV OMP_NUM_THREADS=8
ENV OPENBLAS_NUM_THREADS=8
ENV NUMEXPR_NUM_THREADS=8
ENV TORCH_NUM_THREADS=8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./app ./app

# Run with optimized settings for CPU
CMD ["hypercorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]