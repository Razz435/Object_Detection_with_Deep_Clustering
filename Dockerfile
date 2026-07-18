# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=10000

# Install system dependencies needed for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirement.txt .

# Install light CPU-only PyTorch to fit in 512MB RAM, then install other packages
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirement.txt

# Pre-download YOLOv8n and ResNet-18 weights (lightweight CPU versions)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
RUN python -c "import torchvision.models as models; models.resnet18(weights='DEFAULT')"

# Copy the rest of the application code
COPY . .

# Create necessary local directories with open permissions
RUN mkdir -p uploads detected_objects/all detected_objects/unique detected_objects/clustered && \
    chmod -R 777 uploads detected_objects

# Expose port 10000 (Render default port)
EXPOSE 10000

# Command to run the application using Gunicorn (single-threaded to save RAM)
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "--threads", "1", "--timeout", "120", "app:app"]
