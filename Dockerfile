# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=7860

# Install system dependencies needed for OpenCV and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirement.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Pre-download YOLOv8n and ResNet-18 weights so startup and first runs are fast
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
RUN python -c "import torchvision.models as models; models.resnet18(weights='DEFAULT')"

# Copy the rest of the application code
COPY . .

# Create necessary local directories with open permissions
RUN mkdir -p uploads detected_objects/all detected_objects/unique detected_objects/clustered && \
    chmod -R 777 uploads detected_objects

# Expose port 7860 (Hugging Face Spaces default port)
EXPOSE 7860

# Command to run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "120", "app:app"]
