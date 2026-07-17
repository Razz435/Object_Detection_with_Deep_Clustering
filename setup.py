#!/usr/bin/env python3
import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def create_directories():
    """Create necessary directories"""
    print("Creating directories...")
    directories = [
        'uploads',
        'detected_objects/all',
        'detected_objects/unique',
        'detected_objects/clustered',
        'static/css',
        'static/js',
        'templates',
        'models',
        'utils'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}")

def download_yolo_model():
    """Download YOLO model if not exists"""
    print("Checking for YOLO model...")
    if not os.path.exists('yolov8n.pt'):
        print("Downloading YOLOv8n model...")
        subprocess.check_call([
            sys.executable, "-c", 
            "from ultralytics import YOLO; YOLO('yolov8n.pt')"
        ])
        print("Model downloaded successfully!")
    else:
        print("YOLO model already exists.")

def create_init_files():
    """Create __init__.py files for packages"""
    init_files = ['models/__init__.py', 'utils/__init__.py']
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write('# Package initialization\n')
        print(f"Created: {init_file}")

def main():
    print("="*60)
    print("Setting up Dual-Box Object Detection System with Spectral Clustering")
    print("="*60)
    
    create_directories()
    create_init_files()
    install_requirements()
    download_yolo_model()
    
    print("\n" + "="*60)
    print("Setup completed successfully!")
    print("\nTo start the application:")
    print("1. python app.py")
    print("2. Open http://localhost:5000 in your browser")
    print("="*60)

if __name__ == "__main__":
    main()