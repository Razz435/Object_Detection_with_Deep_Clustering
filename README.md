# Object Detection with Deep Clustering

A Flask-based real-time object detection and visual clustering system built with YOLOv8, OpenCV, and NumPy. The app can process webcam feeds or uploaded videos, detect objects frame by frame, save cropped detections, and organize them into all, unique, and clustered object groups.

## Features

- Real-time object detection with YOLOv8
- Webcam and video upload support
- Continuous frame processing in a Flask web app
- Crops and stores detected objects for later review
- Groups objects into:
  - all detections
  - unique detections
  - clustered detections
- Duplicate removal / unique object extraction
- Similar object clustering with configurable cluster counts
- Live MJPEG video preview in the browser
- REST endpoints for status, model switching, database management, and object retrieval

## Tech Stack

- Python
- Flask
- Flask-CORS
- OpenCV
- NumPy
- Ultralytics YOLOv8

## Requirements

Install the dependencies listed in `requirement.txt`:

- flask
- flask-cors
- opencv-python
- numpy
- ultralytics

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Razz435/Object_Detection_with_Deep_Clustering.git
cd Object_Detection_with_Deep_Clustering
```

2. Install dependencies:

```bash
pip install -r requirement.txt
```

3. Run the app:

```bash
python app.py
```

The app will start on `http://localhost:5000`.

## Setup Notes

The project initializes these folders automatically when the app starts:

- `uploads/`
- `detected_objects/all/`
- `detected_objects/unique/`
- `detected_objects/clustered/`

The app also loads `yolov8n.pt` by default and supports model switching at runtime.

## Usage

### 1. Open the web app

Visit `http://localhost:5000` in your browser.

### 2. Start detection

- Upload a video using the `/upload_video` endpoint, or
- Start webcam processing using the `/start_webcam` endpoint

### 3. View output

The app streams processed frames through `/video_feed` and stores detected objects for inspection.

### 4. Manage detections

Available endpoints include:

- `/get_all_objects`
- `/get_unique_objects`
- `/get_clustered_objects`
- `/get_statistics`
- `/discard_duplicates`
- `/cluster_objects`
- `/clear_database`
- `/current_status`
- `/current_model`
- `/change_model`
- `/stop_processing`

## Model Support

Available YOLO models include:

- `yolov8n.pt` — fastest, least accurate
- `yolov8s.pt` — balanced
- `yolov8m.pt` — more accurate
- `yolov8l.pt` — high accuracy
- `yolov8x.pt` — most accurate, slowest

## Project Structure

```text
Object_Detection_with_Deep_Clustering/
├── app.py
├── config.py
├── models/
├── utils/
├── templates/
├── uploads/
├── detected_objects/
│   ├── all/
│   ├── unique/
│   └── clustered/
├── requirement.txt
└── README.md
```

## Notes

- `setup.py` in this repository is a setup helper that creates directories, installs dependencies, and checks for the YOLO model.
- The repository uses `requirement.txt` rather than `requirements.txt`.
- The app is currently configured for a Flask web interface rather than a standalone CLI detector class.

## License

No license file is currently present in the repository.

## Contact

For questions or issues, open a GitHub issue in this repository.
