import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from flask_cors import CORS
import torch
# Force PyTorch to use a single thread to save memory on CPU
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from ultralytics import YOLO
import threading
import time

from config import Config
from models.object_database import ObjectDatabase
from utils.helpers import allowed_file, resize_frame, draw_detection, crop_object, create_blank_frame, encode_frame_to_jpeg

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Initialize directories
Config.init_directories()

# ── YOLO model state ─────────────────────────────────────────────────
AVAILABLE_MODELS = {
    'yolov8n.pt': 'YOLOv8 Nano  (fastest, least accurate)',
    'yolov8s.pt': 'YOLOv8 Small (balanced)',
    'yolov8m.pt': 'YOLOv8 Medium (more accurate)',
    'yolov8l.pt': 'YOLOv8 Large (high accuracy)',
    'yolov8x.pt': 'YOLOv8 XLarge (most accurate, slowest)',
}

model             = None
current_model_name = 'yolov8n.pt'
model_loading      = False
model_load_lock    = threading.Lock()

def load_yolo_model(model_name: str) -> tuple:
    """Load (or download) a YOLO model by name. Thread-safe."""
    global model, current_model_name, model_loading
    with model_load_lock:
        model_loading = True
        try:
            print(f'Loading YOLO model: {model_name} …')
            new_model = YOLO(model_name)
            model = new_model
            current_model_name = model_name
            print(f'Model {model_name} loaded successfully!')
            return True, None
        except Exception as exc:
            print(f'Error loading model {model_name}: {exc}')
            return False, str(exc)
        finally:
            model_loading = False

# Load default model at startup
load_yolo_model('yolov8n.pt')

# Error handlers to ensure JSON is always returned
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'success': False,
        'error': 'File is too large. Maximum allowed size is 2 GB.'
    }), 413

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error occurred.'
    }), 500

# ── Global processing state ────────────────────────────────────────────
current_frame      = None
frame_lock         = threading.Lock()
processing_thread  = None
is_processing      = False
video_path         = None
cap                = None
current_video_file = None

# Initialize database
object_db = ObjectDatabase()

def process_video():
    """Main video processing function"""
    global current_frame, is_processing, video_path, cap
    
    if model is None:
        print("YOLO model not loaded!")
        return
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error opening video source: {video_path}")
            is_processing = False
            return
        
        print(f"Continuous processing started: {video_path}")
        
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        frame_interval = max(1, int(fps / Config.FRAME_PROCESS_INTERVAL))
        
        while is_processing and cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                continue
            
            frame_count += 1
            
            # Process every Nth frame
            if frame_count % frame_interval != 0:
                continue
            
            try:
                # Resize frame to 320 for extremely lightweight processing
                frame_small = resize_frame(frame, 320)
                
                # Run YOLO detection with imgsz=320
                results = model(frame_small, conf=Config.DETECTION_CONFIDENCE, imgsz=320, verbose=False)
                
                display_frame = frame_small.copy()
                
                # Process detections
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes
                        
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            cls = int(box.cls[0].cpu().numpy())
                            label = model.names[cls]
                            
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # Crop and save object
                            cropped = crop_object(frame_small, x1, y1, x2, y2)
                            
                            if cropped is not None and cropped.size > 0:
                                frame_info = {
                                    'frame_number': frame_count,
                                    'timestamp': frame_count / fps,
                                    'video_position': cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                                }
                                
                                object_db.add_to_all_objects(label, cropped, conf, frame_info)
                            
                            # Draw detection on display frame
                            draw_detection(display_frame, x1, y1, x2, y2, label, conf)
                
                with frame_lock:
                    current_frame = display_frame.copy()
                
                # Garbage collect unused memory immediately
                import gc
                gc.collect()
                    
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue
            
            # Small delay to control processing rate
            time.sleep(0.033)
    
    except Exception as e:
        print(f"Error in video processing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if cap:
            cap.release()
        print("Processing stopped")

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global video_path, is_processing, processing_thread, current_video_file
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Stop current processing
        is_processing = False
        if processing_thread and processing_thread.is_alive():
            processing_thread.join(timeout=2)
        
        # Save new video
        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename)
        video_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(video_path)
        current_video_file = filename
        
        # Start processing
        is_processing = True
        processing_thread = threading.Thread(target=process_video)
        processing_thread.daemon = True
        processing_thread.start()
        
        return jsonify({
            'success': True,
            'filename': filename,
            'message': 'Video uploaded and continuous detection started'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if current_frame is not None:
                    frame_bytes = encode_frame_to_jpeg(current_frame)
                    if frame_bytes is None:
                        blank = create_blank_frame('Processing...')
                        frame_bytes = encode_frame_to_jpeg(blank)
                else:
                    blank = create_blank_frame('Upload video to start')
                    frame_bytes = encode_frame_to_jpeg(blank)
            
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/all_objects/<path:filename>')
def serve_all_object(filename):
    return send_from_directory(Config.ALL_OBJECTS_FOLDER, filename)

@app.route('/unique_objects/<path:filename>')
def serve_unique_object(filename):
    return send_from_directory(Config.UNIQUE_OBJECTS_FOLDER, filename)

@app.route('/clustered_objects/<path:filename>')
def serve_clustered_object(filename):
    return send_from_directory(Config.CLUSTERED_OBJECTS_FOLDER, filename)

@app.route('/get_all_objects')
def get_all_objects():
    try:
        objects = object_db.get_all_objects('all')
        for obj in objects:
            if 'web_path' not in obj or not obj['web_path']:
                label_clean = obj['label'].replace(' ', '_').lower()
                obj['web_path'] = f"/all_objects/{label_clean}/{obj['filename']}"
        return jsonify({
            'objects': objects[:200],
            'success': True
        })
    except Exception as e:
        print(f"Error getting all objects: {e}")
        return jsonify({'objects': [], 'success': False, 'error': str(e)})

@app.route('/get_unique_objects')
def get_unique_objects():
    try:
        objects = object_db.get_all_objects('unique')
        for obj in objects:
            if 'web_path' not in obj or not obj['web_path']:
                label_clean = obj['label'].replace(' ', '_').lower()
                obj['web_path'] = f"/unique_objects/{label_clean}/{obj['filename']}"
        return jsonify({
            'objects': objects[:200],
            'success': True
        })
    except Exception as e:
        print(f"Error getting unique objects: {e}")
        return jsonify({'objects': [], 'success': False, 'error': str(e)})

@app.route('/get_clustered_objects')
def get_clustered_objects():
    try:
        objects = object_db.get_all_objects('clustered')
        return jsonify({
            'objects': objects,
            'success': True
        })
    except Exception as e:
        print(f"Error getting clustered objects: {e}")
        return jsonify({'objects': [], 'success': False, 'error': str(e)})

@app.route('/get_statistics')
def get_statistics():
    try:
        stats = object_db.get_statistics()
        return jsonify({'stats': stats, 'success': True})
    except Exception as e:
        print(f"Error getting statistics: {e}")
        return jsonify({'stats': {}, 'success': False, 'error': str(e)})

@app.route('/clear_database', methods=['POST'])
def clear_database():
    try:
        data = request.json
        folder_type = data.get('type', 'all')
        result = object_db.clear_database(folder_type)
        return jsonify(result)
    except Exception as e:
        print(f"Error clearing database: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/start_webcam')
def start_webcam():
    global video_path, is_processing, processing_thread
    
    try:
        # ── Pre-flight: probe that the webcam is actually accessible ──
        probe = cv2.VideoCapture(0)
        if not probe.isOpened():
            probe.release()
            return jsonify({
                'success': False,
                'error': 'No webcam found (cv2.VideoCapture(0) failed). '
                         'Make sure a camera is connected and not in use by another app.'
            }), 400
        probe.release()

        # ── Stop any current processing ──
        is_processing = False
        if processing_thread and processing_thread.is_alive():
            processing_thread.join(timeout=2)
        
        video_path = 0
        is_processing = True
        processing_thread = threading.Thread(target=process_video)
        processing_thread.daemon = True
        processing_thread.start()
        
        return jsonify({'success': True, 'message': 'Webcam started with continuous detection'})
    except Exception as e:
        print(f"Error starting webcam: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/stop_processing')
def stop_processing():
    global is_processing
    is_processing = False
    return jsonify({'success': True, 'message': 'Processing stopped'})

@app.route('/discard_duplicates', methods=['POST'])
def discard_duplicates():
    try:
        data = request.json
        delete_original = data.get('delete_original', False)
        result = object_db.process_discard_duplicates(delete_original)
        
        if result['success']:
            return jsonify({
                'success': True,
                'message': f"Found {result['stats']['unique_found']} unique objects",
                'result': result
            })
        else:
            return jsonify({'success': False, 'error': result.get('error', 'Unknown error')}), 500
            
    except Exception as e:
        print(f"Error in discard_duplicates: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/cluster_objects', methods=['POST'])
def cluster_objects():
    try:
        data = request.json
        n_clusters = data.get('n_clusters')
        use_optimal = data.get('use_optimal', True)
        
        if n_clusters is not None:
            try:
                n_clusters = int(n_clusters)
                if n_clusters < 2:
                    n_clusters = 2
            except:
                n_clusters = None
        
        print(f"Clustering request: n_clusters={n_clusters}, use_optimal={use_optimal}")
        
        result = object_db.cluster_similar_objects(
            n_clusters=n_clusters,
            use_optimal=use_optimal
        )
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in cluster_objects: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/current_status')
def current_status():
    try:
        stats = object_db.get_statistics()
        return jsonify({
            'is_processing':  is_processing,
            'model_loading':  model_loading,
            'current_model':  current_model_name,
            'video_file':     current_video_file or 'Webcam',
            'all_objects':    stats['all_objects'],
            'unique_objects': stats['unique_objects'],
            'clusters':       stats['clusters'],
        })
    except Exception as exc:
        print(f'Error getting status: {exc}')
        return jsonify({
            'is_processing': is_processing,
            'model_loading': model_loading,
            'current_model': current_model_name,
            'video_file':    current_video_file or 'Webcam',
            'all_objects': 0, 'unique_objects': 0, 'clusters': 0,
        })

@app.route('/current_model')
def get_current_model():
    return jsonify({
        'model':    current_model_name,
        'loading':  model_loading,
        'models':   AVAILABLE_MODELS,
    })

@app.route('/change_model', methods=['POST'])
def change_model():
    global is_processing, processing_thread
    data       = request.json or {}
    model_name = data.get('model', 'yolov8n.pt')

    if model_name not in AVAILABLE_MODELS:
        return jsonify({'success': False,
                        'error': f'Unknown model: {model_name}'}), 400

    # Stop any ongoing detection first
    is_processing = False
    if processing_thread and processing_thread.is_alive():
        processing_thread.join(timeout=3)

    # Load in a background thread so the HTTP response returns immediately
    def _load():
        load_yolo_model(model_name)

    t = threading.Thread(target=_load, daemon=True)
    t.start()

    return jsonify({
        'success': True,
        'message': f'Loading {model_name} in background… detection paused.',
        'model':   model_name,
    })

@app.route('/clear_database', methods=['POST'])
def clear_database_route():
    try:
        data        = request.json or {}
        folder_type = data.get('type', 'all')   # 'all' | 'unique' | 'clustered' | 'everything'
        result      = object_db.clear_database(folder_type)
        return jsonify(result)
    except Exception as exc:
        print(f'Error clearing database: {exc}')
        return jsonify({'success': False, 'error': str(exc)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Dual-Box Object Detection System with Spectral Clustering")
    print("="*60)
    print(f"Upload folder: {Config.UPLOAD_FOLDER}")
    print(f"All objects folder: {Config.ALL_OBJECTS_FOLDER}")
    print(f"Unique objects folder: {Config.UNIQUE_OBJECTS_FOLDER}")
    print(f"Clustered objects folder: {Config.CLUSTERED_OBJECTS_FOLDER}")
    print(f"Access at: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)