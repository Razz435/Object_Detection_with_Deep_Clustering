import os

class Config:
    SECRET_KEY = 'your_secret_key_here_change_in_production'
    UPLOAD_FOLDER = 'uploads'
    ALL_OBJECTS_FOLDER = 'detected_objects/all'
    UNIQUE_OBJECTS_FOLDER = 'detected_objects/unique'
    CLUSTERED_OBJECTS_FOLDER = 'detected_objects/clustered'
    MAX_CONTENT_LENGTH = 2048 * 1024 * 1024  # 2GB
    
    # Clustering parameters
    DEFAULT_CLUSTER_N_CLUSTERS = 3  # Default number of clusters for Spectral Clustering
    DEFAULT_CLUSTER_AFFINITY = 'rbf'  # Affinity type: 'rbf', 'nearest_neighbors', 'precomputed'
    DEFAULT_CLUSTER_GAMMA = 1.0  # Kernel coefficient for RBF
    
    # Detection parameters
    DETECTION_CONFIDENCE = 0.5
    FRAME_PROCESS_INTERVAL = 3  # Process every 3rd frame
    WEBCAM_FPS = 10
    
    # Similarity thresholds
    UNIQUE_SIMILARITY_THRESHOLD = 0.85  # 85% similarity required for duplicates (slightly lower for better detection)
    
    @staticmethod
    def init_directories():
        """Create necessary directories"""
        directories = [
            Config.UPLOAD_FOLDER,
            Config.ALL_OBJECTS_FOLDER,
            Config.UNIQUE_OBJECTS_FOLDER,
            Config.CLUSTERED_OBJECTS_FOLDER
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")