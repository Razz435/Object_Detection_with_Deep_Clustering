import cv2
import numpy as np

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv'}


def allowed_file(filename: str) -> bool:
    """Check whether the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def resize_frame(frame: np.ndarray, target_width: int = 640) -> np.ndarray:
    """Resize a frame while maintaining aspect ratio."""
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def draw_detection(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                   label: str, conf: float) -> None:
    """Draw a bounding box and label on *frame* in-place."""
    color = (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.2f}"
    font_scale = 0.6
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(frame, (x1, y1 - text_h - baseline - 4), (x1 + text_w, y1), color, -1)
    cv2.putText(frame, text, (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)


def crop_object(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray | None:
    """Safely crop a detected object region from *frame*."""
    h, w = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def create_blank_frame(message: str = '', width: int = 640, height: int = 480) -> np.ndarray:
    """Create a black frame with an optional centred text message."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    if message:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (200, 200, 200)
        (text_w, text_h), _ = cv2.getTextSize(message, font, font_scale, thickness)
        x = (width - text_w) // 2
        y = (height + text_h) // 2
        cv2.putText(frame, message, (x, y), font, font_scale, color, thickness)
    return frame


def encode_frame_to_jpeg(frame: np.ndarray, quality: int = 80) -> bytes | None:
    """Encode an OpenCV frame to JPEG bytes."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ret, buffer = cv2.imencode('.jpg', frame, encode_params)
    if not ret:
        return None
    return buffer.tobytes()
