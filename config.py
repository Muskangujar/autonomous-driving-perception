"""
Configuration settings for the Autonomous Driving Perception Pipeline.
"""

# ─── Frame Processing ───────────────────────────────────────────────
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
SKIP_FRAMES = 0  # 0 = process every frame, 1 = skip every other, etc.

# ─── Lane Detection ─────────────────────────────────────────────────
GAUSSIAN_KERNEL = (5, 5)
CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_RHO = 2
HOUGH_THETA_DIVIDER = 180  # np.pi / this value
HOUGH_THRESHOLD = 100
HOUGH_MIN_LINE_LENGTH = 40
HOUGH_MAX_LINE_GAP = 25
LANE_COLOR = (0, 255, 0)  # Green
LANE_THICKNESS = 5

# ─── Object Detection (YOLOv8) ──────────────────────────────────────
YOLO_MODEL = "yolov8n.pt"
YOLO_CONFIDENCE = 0.4
YOLO_IOU = 0.45
YOLO_IMG_SIZE = 640

# COCO class IDs for target objects
TARGET_CLASSES = {
    0: "person",
    2: "car",
    9: "traffic light",
}

# Bounding box colors (BGR)
BBOX_COLORS = {
    "person": (0, 0, 255),       # Red
    "car": (255, 165, 0),        # Orange
    "traffic light": (0, 255, 255),  # Yellow
}

# ─── Display ─────────────────────────────────────────────────────────
FPS_COLOR = (0, 255, 255)
FPS_FONT_SCALE = 0.8
FPS_THICKNESS = 2
WINDOW_NAME = "Autonomous Driving Perception Pipeline"
