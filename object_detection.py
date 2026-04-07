import cv2
from ultralytics import YOLO
import config
from utils import draw_label


class ObjectDetector:
    """Wraps a YOLOv8n model for targeted object detection."""

    def __init__(self):
        print("[ObjectDetector] Loading YOLOv8n model …")
        self.model = YOLO(config.YOLO_MODEL)
        print("[ObjectDetector] Model loaded successfully.")

    def detect(self, frame):
        """
        Run inference on *frame* (BGR, any size).
        Returns a copy of *frame* with bounding boxes drawn for target classes.
        """
        results = self.model(
            frame,
            imgsz=config.YOLO_IMG_SIZE,
            conf=config.YOLO_CONFIDENCE,
            iou=config.YOLO_IOU,
            verbose=False,
        )

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id not in config.TARGET_CLASSES:
                    continue

                cls_name = config.TARGET_CLASSES[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                color = config.BBOX_COLORS.get(cls_name, (255, 255, 255))

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Label
                label = f"{cls_name} {conf:.0%}"
                draw_label(frame, label, (x1, y1 - 2), color)

        return frame
