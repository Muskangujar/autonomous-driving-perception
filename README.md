# 🚗 Autonomous Driving Perception Pipeline

Real-time perception system combining **lane detection** (classical OpenCV) and **object detection** (YOLOv8 nano) into a single video stream — optimised for CPU-only laptops.

---

## Features

| Module | Technique | Details |
|--------|-----------|---------|
| **Lane Detection** | OpenCV | Grayscale → Gaussian blur → Canny → ROI mask → Hough lines → average left/right |
| **Object Detection** | YOLOv8n | Detects **person**, **car**, **traffic light** with bounding boxes + confidence |
| **Integration** | `main.py` | Combines both pipelines, overlays FPS, supports webcam & video files |

## Quick Start

```bash
# 1 — Install dependencies
pip install -r requirements.txt

# 2 — Run with webcam
python main.py

# 3 — Run with a video file
python main.py --source path/to/video.mp4

# 4 — Quit
#   Press 'q' in the video window
```

## Project Structure

```
├── main.py              # Entry point — ties everything together
├── lane_detection.py    # Classical lane detection (OpenCV)
├── object_detection.py  # YOLOv8n wrapper
├── config.py            # All tuneable parameters
├── utils.py             # FPS counter, drawing helpers
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Configuration

All parameters live in `config.py`:

- **Frame size**: `FRAME_WIDTH`, `FRAME_HEIGHT`
- **Lane detection**: Canny thresholds, Hough params, ROI shape
- **Object detection**: model path, confidence, IoU, target classes
- **Display**: FPS font, colours, window name

## Performance Tips

- Lower `FRAME_WIDTH` / `FRAME_HEIGHT` for faster processing.
- Increase `SKIP_FRAMES` to drop intermediate frames.
- Raise `YOLO_CONFIDENCE` to reduce false positives and speed up NMS.

## Requirements

- Python 3.9+
- 16 GB RAM recommended
- Works on CPU (no GPU needed)

## License

MIT
