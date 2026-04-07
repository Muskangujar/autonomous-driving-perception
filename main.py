"""
Autonomous Driving Perception Pipeline — Main Entry Point

Combines lane detection (OpenCV) and object detection (YOLOv8n)
into a single real-time video stream with FPS overlay.

Usage:
    python main.py                  # webcam (device 0)
    python main.py --source video.mp4   # video file
    python main.py --source 1       # webcam device 1
"""

import argparse
import sys
import cv2

import config
from lane_detection import detect_lanes
from object_detection import ObjectDetector
from utils import FPSCounter, resize_frame, draw_fps


def parse_args():
    parser = argparse.ArgumentParser(
        description="Autonomous Driving Perception Pipeline"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: webcam index (0, 1, …) or path to video file.",
    )
    return parser.parse_args()


def open_source(source_str: str) -> cv2.VideoCapture:
    """Open a webcam or video file."""
    # Try to interpret as integer (webcam index)
    try:
        idx = int(source_str)
        cap = cv2.VideoCapture(idx)
    except ValueError:
        cap = cv2.VideoCapture(source_str)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source_str}")
        sys.exit(1)

    print(f"[main] Opened source: {source_str}")
    return cap


def main():
    args = parse_args()

    # ── Initialise components ────────────────────────────────────────
    print("=" * 60)
    print("  Autonomous Driving Perception Pipeline")
    print("=" * 60)

    cap = open_source(args.source)
    detector = ObjectDetector()
    fps_counter = FPSCounter()

    frame_idx = 0
    print("[main] Starting processing loop. Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[main] End of video stream.")
            break

        # Optional frame skipping for performance
        frame_idx += 1
        if config.SKIP_FRAMES > 0 and (frame_idx % (config.SKIP_FRAMES + 1)) != 0:
            continue

        # Resize for faster processing
        frame = resize_frame(frame)

        # ── Lane detection ───────────────────────────────────────────
        frame = detect_lanes(frame)

        # ── Object detection ─────────────────────────────────────────
        frame = detector.detect(frame)

        # ── FPS overlay ──────────────────────────────────────────────
        fps = fps_counter.tick()
        draw_fps(frame, fps)

        # ── Display ──────────────────────────────────────────────────
        cv2.imshow(config.WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[main] 'q' pressed — exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[main] Pipeline shut down cleanly.")


if __name__ == "__main__":
    main()
