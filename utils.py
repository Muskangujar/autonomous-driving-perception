"""
Utility functions for the Autonomous Driving Perception Pipeline.
"""

import cv2
import time
import config


class FPSCounter:
    """Tracks and smooths FPS over a rolling window."""

    def __init__(self, window_size: int = 30):
        self._window_size = window_size
        self._times: list[float] = []
        self._prev_time = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        dt = now - self._prev_time
        self._prev_time = now
        self._times.append(dt)
        if len(self._times) > self._window_size:
            self._times.pop(0)
        avg_dt = sum(self._times) / len(self._times)
        return 1.0 / avg_dt if avg_dt > 0 else 0.0


def resize_frame(frame, width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT):
    """Resize a frame to the target resolution."""
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def draw_fps(frame, fps: float):
    """Overlay the FPS counter on the frame."""
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        config.FPS_FONT_SCALE,
        config.FPS_COLOR,
        config.FPS_THICKNESS,
        cv2.LINE_AA,
    )
    return frame


def draw_label(frame, text: str, position: tuple, color: tuple):
    """Draw a text label with a filled background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    cv2.rectangle(frame, (x, y - th - 4), (x + tw + 4, y + 4), color, -1)
    cv2.putText(frame, text, (x + 2, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame
