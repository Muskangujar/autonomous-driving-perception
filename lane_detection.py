"""
Lane detection module using classical computer vision (OpenCV).

Pipeline:
  1. Convert to grayscale
  2. Gaussian blur
  3. Canny edge detection
  4. Region-of-interest masking (triangular)
  5. Hough line transform
  6. Average & extrapolate left / right lanes
  7. Draw lane overlay
"""

import cv2
import numpy as np
import config


# ──────────────────────────────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────────────────────────────

def _region_of_interest(edges, frame_shape):
    """Apply a triangular ROI mask to keep only the road area."""
    h, w = frame_shape[:2]
    mask = np.zeros_like(edges)

    # Triangle: bottom-left, top-center, bottom-right
    polygon = np.array([[
        (0, h),
        (w // 2, int(h * 0.55)),
        (w, h),
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(edges, mask)


def _average_slope_intercept(lines, frame_shape):
    """Separate lines into left / right lanes and average them."""
    h, w = frame_shape[:2]
    left_fit, right_fit = [], []

    if lines is None:
        return None, None

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x1 == x2:
            continue
        params = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = params[0], params[1]

        if slope < -0.5:
            left_fit.append((slope, intercept))
        elif slope > 0.5:
            right_fit.append((slope, intercept))

    left_line = _make_line(frame_shape, left_fit)
    right_line = _make_line(frame_shape, right_fit)
    return left_line, right_line


def _make_line(frame_shape, fits):
    """Convert averaged slope/intercept into pixel coordinates."""
    if not fits:
        return None
    h = frame_shape[0]
    avg = np.average(fits, axis=0)
    slope, intercept = avg
    if abs(slope) < 1e-6:
        return None
    y1 = h                       # bottom of frame
    y2 = int(h * 0.6)           # extend towards horizon
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def _draw_lanes(frame, left, right):
    """Overlay semi-transparent lane lines on the frame."""
    overlay = np.zeros_like(frame)
    for line in (left, right):
        if line is not None:
            x1, y1, x2, y2 = line
            cv2.line(overlay, (x1, y1), (x2, y2),
                     config.LANE_COLOR, config.LANE_THICKNESS)
    return cv2.addWeighted(frame, 1.0, overlay, 0.8, 0)


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def detect_lanes(frame):
    """
    Run the full lane-detection pipeline on *frame* (BGR).
    Returns a copy of *frame* with lane overlays drawn.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, config.GAUSSIAN_KERNEL, 0)
    edges = cv2.Canny(blur, config.CANNY_LOW, config.CANNY_HIGH)
    roi = _region_of_interest(edges, frame.shape)

    lines = cv2.HoughLinesP(
        roi,
        rho=config.HOUGH_RHO,
        theta=np.pi / config.HOUGH_THETA_DIVIDER,
        threshold=config.HOUGH_THRESHOLD,
        minLineLength=config.HOUGH_MIN_LINE_LENGTH,
        maxLineGap=config.HOUGH_MAX_LINE_GAP,
    )

    left, right = _average_slope_intercept(lines, frame.shape)
    result = _draw_lanes(frame, left, right)
    return result
