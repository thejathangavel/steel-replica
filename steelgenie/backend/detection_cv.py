import cv2
import numpy as np
from pathlib import Path

def create_column_template(size=40):
    """
    Synthesize the structural column symbol:
    - Outer cross (flanges + web of I-section in plan view)
    - Small center square (column core)
    - Small tick line above
    """
    canvas = np.zeros((size * 3, size * 3), dtype=np.uint8)
    cx, cy = size + size // 2, size + size // 2
    t = max(2, size // 12)   # line thickness

    # Horizontal flange line
    cv2.line(canvas, (cx - size, cy), (cx + size, cy), 255, t * 2)
    # Vertical web line
    cv2.line(canvas, (cx, cy - size), (cx, cy + size), 255, t * 2)
    # Small center square (column core box)
    half = size // 6
    cv2.rectangle(canvas, (cx - half, cy - half), (cx + half, cy + half), 255, t)
    # Small tick line above (as seen in the drawing)
    cv2.line(canvas, (cx, cy - size), (cx, cy - size - size // 3), 255, t)

    return canvas

def match_rotated_scaled(image_gray, template, angle_step=5, scale_range=(0.4, 2.0),
                          scale_steps=20, threshold=0.55):
    """
    Multi-scale + multi-rotation template matching.
    Returns list of (x, y, w, h, angle, score) for each detected column.
    """
    detections = []
    h_t, w_t = template.shape

    scales = np.linspace(scale_range[0], scale_range[1], scale_steps)
    angles = range(0, 180, angle_step)   # I/cross symbol repeats every 90°

    for scale in scales:
        new_w = int(w_t * scale)
        new_h = int(h_t * scale)
        if new_w < 10 or new_h < 10:
            continue
        resized = cv2.resize(template, (new_w, new_h))

        for angle in angles:
            M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle, 1.0)
            rotated = cv2.warpAffine(resized, M, (new_w, new_h))

            # Skip if template larger than image
            if rotated.shape[0] >= image_gray.shape[0] or \
               rotated.shape[1] >= image_gray.shape[1]:
                continue

            result = cv2.matchTemplate(image_gray, rotated, cv2.TM_CCOEFF_NORMED)
            locs = np.where(result >= threshold)

            for pt in zip(*locs[::-1]):   # (x, y)
                detections.append({
                    "x": pt[0], "y": pt[1],
                    "w": new_w, "h": new_h,
                    "angle": angle,
                    "score": float(result[pt[1], pt[0]])
                })

    return detections

def non_max_suppression(detections, overlap_thresh=0.3):
    """Remove duplicate detections using IoU-based NMS."""
    if not detections:
        return []

    boxes = np.array([[d["x"], d["y"], d["x"] + d["w"], d["y"] + d["h"]]
                      for d in detections], dtype=float)
    scores = np.array([d["score"] for d in detections])

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        iou = (w * h) / areas[order[1:]]
        order = order[np.where(iou <= overlap_thresh)[0] + 1]

    return [detections[i] for i in keep]

def detect_columns_by_contour(image_gray, min_area=100, max_area=8000,
                               cross_ratio_tol=0.35):
    """
    Fallback: detect cross-shaped contours directly.
    Works well on clean CAD drawings where lines are crisp.
    """
    blurred = cv2.GaussianBlur(image_gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Close small gaps in lines
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    columns = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h if h > 0 else 0

        # A cross/I-symbol bounding box is roughly square
        if not (1 - cross_ratio_tol < aspect < 1 + cross_ratio_tol):
            continue

        # Check that contour fills ~40-80% of its bounding box (cross shape)
        fill = area / (w * h)
        if not (0.25 < fill < 0.75):
            continue

        columns.append({"x": x, "y": y, "w": w, "h": h,
                        "angle": 0, "score": fill})

    return columns

def detect_column_symbols(image, template_size=40, threshold=0.55):
    """
    Unified function for integration into main pipeline.
    Expects a BGR or Gray image.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    template = create_column_template(template_size)
    raw_detections = match_rotated_scaled(gray, template, threshold=threshold)
    contour_detections = detect_columns_by_contour(gray)

    all_detections = raw_detections + contour_detections
    columns = non_max_suppression(all_detections, overlap_thresh=0.3)
    
    return columns
