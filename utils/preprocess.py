"""
preprocess.py
-------------
OpenCV-based image preprocessing pipeline:
  - Noise reduction
  - Contrast enhancement (CLAHE)
  - Edge detection (Canny)
  - Damage region segmentation (GrabCut + contours)
  - Feature extraction helpers
"""

import cv2
import numpy as np
from pathlib import Path


# ── Core preprocessing ───────────────────────────────────────────────────────
def load_image(img_path: str) -> np.ndarray:
    """Load image from disk in BGR (OpenCV default)."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {img_path}")
    return img


def resize_image(img: np.ndarray, size: tuple = (224, 224)) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def denoise(img: np.ndarray) -> np.ndarray:
    """Non-local means denoising — preserves edges better than Gaussian."""
    return cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10,
                                           templateWindowSize=7,
                                           searchWindowSize=21)


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE to L-channel of LAB image for local contrast boost."""
    lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def detect_edges(img: np.ndarray) -> np.ndarray:
    """Canny edge detection on grayscale image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, threshold1=50, threshold2=150)


def segment_damage_region(img: np.ndarray) -> tuple:
    """
    Use GrabCut to roughly segment foreground (vehicle + damage).
    Returns (mask, segmented_img).
    """
    h, w    = img.shape[:2]
    rect    = (10, 10, w - 20, h - 20)          # tight rect around vehicle
    mask    = np.zeros((h, w), dtype=np.uint8)
    bgd_mdl = np.zeros((1, 65), dtype=np.float64)
    fgd_mdl = np.zeros((1, 65), dtype=np.float64)

    cv2.grabCut(img, mask, rect, bgd_mdl, fgd_mdl,
                iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                        255, 0).astype(np.uint8)
    segmented = cv2.bitwise_and(img, img, mask=fg_mask)
    return fg_mask, segmented


def extract_damage_contours(edges: np.ndarray,
                             min_area: int = 500) -> list:
    """
    Find contours in the edge map that exceed min_area.
    Returns list of contours sorted by area (largest first).
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    filtered = [c for c in contours if cv2.contourArea(c) > min_area]
    return sorted(filtered, key=cv2.contourArea, reverse=True)


def annotate_damage(img: np.ndarray, contours: list,
                    color: tuple = (0, 0, 255)) -> np.ndarray:
    """Draw bounding boxes around detected damage regions."""
    annotated = img.copy()
    for c in contours[:5]:                        # top-5 largest regions
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
    return annotated


# ── Full pipeline ─────────────────────────────────────────────────────────────
def full_pipeline(img_path: str,
                  save_steps: bool = False,
                  out_dir: str = "outputs") -> dict:
    """
    Run the complete preprocessing pipeline on one image.

    Returns dict with all intermediate results:
        original, denoised, enhanced, edges, segmented,
        annotated, damage_regions (list of bounding boxes)
    """
    img       = load_image(img_path)
    img       = resize_image(img)
    denoised  = denoise(img)
    enhanced  = enhance_contrast(denoised)
    edges     = detect_edges(enhanced)
    _, seg    = segment_damage_region(enhanced)
    contours  = extract_damage_contours(edges)
    annotated = annotate_damage(enhanced, contours)

    damage_boxes = []
    for c in contours[:5]:
        x, y, w, h = cv2.boundingRect(c)
        damage_boxes.append({"x": int(x), "y": int(y),
                              "w": int(w), "h": int(h),
                              "area": int(cv2.contourArea(c))})

    result = {
        "original":      img,
        "denoised":      denoised,
        "enhanced":      enhanced,
        "edges":         edges,
        "segmented":     seg,
        "annotated":     annotated,
        "damage_regions": damage_boxes,
        "n_regions":     len(damage_boxes),
    }

    if save_steps:
        out = Path(out_dir) / "preprocessing"
        out.mkdir(parents=True, exist_ok=True)
        for name, frame in result.items():
            if isinstance(frame, np.ndarray):
                cv2.imwrite(str(out / f"{name}.jpg"), frame)
        print(f"[Preprocess] Steps saved to {out}")

    return result


# ── Feature helpers ───────────────────────────────────────────────────────────
def compute_damage_score(damage_regions: list,
                          img_area: int = 224 * 224) -> float:
    """
    Simple heuristic: ratio of total damage contour area to image area.
    Returns value 0.0 – 1.0.
    """
    total = sum(r["area"] for r in damage_regions)
    return min(total / img_area, 1.0)
