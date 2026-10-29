"""Lab 3 - sampling (spatial resolution) and quantization (grey levels)."""
import cv2
import numpy as np


def apply_sampling(img, factor):
    """Keep every `factor`-th pixel, then blow it back up to the original size."""
    if factor <= 1:
        return img.copy()
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def apply_quantization(img, levels):
    """Reduce the image to `levels` grey/colour levels per channel."""
    if levels >= 256:
        return img.copy()
    step = 256.0 / levels
    q = np.floor(img.astype(np.float32) / step) * step + step / 2.0
    return np.clip(q, 0, 255).astype(np.uint8)


def detail_score(img):
    """How much detail survives? Count Canny edge pixels."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return int(np.count_nonzero(cv2.Canny(gray, 100, 200)))
