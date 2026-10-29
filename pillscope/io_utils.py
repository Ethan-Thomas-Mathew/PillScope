"""Small file helpers."""
from pathlib import Path

import cv2
import numpy as np

MAX_SIDE = 1200      # bigger photos are shrunk so every tool stays fast


def imread(path, max_side=None):
    """Read an image as BGR. Works with non-English folder names on Windows. Returns None on failure."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        return None
    if img is not None and max_side:
        h, w = img.shape[:2]
        if max(h, w) > max_side:
            f = max_side / float(max(h, w))
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return img


def imwrite(path, img):
    """Write an image (format chosen from the extension). Works with non-English folder names."""
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"Could not encode {path}")
    buf.tofile(str(path))
