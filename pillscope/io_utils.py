"""Small file helpers."""
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

MAX_SIDE = 1200      # bigger photos are shrunk so every tool stays fast

PathLike = Union[str, Path]
BGRImage = np.ndarray  # cv2's stubs aren't precise enough for a tighter dtype here


def imread(path: PathLike, max_side: Optional[int] = MAX_SIDE) -> Optional[BGRImage]:
    """Read an image as BGR. Works with non-English folder names on Windows. Returns None on failure.
    Shrinks the image if it's bigger than `max_side` on its longest edge (pass max_side=None to skip)."""
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


def imwrite(path: PathLike, img: BGRImage) -> None:
    """Write an image (format chosen from the extension). Works with non-English folder names."""
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise OSError(f"Could not encode {path}")
    buf.tofile(str(path))
