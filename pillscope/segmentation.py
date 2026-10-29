"""Lab 7 - segmentation: region growing, plus automatic pill extraction."""
from collections import deque

import cv2
import numpy as np

FOUR_NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
EIGHT_NEIGHBORS = FOUR_NEIGHBORS + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


# ---------------------------------------------------------------- region growing (Lab 7)
def region_growing(gray, seed, threshold, connectivity=4):
    """Start at `seed` (row, col) and add neighbours whose grey value is within
    `threshold` of the region's running mean. Returns (mask, pixel count, mean)."""
    h, w = gray.shape
    r0 = int(np.clip(seed[0], 0, h - 1))
    c0 = int(np.clip(seed[1], 0, w - 1))
    gray_f = gray.astype(np.float64)
    offsets = EIGHT_NEIGHBORS if connectivity == 8 else FOUR_NEIGHBORS

    region = np.zeros((h, w), dtype=bool)
    seen = np.zeros((h, w), dtype=bool)
    region[r0, c0] = seen[r0, c0] = True
    total, count = gray_f[r0, c0], 1

    queue = deque([(r0, c0)])
    while queue:
        r, c = queue.popleft()
        mean = total / count
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
                seen[nr, nc] = True
                if abs(gray_f[nr, nc] - mean) <= threshold:
                    region[nr, nc] = True
                    total += gray_f[nr, nc]
                    count += 1
                    queue.append((nr, nc))
    return region, count, total / count


def make_overlay(gray, region, color=(255, 60, 60), alpha=0.45):
    """Paint the region on top of the grey image (returns RGB)."""
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float64)
    out = base.copy()
    out[region] = (1 - alpha) * base[region] + alpha * np.array(color, dtype=np.float64)
    return out.astype(np.uint8)


# ---------------------------------------------------------------- automatic pill mask
def _largest_blob(binary):
    """Keep only the biggest white blob and fill any holes inside it."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(binary)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        cv2.drawContours(out, [biggest], -1, 255, thickness=-1)
    return out


def _clean(binary):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    return _largest_blob(binary)


def _background_color(lab):
    """Median colour of a thin frame around the picture = the background."""
    b = 4
    frame = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                            lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    return np.median(frame, axis=0)


def segment_threshold(img_bgr):
    """Pill = pixels whose colour differs from the background (Otsu picks the cut-off)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    dist = np.sqrt(((lab - _background_color(lab)) ** 2).sum(axis=2))
    dist8 = np.clip(dist * 4, 0, 255).astype(np.uint8)
    cut, binary = cv2.threshold(dist8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cut < 20:                                       # Otsu found nothing meaningful
        binary = (dist8 >= 20).astype(np.uint8) * 255
    return _clean(binary) > 0


def segment_region_growing(img_bgr, threshold=14):
    """Grow the background from the four corners, then flip it: what is left is the pill."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    background = np.zeros((h, w), dtype=bool)
    for seed in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        region, _, _ = region_growing(gray, seed, threshold, connectivity=4)
        background |= region
    return _clean((~background).astype(np.uint8) * 255) > 0


def segment_pill(img_bgr, method="threshold"):
    """Return a boolean mask of the pill. Falls back to 'whole image' if nothing is found."""
    mask = segment_region_growing(img_bgr) if method == "region_growing" else segment_threshold(img_bgr)
    frac = mask.mean()
    if frac < 0.03 or frac > 0.97:          # almost nothing / almost everything = failed
        return np.ones(mask.shape, dtype=bool)
    return mask


# ---------------------------------------------------------------- shape analysis
def describe_shape(mask):
    """Measure the pill outline and name its shape (names match the database)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"area": 0, "perimeter": 0.0, "circularity": 0.0, "aspect": 1.0, "vertices": 0, "shape": "other"}
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    perim = float(cv2.arcLength(c, True))
    hull_perim = float(cv2.arcLength(cv2.convexHull(c), True))      # smooth outline: ignores pixel jaggies
    circ = 4 * np.pi * area / (hull_perim ** 2) if hull_perim > 0 else 0.0
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    aspect = max(rw, rh) / max(min(rw, rh), 1.0)
    vertices = len(cv2.approxPolyDP(c, 0.03 * perim, True))

    if circ > 0.80 and aspect < 1.2:
        shape = "round"
    elif aspect >= 2.2:
        shape = "capsule"
    elif aspect >= 1.6:
        shape = "oblong"
    elif aspect >= 1.2 and circ > 0.70:
        shape = "oval"
    elif vertices == 3:
        shape = "triangle"
    elif vertices == 4:
        shape = "square" if aspect < 1.2 else "rectangle"
    elif vertices == 5:
        shape = "pentagon"
    elif vertices == 6:
        shape = "hexagon"
    else:
        shape = "other"
    return {"area": int(area), "perimeter": round(perim, 1), "circularity": round(float(circ), 3),
            "aspect": round(float(aspect), 2), "vertices": vertices, "shape": shape}


def crop_to_mask(img, mask, size, fill="mean", erode=4):
    """Cut the pill out, make it square, resize to `size`.
    Background pixels are painted with the pill's average colour so they make no keypoints.
    Returns (crop image, crop mask)."""
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop, cmask = img[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1]
    mean_color = crop[cmask].mean(axis=0) if cmask.any() else crop.reshape(-1, 3).mean(axis=0)
    crop[~cmask] = mean_color.astype(np.uint8) if fill == "mean" else 255
    h, w = crop.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    sq = np.empty((side, side, 3), dtype=np.uint8)
    sq[:] = mean_color.astype(np.uint8)
    sq[top:top + h, left:left + w] = crop
    sq_mask = np.zeros((side, side), dtype=np.uint8)
    sq_mask[top:top + h, left:left + w] = cmask.astype(np.uint8) * 255
    sq = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)
    sq_mask = cv2.resize(sq_mask, (size, size), interpolation=cv2.INTER_NEAREST)
    if erode > 0:                                   # stay off the rim of the pill
        sq_mask = cv2.erode(sq_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1,) * 2))
    return sq, sq_mask > 0
