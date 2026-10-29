"""Lab 8 - feature detection and description: SIFT, ORB, HOG, matching."""
import cv2
import numpy as np

RATIO_THRESH = 0.75


# ---------------------------------------------------------------- local features
def detect_sift(gray, mask=None):
    return cv2.SIFT_create().detectAndCompute(gray, None if mask is None else mask.astype(np.uint8) * 255)


def detect_orb(gray, nfeatures=500, mask=None):
    return cv2.ORB_create(nfeatures=nfeatures).detectAndCompute(gray, None if mask is None else mask.astype(np.uint8) * 255)


def rotate_image(gray, angle_deg):
    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(gray, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def ratio_test_matches(desc1, desc2, norm_type=cv2.NORM_L2, ratio=RATIO_THRESH):
    """Lowe's ratio test: keep a match only if it is clearly better than the 2nd best."""
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []
    good = []
    for pair in cv2.BFMatcher(norm_type).knnMatch(desc1, desc2, k=2):
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            good.append(pair[0])
    return good


def geometric_inliers(kp1_pts, kp2_pts, matches, reproj=6.0):
    """Keep only matches that agree on one rotation+scale+shift (RANSAC). Returns inlier count."""
    if len(matches) < 3:
        return 0
    src = np.float32([kp1_pts[m.queryIdx] for m in matches])
    dst = np.float32([kp2_pts[m.trainIdx] for m in matches])
    _, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=reproj)
    return 0 if inliers is None else int(inliers.sum())


# ---------------------------------------------------------------- HOG (global feature)
def compute_hog(gray, cell_size=8, nbins=9):
    """Histogram of Oriented Gradients, one normalised histogram per cell."""
    h, w = gray.shape
    hc, wc = max(1, h // cell_size), max(1, w // cell_size)
    crop = gray[:hc * cell_size, :wc * cell_size].astype(np.float32)
    gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle = angle % 180.0
    bin_idx = np.minimum((angle / (180.0 / nbins)).astype(np.int32), nbins - 1)

    cell_hist = np.zeros((hc, wc, nbins), dtype=np.float32)
    for b in range(nbins):
        weighted = np.where(bin_idx == b, mag, 0.0)
        cell_hist[:, :, b] = weighted.reshape(hc, cell_size, wc, cell_size).sum(axis=(1, 3))
    norms = np.linalg.norm(cell_hist, axis=2, keepdims=True)
    return np.where(norms > 1e-6, cell_hist / np.maximum(norms, 1e-6), 0).astype(np.float32)


def hog_vector(gray, cell_size=16, nbins=9):
    v = compute_hog(gray, cell_size, nbins).flatten()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def hog_visualization(gray, cell_hist, cell_size=8, nbins=9):
    """Draw a little line for each strong gradient direction in each cell."""
    hc, wc, _ = cell_hist.shape
    vis = cv2.cvtColor(gray[:hc * cell_size, :wc * cell_size], cv2.COLOR_GRAY2BGR)
    max_len = cell_size * 0.5
    for cy in range(hc):
        for cx in range(wc):
            x0, y0 = int(cx * cell_size + cell_size / 2), int(cy * cell_size + cell_size / 2)
            hist = cell_hist[cy, cx]
            peak = float(hist.max())
            if peak < 1e-6:
                continue
            for b in range(nbins):
                strength = float(hist[b]) / peak
                if strength < 0.15:
                    continue
                ang = np.deg2rad(b * (180.0 / nbins) + (90.0 / nbins) + 90.0)
                dx, dy = max_len * strength * np.cos(ang), max_len * strength * np.sin(ang)
                cv2.line(vis, (int(x0 - dx), int(y0 - dy)), (int(x0 + dx), int(y0 + dy)), (0, 255, 255), 1, cv2.LINE_AA)
    return vis


def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0
