"""Lab 6 - edge detection: Sobel, Canny, Laplacian-of-Gaussian."""
import cv2
import numpy as np


def preprocess(gray, denoise=True, use_clahe=True):
    """Median blur removes speckle noise; CLAHE boosts local contrast."""
    out = gray
    if denoise:
        out = cv2.medianBlur(out, 5)
    if use_clahe:
        out = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(out)
    return out


def sobel_magnitude(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)


def sobel_edges(gray, threshold):
    return (sobel_magnitude(gray) >= threshold).astype(np.uint8) * 255


def canny_edges(gray, low, high):
    return cv2.Canny(gray, low, high, L2gradient=True)


def log_magnitude(gray, sigma=1.4, blur_ksize=5, lap_ksize=3):
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), sigma)
    return np.abs(cv2.Laplacian(blurred, cv2.CV_64F, ksize=lap_ksize))


def log_edges(gray, threshold):
    return (log_magnitude(gray) >= threshold).astype(np.uint8) * 255


def edge_pixel_count(binary_edges):
    return int(np.count_nonzero(binary_edges))


def run_all_methods(gray, denoise, use_clahe, canny_low, canny_high, sobel_thresh, log_thresh):
    """Run the three detectors on the same preprocessed image."""
    pre = preprocess(gray, denoise, use_clahe)
    results = {
        "Sobel": sobel_edges(pre, sobel_thresh),
        "Canny": canny_edges(pre, canny_low, canny_high),
        "LoG": log_edges(pre, log_thresh),
    }
    counts = {name: edge_pixel_count(e) for name, e in results.items()}
    return pre, results, counts
