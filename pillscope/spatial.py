"""Lab 2 - spatial filtering: correlation vs convolution with 3x3 kernels."""
import cv2
import numpy as np

KERNELS = {
    "Sobel X (vertical edges)":   np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
    "Sobel Y (horizontal edges)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
    "Prewitt X":                  np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32),
    "Emboss":                     np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32),
    "Identity":                   np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32),
    "Box Blur (3x3)":             np.ones((3, 3), dtype=np.float32) / 9.0,
    "Gaussian Blur (3x3)":        np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0,
    "Sharpen":                    np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    "Laplacian":                  np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
}


def is_symmetric_kernel(kernel):
    """For a symmetric kernel, convolution and correlation give the same result."""
    return bool(np.array_equal(kernel, np.flip(kernel)))


def render_filtered(out, kernel):
    """Turn a float filter result into a displayable 8-bit image."""
    if abs(float(kernel.sum())) < 1e-6:          # edge-type kernel: centre values on grey
        peak = float(np.max(np.abs(out)))
        if peak < 1e-9:
            return np.full(out.shape, 128, dtype=np.uint8)
        return np.clip(out / peak * 127.0 + 128.0, 0, 255).astype(np.uint8)
    return np.clip(out, 0, 255).astype(np.uint8)


def correlate2d(gray, kernel, border=cv2.BORDER_REPLICATE):
    """Slide the kernel over the image (no flipping). Returns (float result, 8-bit result)."""
    img = gray.astype(np.float32)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = cv2.copyMakeBorder(img, ph, ph, pw, pw, border)
    h, w = gray.shape
    out = np.zeros((h, w), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            if kernel[i, j] != 0:
                out += kernel[i, j] * padded[i:i + h, j:j + w]
    return out, render_filtered(out, kernel)


def convolve2d(gray, kernel, border=cv2.BORDER_REPLICATE):
    """Convolution = correlation with the kernel flipped in both directions."""
    return correlate2d(gray, np.flip(kernel), border)
