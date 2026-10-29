"""Lab 5 - Fourier transform: spectra, ideal low/high-pass filters, texture descriptor."""
import numpy as np


def to_display(arr):
    """Stretch any array to 0-255 so it can be shown as an image."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)


def compute_fshift(gray):
    """2D FFT with the zero frequency moved to the centre."""
    return np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))


def magnitude_spectrum(fshift):
    return to_display(np.log1p(np.abs(fshift)))


def phase_spectrum(fshift):
    return to_display(np.angle(fshift))


def cosine_spectrum(fshift):
    return to_display(np.log1p(np.abs(fshift.real)))


def sine_spectrum(fshift):
    return to_display(np.log1p(np.abs(fshift.imag)))


def reconstruct(fshift):
    """Inverse FFT, stretched to 0-255."""
    img = np.real(np.fft.ifft2(np.fft.ifftshift(fshift)))
    return to_display(img)


def distance_grid(shape):
    rows, cols = shape
    u = (np.arange(rows) - rows // 2).reshape(-1, 1)
    v = (np.arange(cols) - cols // 2).reshape(1, -1)
    return np.sqrt(u ** 2 + v ** 2)


def ideal_lowpass_mask(shape, d0):
    return (distance_grid(shape) <= d0).astype(np.float64)


def ideal_highpass_mask(shape, d0):
    return 1.0 - ideal_lowpass_mask(shape, d0)


def apply_ideal_filter(gray, d0, kind):
    """kind = 'low' (blur) or 'high' (keep edges/detail). Returns (result, mask image)."""
    fshift = compute_fshift(gray)
    mask = ideal_lowpass_mask(gray.shape, d0) if kind == "low" else ideal_highpass_mask(gray.shape, d0)
    return reconstruct(fshift * mask), (mask * 255).astype(np.uint8)


def max_radius(shape):
    rows, cols = shape
    return int(np.hypot(rows / 2, cols / 2)) + 1


def radial_energy(gray, bins=12):
    """Texture descriptor: how much spectrum energy sits at each distance from the centre.
    Because it only depends on distance, rotating the pill does not change it."""
    f = np.abs(compute_fshift(gray.astype(np.float64) - gray.mean()))
    dist = distance_grid(gray.shape)
    edges = np.linspace(0, dist.max() * 0.7, bins + 1)
    profile = np.array([np.log1p(f[(dist >= edges[i]) & (dist < edges[i + 1])].mean())
                        for i in range(bins)], dtype=np.float32)
    norm = np.linalg.norm(profile)
    return profile / norm if norm > 0 else profile
