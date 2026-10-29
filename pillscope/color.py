"""Lab 4 - color spaces, channels, and the pill 'color signature'."""
import cv2
import numpy as np

COLOR_SPACES = {
    "Grayscale": (cv2.COLOR_BGR2GRAY, ("Gray",)),
    "HSV":       (cv2.COLOR_BGR2HSV,   ("Hue", "Saturation", "Value")),
    "HLS":       (cv2.COLOR_BGR2HLS,   ("Hue", "Lightness", "Saturation")),
    "YCrCb":     (cv2.COLOR_BGR2YCrCb, ("Luma (Y)", "Cr (Red diff)", "Cb (Blue diff)")),
    "LAB":       (cv2.COLOR_BGR2LAB,   ("L (Lightness)", "a (Green-Red)", "b (Blue-Yellow)")),
    "XYZ":       (cv2.COLOR_BGR2XYZ,   ("X", "Y", "Z")),
}

# OpenCV stores hue as 0-179, every other channel as 0-255
CHANNEL_RANGE = {
    "HSV": [(0, 180), (0, 256), (0, 256)],
    "HLS": [(0, 180), (0, 256), (0, 256)],
}

HUE_COLOR_NAMES = [
    (0, 8, "red"), (8, 20, "orange"), (20, 33, "yellow"), (33, 78, "green"),
    (78, 100, "cyan"), (100, 130, "blue"), (130, 150, "purple"),
    (150, 170, "pink"), (170, 180, "red"),
]


def convert(img_bgr, space):
    return cv2.cvtColor(img_bgr, COLOR_SPACES[space][0])


def get_channels(img_bgr, space):
    converted = convert(img_bgr, space)
    labels = COLOR_SPACES[space][1]
    if converted.ndim == 2:
        return labels, [converted]
    return labels, [converted[:, :, i] for i in range(converted.shape[2])]


def channel_histogram(channel, space, idx):
    lo, hi = CHANNEL_RANGE.get(space, [(0, 256)] * 3)[idx]
    hist = cv2.calcHist([channel], [0], None, [hi - lo], [lo, hi])
    return hist.flatten(), lo, hi


def classify_color_name(h, s, v):
    """Turn mean H, S, V into a plain colour word."""
    if v < 50:
        return "black"
    if s < 35 and v > 200:
        return "white"
    if s < 35:
        return "gray"
    for lo, hi, name in HUE_COLOR_NAMES:
        if lo <= h < hi:
            return name
    return "unknown"


def _circular_mean_hue(hue, weights=None):
    """Hue is an angle (red is both 0 and 179), so a plain average can be wrong."""
    angle = hue.astype(np.float32) * 2.0 * np.pi / 180.0
    s, c = np.sin(angle), np.cos(angle)
    if weights is not None:
        s, c = s[weights], c[weights]
    return float((np.arctan2(s.mean(), c.mean()) % (2 * np.pi)) * 180.0 / (2 * np.pi))


def color_signature(img_bgr, mask=None):
    """Colour of the pill (only pixels inside `mask`, if given).

    Speckled pills have a pale base plus a few strongly coloured dots. A plain average would be
    pulled towards the dots, so the base colour is measured on the 75% least colourful pixels.
    The dots are reported separately as 'accent'."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    sel = mask.astype(bool) if mask is not None and mask.any() else np.ones(hsv.shape[:2], bool)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    base = sel & (sat <= np.percentile(sat[sel], 75))
    h = _circular_mean_hue(hue, base)
    s, v = float(np.median(sat[base])), float(np.median(val[base]))

    speck = sel & (sat > 90) & (val > 50)
    fraction = float(speck.sum()) / float(sel.sum())
    accent = ""
    if fraction > 0.02:
        accent = classify_color_name(_circular_mean_hue(hue, speck), float(sat[speck].mean()), float(val[speck].mean()))
    return {
        "hue": round(h, 1), "saturation": round(s, 1), "value": round(v, 1),
        "L": round(float(np.median(lab[:, :, 0][base])), 1),
        "a": round(float(np.median(lab[:, :, 1][base])), 1),
        "b": round(float(np.median(lab[:, :, 2][base])), 1),
        "colorName": classify_color_name(h, s, v),
        "accent": accent, "accentPercent": round(100.0 * fraction, 1),
    }


def lab_histogram(img_bgr, mask=None, bins=8):
    """Rotation-proof colour descriptor: a normalised 3D histogram in LAB space."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    m = None if mask is None else mask.astype(np.uint8) * 255
    hist = cv2.calcHist([lab], [0, 1, 2], m, [bins] * 3, [0, 256] * 3).flatten()
    total = hist.sum()
    return (hist / total).astype(np.float32) if total > 0 else hist.astype(np.float32)
