"""Lab 1 - basic image operations: format/color conversion, geometry, arithmetic."""
import cv2
import numpy as np

FILL = (235, 235, 235)   # light grey used for empty corners after rotate/shift


# ---------- conversions ----------
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_binary(img, thresh=127):
    _, binary = cv2.threshold(to_gray(img), thresh, 255, cv2.THRESH_BINARY)
    return binary


def to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def save_as(img, path):
    """Save in any format OpenCV knows (decided by the file extension)."""
    from .io_utils import imwrite
    imwrite(path, img)


# ---------- geometry ----------
def rotate(img, angle):
    """Rotate and grow the canvas so no corner is cut off."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    m[0, 2] += new_w / 2 - center[0]
    m[1, 2] += new_h / 2 - center[1]
    return cv2.warpAffine(img, m, (new_w, new_h), borderValue=FILL)


def scale(img, factor):
    if factor <= 0:
        raise ValueError("Scale factor must be greater than 0")
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)


def translate(img, dx, dy):
    h, w = img.shape[:2]
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (w, h), borderValue=FILL)


def flip(img, code):
    """code: 1 = horizontal, 0 = vertical, -1 = both."""
    return cv2.flip(img, code)


# ---------- arithmetic / bitwise ----------
def add_value(img, value):
    return cv2.add(img, np.full(img.shape, value, dtype=np.uint8))


def subtract_value(img, value):
    return cv2.subtract(img, np.full(img.shape, value, dtype=np.uint8))


def bitwise(img, op):
    """NOT / AND / OR / XOR against a white centred rectangle."""
    op = op.upper()
    if op == "NOT":
        return cv2.bitwise_not(img)
    h, w = img.shape[:2]
    mask = np.zeros_like(img)
    cv2.rectangle(mask, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (255, 255, 255), -1)
    funcs = {"AND": cv2.bitwise_and, "OR": cv2.bitwise_or, "XOR": cv2.bitwise_xor}
    return funcs[op](img, mask)


# name -> function(img, p) used by the Explore tab; p is the slider value
OPERATIONS = {
    "Grayscale": lambda img, p: to_gray(img),
    "Binary (threshold = p)": lambda img, p: to_binary(img, int(p)),
    "HSV": lambda img, p: to_hsv(img),
    "Rotate (p degrees)": lambda img, p: rotate(img, p),
    "Scale (p / 100)": lambda img, p: scale(img, max(p, 1) / 100.0),
    "Shift right/down (p px)": lambda img, p: translate(img, p, p),
    "Flip horizontal": lambda img, p: flip(img, 1),
    "Flip vertical": lambda img, p: flip(img, 0),
    "Brighten (+p)": lambda img, p: add_value(img, int(p)),
    "Darken (-p)": lambda img, p: subtract_value(img, int(p)),
    "Invert (NOT)": lambda img, p: bitwise(img, "NOT"),
    "AND with box": lambda img, p: bitwise(img, "AND"),
    "OR with box": lambda img, p: bitwise(img, "OR"),
    "XOR with box": lambda img, p: bitwise(img, "XOR"),
}
