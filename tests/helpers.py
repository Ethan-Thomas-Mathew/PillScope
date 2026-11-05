"""Shared test helpers - not collected by pytest (no test_ prefix)."""
import cv2
import numpy as np


def make_pill_photo(bgr_color=(170, 170, 210), shape="circle", size=240, bg=(245, 245, 245)):
    """A simple synthetic 'pill' photo: a filled shape on a plain background.

    Good enough to exercise segmentation/colour/shape without needing a real photo."""
    img = np.full((size, size, 3), bg, dtype=np.uint8)
    margin = size // 5
    if shape == "circle":
        cv2.circle(img, (size // 2, size // 2), size // 2 - margin, bgr_color, -1)
    elif shape == "square":
        cv2.rectangle(img, (margin, margin), (size - margin, size - margin), bgr_color, -1)
    elif shape == "capsule":
        h = size // 4
        cv2.rectangle(img, (margin, size // 2 - h // 2), (size - margin, size // 2 + h // 2), bgr_color, -1)
        cv2.circle(img, (margin, size // 2), h // 2, bgr_color, -1)
        cv2.circle(img, (size - margin, size // 2), h // 2, bgr_color, -1)
    else:
        raise ValueError(f"unknown shape: {shape}")
    return img
