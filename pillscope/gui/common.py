"""Shared bits for the GUI: the app's current state and small image helpers."""
import cv2
from PIL import Image, ImageTk

from ..config import Settings
from ..database import PillDatabase
from ..io_utils import MAX_SIDE, imread


def to_photo(img, max_w, max_h=None, rgb=False):
    """numpy image (grey / BGR / RGB) -> Tk image that fits inside max_w x max_h."""
    max_h = max_h or max_w
    if img.ndim == 2:
        shown = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif rgb:
        shown = img
    else:
        shown = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(shown)
    pil.thumbnail((max_w, max_h))
    return ImageTk.PhotoImage(pil)


class AppState:
    """What every tab shares: the photo being studied, the settings and the database."""

    def __init__(self):
        self.settings = Settings()
        self.db = PillDatabase()
        self.image = None          # BGR
        self.path = None
        self._listeners = []

    def subscribe(self, callback):
        self._listeners.append(callback)

    def load_path(self, path):
        img = imread(path, MAX_SIDE)
        if img is None:
            raise ValueError(f"Could not read the image:\n{path}")
        self.set_image(img, path)

    def set_image(self, img, path=None):
        self.image, self.path = img, path
        for callback in self._listeners:
            callback()
