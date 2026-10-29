"""Paths and default settings in one place."""
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GALLERY_DIR = DATA_DIR / "gallery"      # reference pill images (the "database" photos)
SAMPLES_DIR = DATA_DIR / "samples"      # a few test photos to try the app with
DB_CSV = DATA_DIR / "pill_database.csv"
CACHE_FILE = DATA_DIR / "gallery_features.pkl"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
IMAGE_FILETYPES = [("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]


@dataclass
class Settings:
    """Knobs of the identification pipeline (all editable in the app)."""
    work_size: int = 192          # the pill crop is resized to work_size x work_size
    denoise: bool = True          # Lab 6: median blur before analysis
    clahe: bool = True            # Lab 6: local contrast boost
    seg_method: str = "threshold"  # "threshold" or "region_growing" (Lab 7)
    sift_ratio: float = 0.75      # Lab 8: Lowe ratio test
    shortlist: int = 400          # how many database pills get the slower SIFT check (best by quick clues first)
    top_k: int = 5                # how many matches to show
