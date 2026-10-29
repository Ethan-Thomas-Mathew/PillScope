"""The pill database: a CSV of pill facts + cached image features for fast matching."""
import csv
import datetime
import pickle
import shutil
from pathlib import Path


from . import color
from .io_utils import imread
from .config import CACHE_FILE, DB_CSV, GALLERY_DIR, IMAGE_PATTERNS, Settings
from .pipeline import analyze

FIELDNAMES = [
    "id", "dateAdded", "imageFilename", "name", "shape", "colorName",
    "imprintCode", "dosageMg", "manufacturer",
    "hue", "saturation", "value", "L", "a", "b", "notes",
]
SHAPES = ["round", "oval", "capsule", "oblong", "triangle", "diamond",
          "hexagon", "pentagon", "square", "rectangle", "other"]
REQUIRED_FIELDS = ["name", "shape", "imprintCode", "dosageMg", "manufacturer"]


class PillDatabase:
    def __init__(self, csv_path=DB_CSV, gallery_dir=GALLERY_DIR, cache_path=CACHE_FILE):
        self.csv_path = Path(csv_path)
        self.gallery_dir = Path(gallery_dir)
        self.cache_path = Path(cache_path)
        self.rows = []
        self.features = {}        # imageFilename -> lite PillAnalysis
        self._fingerprint = None
        self.load()

    # ---------------------------------------------------------- csv
    def load(self):
        self.rows = []
        if self.csv_path.exists():
            with open(self.csv_path, newline="", encoding="utf-8") as f:
                self.rows = list(csv.DictReader(f))

    def save(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            for r in self.rows:
                writer.writerow({k: r.get(k, "") for k in FIELDNAMES})

    def next_id(self):
        ids = [int(r["id"]) for r in self.rows if str(r.get("id", "")).isdigit()]
        return max(ids) + 1 if ids else 1

    def image_path(self, row):
        return self.gallery_dir / row["imageFilename"]

    def find(self, row_id):
        return next((r for r in self.rows if str(r["id"]) == str(row_id)), None)

    def delete(self, row_id):
        self.rows = [r for r in self.rows if str(r["id"]) != str(row_id)]

    # ---------------------------------------------------------- adding pills
    def make_row(self, analysis, filename, name="", **extra):
        """Build a database row from an analysed photo (shape/colour filled automatically)."""
        row = {k: "" for k in FIELDNAMES}
        row.update(id=self.next_id(), dateAdded=datetime.date.today().isoformat(),
                   imageFilename=filename, name=name or f"Pill {self.next_id():03d}",
                   shape=analysis.shape_info["shape"], colorName=self._color_text(analysis.color_info),
                   hue=analysis.color_info["hue"], saturation=analysis.color_info["saturation"],
                   value=analysis.color_info["value"], L=analysis.color_info["L"],
                   a=analysis.color_info["a"], b=analysis.color_info["b"])
        row.update({k: v for k, v in extra.items() if k in FIELDNAMES})
        return row

    @staticmethod
    def _color_text(info):
        """'white' or, for speckled pills, 'white, blue speckles'."""
        return info["colorName"] + (f", {info['accent']} speckles" if info.get("accent") else "")

    def add_image(self, image_path, settings=None, **fields):
        """Copy a photo into the gallery, analyse it, and add a row."""
        image_path = Path(image_path)
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        dest = self.gallery_dir / image_path.name
        if dest.exists() and dest.resolve() != image_path.resolve():
            dest = self.gallery_dir / f"{image_path.stem}_{self.next_id()}{image_path.suffix}"
        if not dest.exists():
            shutil.copy2(image_path, dest)
        img = imread(dest)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        a = analyze(img, settings)
        row = self.make_row(a, dest.name, **fields)
        self.rows.append(row)
        self.features[dest.name] = a.lite()
        return row

    def import_folder(self, folder, settings=None, progress=None):
        """Add every image in a folder."""
        files = sorted(p for pat in IMAGE_PATTERNS for p in Path(folder).glob(pat))
        for i, p in enumerate(files):
            self.add_image(p, settings)
            if progress:
                progress(i + 1, len(files))
        return len(files)

    # ---------------------------------------------------------- feature cache
    @staticmethod
    def _fp(settings):
        return (settings.work_size, settings.denoise, settings.clahe, settings.seg_method)

    def load_cache(self, settings):
        self.features = {}
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    saved = pickle.load(f)
                if saved.get("fingerprint") == self._fp(settings):
                    self.features = saved["features"]
            except Exception:
                self.features = {}
        self._fingerprint = self._fp(settings)

    def ensure_features(self, settings=None, progress=None):
        """Make sure every row has features; compute the missing ones and save the cache."""
        settings = settings or Settings()
        if self._fingerprint != self._fp(settings):
            self.load_cache(settings)
        todo = [r for r in self.rows if r["imageFilename"] not in self.features]
        for i, row in enumerate(todo):
            img = imread(self.image_path(row))
            if img is not None:
                self.features[row["imageFilename"]] = analyze(img, settings).lite()
            if progress:
                progress(i + 1, len(todo))
        if todo:
            self.save_cache()
        return len(todo)

    def save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump({"fingerprint": self._fingerprint, "features": self.features}, f)

    def entries(self):
        """(row, features) pairs that are ready for matching."""
        return [(r, self.features[r["imageFilename"]]) for r in self.rows
                if r["imageFilename"] in self.features]

    # ---------------------------------------------------------- checks (from pillDatabase.py)
    @staticmethod
    def missing_required_fields(row):
        return [f for f in REQUIRED_FIELDS if not str(row.get(f, "")).strip()]

    @staticmethod
    def color_mismatch(row):
        """If the typed colour disagrees with the measured colour, return the measured name."""
        name = str(row.get("colorName", "")).strip().lower()
        h, s, v = row.get("hue", ""), row.get("saturation", ""), row.get("value", "")
        if not name or "" in (h, s, v):
            return None
        try:
            detected = color.classify_color_name(float(h), float(s), float(v))
        except ValueError:
            return None
        if detected != "unknown" and detected not in name and name not in detected:
            return detected
        return None

    def duplicate_imprints(self):
        seen, dupes = {}, []
        for r in self.rows:
            key = (str(r.get("imprintCode", "")).strip().lower(), str(r.get("dosageMg", "")).strip())
            if not key[0]:
                continue
            if key in seen:
                dupes.append((seen[key], r))
            else:
                seen[key] = r
        return dupes

    def validate(self):
        """Return a list of human-readable problems."""
        problems = []
        no_imprint = sum(1 for r in self.rows if not str(r.get("imprintCode", "")).strip())
        for r in self.rows:
            missing = [f for f in self.missing_required_fields(r) if f not in ("imprintCode", "dosageMg", "manufacturer")]
            if missing:
                problems.append(f"Pill {r['id']}: missing {', '.join(missing)}")
            m = self.color_mismatch(r)
            if m:
                problems.append(f"Pill {r['id']}: colour says '{r['colorName']}' but the image looks '{m}'")
        for a, b in self.duplicate_imprints():
            problems.append(f"Pills {a['id']} and {b['id']} share imprint '{a['imprintCode']}' + dose {a['dosageMg']}")
        if no_imprint:
            problems.append(f"{no_imprint} pill(s) have no imprint code / dosage / manufacturer yet (fill these in the Database tab)")
        return problems
