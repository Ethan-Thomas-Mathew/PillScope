"""Measure how well the pipeline re-identifies pills it already knows.

Two tests:

 A) Synthetic: take each database photo, damage it (rotate, shrink, add noise, darken)
    and check the pipeline still names the right pill. The right answer is certain here.

 B) Real rotated photos: the dataset's (N)r30.jpg files are original photos rotated by 30 degrees,
    but the number N often does NOT match the original's number. So the right answer is found
    by a pixel-by-pixel check (rotate every original by 30 degrees and compare), not by the filename.

    python scripts/evaluate.py --source "C:/path/to/pill_images"
"""
import argparse
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pillscope.config import OUTPUT_DIR, Settings          # noqa: E402
from pillscope.database import PillDatabase                # noqa: E402
from pillscope.pipeline import analyze, identify           # noqa: E402

PATTERN = re.compile(r"^\((\d+)\)(.+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def rotate_keep(img, angle):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderValue=(255, 255, 255))


def shrink(img, f=0.7):
    small = cv2.resize(img, None, fx=f, fy=f)
    canvas = np.full_like(img, 255)
    h, w = small.shape[:2]
    y, x = (img.shape[0] - h) // 2, (img.shape[1] - w) // 2
    canvas[y:y + h, x:x + w] = small
    return canvas


def noisy(img):
    rng = np.random.default_rng(0)
    return np.clip(img.astype(np.float32) + rng.normal(0, 12, img.shape), 0, 255).astype(np.uint8)


def darker(img):
    return cv2.convertScaleAbs(img, alpha=0.7, beta=-10)


TESTS = {
    "unchanged": lambda im: im,
    "rotated 30": lambda im: rotate_keep(im, 30),
    "rotated 45": lambda im: rotate_keep(im, 45),
    "rotated 180": lambda im: rotate_keep(im, 180),
    "shrunk to 70%": shrink,
    "gaussian noise": noisy,
    "darker": darker,
}


def is_true_rotation(original, rotated, angles=(-33, -30, -27, 27, 30, 33)):
    """True if `rotated` is `original` turned by about +-30 degrees (pixel check)."""
    if original is None or rotated is None or original.shape != rotated.shape:
        return False
    a = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = a.shape
    best = 1e9
    for ang in angles:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        r = cv2.warpAffine(a, m, (w, h), borderValue=255)
        pill_area = (r < 235) | (b < 235)            # ignore the white background
        best = min(best, float(np.abs(r - b)[pill_area].mean()))
    return best < 8.0     # real rotations score ~2-3, unrelated images score 15 or more


def score(db, settings, items):
    """items: list of (true pill name(s), image). Returns (top1, top5, seconds per photo)."""
    top1 = top5 = 0
    t0 = time.time()
    for truth, img in items:
        truth = {truth} if isinstance(truth, str) else set(truth)
        names = [m.row["name"] for m in identify(analyze(img, settings), db, settings)]
        top1 += bool(names) and names[0] in truth
        top5 += bool(truth & set(names))
    n = max(len(items), 1)
    return top1 / n, top5 / n, (time.time() - t0) / n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--max", type=int, default=0, help="only test the first N database pills (0 = all)")
    ap.add_argument("--only", default="", help="comma-separated test names to run")
    args = ap.parse_args()

    settings = Settings()
    db = PillDatabase()
    if not db.rows:
        sys.exit("Database is empty. Run scripts/build_database.py first.")
    db.ensure_features(settings)
    src = Path(args.source)
    originals = []                          # (name, image) of each database photo
    for row in db.rows:
        img = cv2.imread(str(db.image_path(row)))
        if img is not None:
            originals.append((row["name"], img))
    if args.max:
        originals = originals[:args.max]

    lines = [f"Database: **{len(db.rows)} pills**.", "",
             f"### A) Synthetic damage on {len(originals)} database photos", "",
             "| Test | Top-1 | Top-5 | Seconds / photo |", "|---|---|---|---|"]
    print(f"{len(db.rows)} database pills\n\nA) synthetic damage ({len(originals)} photos each)")
    for test, fn in TESTS.items():
        if args.only and test not in args.only.split(","):
            continue
        t1, t5, sec = score(db, settings, [(n, fn(im)) for n, im in originals])
        print(f"  {test:16s} top-1 {t1:6.1%}   top-5 {t5:6.1%}   {sec:.2f}s/photo")
        lines.append(f"| {test} | {t1:.1%} | {t5:.1%} | {sec:.2f} |")

    # B) real rotated photos from the dataset; the right answer comes from a pixel check
    by_name = {n: im for n, im in originals}
    queries, same_number = [], 0
    for p in sorted(src.iterdir()):
        m = PATTERN.match(p.name)
        if not (m and m.group(2) == "r30"):
            continue
        rotated = cv2.imread(str(p))
        if rotated is None:
            continue
        truth = [n for n, im in by_name.items()
                 if im.shape == rotated.shape and is_true_rotation(im, rotated, angles=(-30, 30))]
        if truth:
            queries.append((truth, rotated))
            same_number += f"Pill {int(m.group(1)):03d}" in truth
    if queries:
        t1, t5, sec = score(db, settings, queries)
        print(f"\nB) real rotated photos: {len(queries)} (r30) photos, {same_number} carry the same number as their original")
        print(f"   top-1 {t1:.1%}   top-5 {t5:.1%}")
        lines += ["", "### B) Real 30-degree rotated photos from the dataset", "",
                  f"{len(queries)} `(N)r30.jpg` photos are an original rotated by 30 degrees (checked pixel by pixel). "
                  f"Only {same_number} of them carry the same number as their original, so the right answer is taken "
                  f"from the pixel check, not the filename.", "",
                  "| Top-1 | Top-5 | Seconds / photo |", "|---|---|---|", f"| {t1:.1%} | {t5:.1%} | {sec:.2f} |"]

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "evaluation.md").write_text("# Evaluation\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved {OUTPUT_DIR / 'evaluation.md'}")


if __name__ == "__main__":
    main()
