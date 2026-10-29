"""Build the pill database from the pill-image dataset.

The dataset has files like  12.jpg  (12)r30.jpg  (12)outline.jpg  -> pill number 12.
12.jpg is the original photo and (12)r30.jpg is that photo rotated by 30 degrees.
We use the originals as the *reference photos* and keep the rotated ones for testing.

    python scripts/build_database.py --source "C:/path/to/pill_images"
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pillscope.config import CACHE_FILE, DB_CSV, GALLERY_DIR, SAMPLES_DIR, Settings   # noqa: E402
from pillscope.database import PillDatabase                               # noqa: E402

PATTERN = re.compile(r"^\((\d+)\)(.+)\.(jpg|jpeg|png)$", re.IGNORECASE)    # (12)r30.jpg
PLAIN = re.compile(r"^(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)                  # 12.jpg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="folder with the (N)variant.jpg images")
    ap.add_argument("--variant", default="plain", help="which variant becomes the reference photo: plain (12.jpg), r15, r30 ...")
    ap.add_argument("--holdout", type=int, nargs="*", default=[252],
                    help="pill numbers kept OUT of the database; their photos go to data/samples as 'unknown pill' tests")
    ap.add_argument("--limit", type=int, default=0, help="only use the first N pills (0 = all)")
    args = ap.parse_args()

    files = {}
    for p in Path(args.source).iterdir():
        m = PLAIN.match(p.name) if args.variant == "plain" else PATTERN.match(p.name)
        if m and (args.variant == "plain" or m.group(2) == args.variant):
            files[int(m.group(1))] = p
    held = {n: files.pop(n) for n in args.holdout if n in files}
    numbers = sorted(files)[: args.limit or None]
    if not numbers:
        sys.exit(f"No matching files found in {args.source}")

    for f in (DB_CSV, CACHE_FILE):
        if f.exists():
            f.unlink()
    if GALLERY_DIR.exists():
        for old in GALLERY_DIR.glob("*"):
            old.unlink()

    settings = Settings()
    db = PillDatabase()
    for i, n in enumerate(numbers, 1):
        db.add_image(files[n], settings, name=f"Pill {n:03d}", notes=f"dataset pill #{n}")
        if i % 50 == 0 or i == len(numbers):
            print(f"  analysed {i}/{len(numbers)}")
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for n, path in held.items():
        shutil.copy2(path, SAMPLES_DIR / f"sample_unknown_pill{n:03d}_not_in_database{path.suffix}")
    db._fingerprint = db._fp(settings)
    db.save()
    db.save_cache()
    print(f"Done: {len(db.rows)} pills -> {DB_CSV}")


if __name__ == "__main__":
    main()
