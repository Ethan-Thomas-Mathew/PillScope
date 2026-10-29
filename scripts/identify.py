"""Identify a pill from the command line (no window needed).

    python scripts/identify.py data/samples/sample_1.jpg
    python scripts/identify.py photo.jpg --report outputs/report.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pillscope import pipeline, report                          # noqa: E402
from pillscope.config import OUTPUT_DIR, Settings              # noqa: E402
from pillscope.database import PillDatabase                    # noqa: E402
from pillscope.io_utils import imread                          # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--report", nargs="?", const=str(OUTPUT_DIR / "report.png"),
                    help="save a one-picture summary (default outputs/report.png)")
    args = ap.parse_args()

    img = imread(args.image)
    if img is None:
        sys.exit(f"Could not read {args.image}")
    settings = Settings()
    db = PillDatabase()
    if not db.rows:
        sys.exit("Database is empty. Run scripts/build_database.py first.")
    n = db.ensure_features(settings)
    if n:
        print(f"(prepared features for {n} database pills)")

    a = pipeline.analyze(img, settings)
    matches = pipeline.identify(a, db, settings)
    print("\n".join(report.describe(a)))
    print()
    print(pipeline.verdict(matches)[1])
    for i, m in enumerate(matches, 1):
        print(f"  {i}. {m.row['name']:<10} score {m.score * 100:5.1f}   SIFT matches {m.inliers}")
    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig = report.build_report_figure(a, matches, settings, db)
        fig.savefig(out, bbox_inches="tight")
        print(f"\nSaved report: {out}")


if __name__ == "__main__":
    main()
