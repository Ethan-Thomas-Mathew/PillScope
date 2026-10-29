"""One-picture summary of an identification (used by the app's 'Save report' and the command line)."""
import cv2
import numpy as np
from matplotlib.figure import Figure

from . import pipeline


def mask_overlay(a):
    out = a.image.copy()
    contours, _ = cv2.findContours(a.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 200, 0), max(2, a.image.shape[0] // 150))
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def keypoint_view(a):
    """Pill crop with SIFT keypoints drawn on it (RGB)."""
    return cv2.cvtColor(cv2.drawKeypoints(a.crop, a.keypoints(), None, color=(0, 200, 255),
                                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS), cv2.COLOR_BGR2RGB)


def describe(a):
    """Lines of text describing the pill that was found."""
    s, c = a.shape_info, a.color_info
    return [
        f"Shape: {s['shape']}  (circularity {s['circularity']}, aspect {s['aspect']})",
        f"Size: {s['area']} px area",
        f"Colour: {c['colorName']}" + (f" + {c['accent']} speckles ({c['accentPercent']}%)" if c.get("accent") else ""),
        f"  HSV = {c['hue']}, {c['saturation']}, {c['value']}",
        f"  LAB = {c['L']}, {c['a']}, {c['b']}",
        f"Canny edge pixels: {a.edge_count}",
        f"SIFT keypoints: {len(a.kp_pts)}",
    ]


def build_report_figure(a, matches, settings, db=None):
    """Figure with the pipeline stages, the answer, and the best match drawn out."""
    fig = Figure(figsize=(12, 7.5), dpi=100)
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.15], hspace=0.35, wspace=0.12)
    stages = [("1. Photo + pill outline", mask_overlay(a), None),
              ("2. Preprocessed", a.gray, "gray"),
              ("3. Canny edges", a.edge_map, "gray"),
              ("4. SIFT keypoints", keypoint_view(a), None)]
    for i, (title, img, cmap) in enumerate(stages):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    ax = fig.add_subplot(gs[0, 4])
    ax.axis("off")
    ax.text(0, 1, "\n".join(describe(a)), va="top", family="monospace", fontsize=8)

    kind, sentence = pipeline.verdict(matches)
    color = {"match": "#1b7f3b", "possible": "#b36b00", "unknown": "#b00020"}[kind]
    fig.suptitle(sentence, color=color, fontsize=13, fontweight="bold", y=0.97)

    axm = fig.add_subplot(gs[1, :3])
    axm.axis("off")
    if matches and db is not None:
        best = matches[0]
        ref = pipeline.analyze(_load(db, best.row), settings)
        axm.imshow(pipeline.draw_match_view(a, ref, settings.sift_ratio))
        axm.set_title(f"Your photo (left) vs {best.row['name']} (right): green lines = matching SIFT keypoints", fontsize=9)

    axt = fig.add_subplot(gs[1, 3:])
    axt.axis("off")
    lines = ["Rank  Pill        Score  SIFT matches"]
    for i, m in enumerate(matches, 1):
        lines.append(f"{i:<5} {m.row['name']:<11} {m.score * 100:5.1f}  {m.inliers}")
    axt.text(0, 1, "Top matches\n\n" + "\n".join(lines), va="top", family="monospace", fontsize=9)
    return fig


def _load(db, row):
    from .io_utils import imread
    return imread(db.image_path(row))
