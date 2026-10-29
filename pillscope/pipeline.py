"""The full identification pipeline.

   photo -> segment pill -> shape + colour -> preprocess -> edges
         -> descriptors (colour histogram, FFT texture, HOG, SIFT) -> compare with database
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import basic_ops, color, edges, features, frequency, segmentation
from .config import Settings

# How much each clue counts in the final score (they add up to 1.0)
WEIGHTS = {"sift": 0.55, "color": 0.20, "texture": 0.10, "hog": 0.15}
SIFT_FULL_SCORE = 30     # this many geometrically-consistent SIFT matches = a perfect SIFT score
CONFIDENT_INLIERS = 20   # measured: right matches have 29+ inliers, pills not in the database have ~3
POSSIBLE_INLIERS = 10


@dataclass
class PillAnalysis:
    """Everything the pipeline learns about one pill photo."""
    image: np.ndarray = None          # original photo (BGR)
    mask: np.ndarray = None           # True where the pill is
    shape_info: dict = field(default_factory=dict)
    color_info: dict = field(default_factory=dict)
    crop: np.ndarray = None           # square pill crop (BGR)
    crop_mask: np.ndarray = None
    gray: np.ndarray = None           # preprocessed grey crop
    edge_map: np.ndarray = None
    edge_count: int = 0
    lab_hist: np.ndarray = None
    radial: np.ndarray = None
    hog: np.ndarray = None
    kp_pts: np.ndarray = None         # SIFT keypoint positions (N, 2)
    kp_size: np.ndarray = None
    kp_angle: np.ndarray = None
    desc: np.ndarray = None           # SIFT descriptors (N, 128)

    def lite(self):
        """Copy without the big images - this is what we store for database pills."""
        keep = PillAnalysis()
        for name in ("shape_info", "color_info", "edge_count", "lab_hist", "radial", "hog",
                     "kp_pts", "kp_size", "kp_angle", "desc"):
            setattr(keep, name, getattr(self, name))
        return keep

    def keypoints(self):
        """Rebuild OpenCV KeyPoint objects (needed for drawing)."""
        return [cv2.KeyPoint(float(x), float(y), float(s), float(a))
                for (x, y), s, a in zip(self.kp_pts, self.kp_size, self.kp_angle)]


@dataclass
class Match:
    row: dict                  # the database row
    score: float               # final score 0..1
    parts: dict                # score of each clue
    inliers: int = 0           # SIFT matches that agree geometrically


# ------------------------------------------------------------------ analysis
def analyze(img_bgr, settings=None):
    s = settings or Settings()
    a = PillAnalysis(image=img_bgr)
    a.mask = segmentation.segment_pill(img_bgr, s.seg_method)                      # Lab 7
    a.shape_info = segmentation.describe_shape(a.mask)
    a.color_info = color.color_signature(img_bgr, a.mask)                         # Lab 4
    a.crop, a.crop_mask = segmentation.crop_to_mask(img_bgr, a.mask, s.work_size)
    a.gray = edges.preprocess(basic_ops.to_gray(a.crop), s.denoise, s.clahe)      # Lab 1 + 6
    a.edge_map = edges.canny_edges(a.gray, 50, 150)                               # Lab 6
    a.edge_count = edges.edge_pixel_count(a.edge_map)
    a.lab_hist = color.lab_histogram(a.crop, a.crop_mask)                         # Lab 4
    a.radial = frequency.radial_energy(a.gray)                                    # Lab 5
    a.hog = features.hog_vector(a.gray)                                           # Lab 8
    kps, desc = features.detect_sift(a.gray, a.crop_mask)                         # Lab 8
    if desc is None:
        kps, desc = [], np.zeros((0, 128), dtype=np.float32)
    a.kp_pts = np.float32([k.pt for k in kps]).reshape(-1, 2)
    a.kp_size = np.float32([k.size for k in kps])
    a.kp_angle = np.float32([k.angle for k in kps])
    a.desc = desc
    return a


# ------------------------------------------------------------------ comparison
def quick_scores(q, r):
    """Cheap clues (colour, texture, HOG) - used to pick a shortlist."""
    color_sim = 1.0 - cv2.compareHist(q.lab_hist, r.lab_hist, cv2.HISTCMP_BHATTACHARYYA)
    return {
        "color": float(np.clip(color_sim, 0, 1)),
        "texture": float(np.clip(features.cosine_similarity(q.radial, r.radial), 0, 1)),
        "hog": float(np.clip(features.cosine_similarity(q.hog, r.hog), 0, 1)),
    }


def sift_score(q, r, ratio):
    """Slower clue: SIFT matches that also agree on one rotation/scale/shift."""
    good = features.ratio_test_matches(q.desc, r.desc, cv2.NORM_L2, ratio)
    inliers = features.geometric_inliers(q.kp_pts, r.kp_pts, good)
    return min(1.0, inliers / SIFT_FULL_SCORE), inliers, good


def combine(parts):
    return sum(WEIGHTS[k] * parts.get(k, 0.0) for k in WEIGHTS)


def identify(query, database, settings=None):
    """Compare a PillAnalysis with every pill in the database; return the best Matches."""
    s = settings or Settings()
    entries = database.entries()          # list of (row, lite analysis)
    if not entries:
        return []

    # Stage 1: quick clues for everyone
    quick = []
    for row, ref in entries:
        parts = quick_scores(query, ref)
        pre = (WEIGHTS["color"] * parts["color"] + WEIGHTS["texture"] * parts["texture"]
               + WEIGHTS["hog"] * parts["hog"])
        quick.append((pre, row, ref, parts))
    quick.sort(key=lambda t: t[0], reverse=True)

    # Stage 2: SIFT only for the shortlist
    matches = []
    for pre, row, ref, parts in quick[:s.shortlist]:
        sc, inliers, _ = sift_score(query, ref, s.sift_ratio)
        parts = dict(parts, sift=sc)
        matches.append(Match(row=row, score=combine(parts), parts=parts, inliers=inliers))
    matches.sort(key=lambda m: m.score, reverse=True)
    return matches[:s.top_k]


def verdict(matches):
    """Plain-language answer: ('match' | 'possible' | 'unknown', sentence)."""
    if not matches:
        return "unknown", "The database is empty - add some pills first."
    best = matches[0]
    name = best.row.get("name", "?")
    if best.inliers >= CONFIDENT_INLIERS:
        return "match", f"Confident match: {name} ({best.inliers} consistent SIFT matches)."
    if best.inliers >= POSSIBLE_INLIERS:
        return "possible", f"Possible match: {name}, but only {best.inliers} consistent SIFT matches. Check it yourself."
    return "unknown", "No reliable match - this pill is probably not in the database."


def draw_match_view(query, ref, ratio=0.75, max_lines=40):
    """Side-by-side picture with lines joining matching SIFT keypoints (RGB)."""
    good = features.ratio_test_matches(query.desc, ref.desc, cv2.NORM_L2, ratio)
    good = sorted(good, key=lambda m: m.distance)[:max_lines]
    out = cv2.drawMatches(query.crop, query.keypoints(), ref.crop, ref.keypoints(), good, None,
                          matchColor=(0, 220, 0), singlePointColor=(200, 120, 0),
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
