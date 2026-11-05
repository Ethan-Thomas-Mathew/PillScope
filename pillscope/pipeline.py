"""The full identification pipeline.

   photo -> segment pill -> shape + colour -> preprocess -> edges
         -> descriptors (colour histogram, FFT texture, HOG, SIFT) -> compare with database
"""
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

from . import basic_ops, color, edges, features, frequency, segmentation
from .config import Settings

BGRImage = np.ndarray  # cv2's stubs aren't precise enough for a tighter dtype here
Mask = np.ndarray

# How much each clue counts in the final score (they add up to 1.0)
WEIGHTS = {"sift": 0.55, "color": 0.20, "texture": 0.10, "hog": 0.15}
SIFT_FULL_SCORE = 30     # this many geometrically-consistent SIFT matches = a perfect SIFT score
CONFIDENT_INLIERS = 20   # measured: right matches have 29+ inliers, pills not in the database have ~3
POSSIBLE_INLIERS = 10


@dataclass
class PillAnalysis:
    """Everything the pipeline learns about one pill photo."""
    image: Optional[BGRImage] = None          # original photo (BGR)
    mask: Optional[Mask] = None               # True where the pill is
    shape_info: Any = field(default_factory=dict)
    color_info: Any = field(default_factory=dict)
    crop: Optional[BGRImage] = None           # square pill crop (BGR)
    crop_mask: Optional[Mask] = None
    gray: Optional[np.ndarray] = None         # preprocessed grey crop
    edge_map: Optional[np.ndarray] = None
    edge_count: int = 0
    lab_hist: Optional[np.ndarray] = None
    radial: Optional[np.ndarray] = None
    hog: Optional[np.ndarray] = None
    kp_pts: Optional[np.ndarray] = None       # SIFT keypoint positions (N, 2)
    kp_size: Optional[np.ndarray] = None
    kp_angle: Optional[np.ndarray] = None
    desc: Optional[np.ndarray] = None         # SIFT descriptors (N, 128)

    def lite(self) -> "PillAnalysis":
        """Copy without the big images - this is what we store for database pills."""
        keep = PillAnalysis()
        for name in ("shape_info", "color_info", "edge_count", "lab_hist", "radial", "hog",
                     "kp_pts", "kp_size", "kp_angle", "desc"):
            setattr(keep, name, getattr(self, name))
        return keep

    def keypoints(self) -> list[cv2.KeyPoint]:
        """Rebuild OpenCV KeyPoint objects (needed for drawing)."""
        assert self.kp_pts is not None and self.kp_size is not None and self.kp_angle is not None
        return [cv2.KeyPoint(float(x), float(y), float(s), float(a))
                for (x, y), s, a in zip(self.kp_pts, self.kp_size, self.kp_angle)]


@dataclass
class Match:
    row: dict[str, Any]        # the database row
    score: float               # final score 0..1
    parts: dict[str, float]    # score of each clue
    inliers: int = 0           # SIFT matches that agree geometrically


# ------------------------------------------------------------------ analysis
def analyze(img_bgr: BGRImage, settings: Optional[Settings] = None) -> PillAnalysis:
    s = settings or Settings()
    mask = segmentation.segment_pill(img_bgr, s.seg_method)
    shape_info = segmentation.describe_shape(mask)
    color_info = color.color_signature(img_bgr, mask)
    crop, crop_mask = segmentation.crop_to_mask(img_bgr, mask, s.work_size)
    gray = edges.preprocess(basic_ops.to_gray(crop), s.denoise, s.clahe)
    edge_map = edges.canny_edges(gray, 50, 150)
    edge_count = edges.edge_pixel_count(edge_map)
    lab_hist = color.lab_histogram(crop, crop_mask)
    radial = frequency.radial_energy(gray)
    hog = features.hog_vector(gray)
    kps, desc = features.detect_sift(gray, crop_mask)
    if desc is None:
        kps, desc = (), np.zeros((0, 128), dtype=np.float32)
    kp_pts = np.array([k.pt for k in kps], dtype=np.float32).reshape(-1, 2)
    kp_size = np.array([k.size for k in kps], dtype=np.float32)
    kp_angle = np.array([k.angle for k in kps], dtype=np.float32)
    return PillAnalysis(image=img_bgr, mask=mask, shape_info=shape_info, color_info=color_info,
                         crop=crop, crop_mask=crop_mask, gray=gray, edge_map=edge_map,
                         edge_count=edge_count, lab_hist=lab_hist, radial=radial, hog=hog,
                         kp_pts=kp_pts, kp_size=kp_size, kp_angle=kp_angle, desc=desc)


# ------------------------------------------------------------------ comparison
def quick_scores(q: PillAnalysis, r: PillAnalysis) -> dict[str, float]:
    """Cheap clues (colour, texture, HOG) - used to pick a shortlist.

    Both analyses are expected to come from analyze() (or its .lite() copy), which always
    fills in these fields, so they're asserted non-None here rather than typed as optional."""
    assert q.lab_hist is not None and r.lab_hist is not None
    assert q.radial is not None and r.radial is not None
    assert q.hog is not None and r.hog is not None
    color_sim = 1.0 - cv2.compareHist(q.lab_hist, r.lab_hist, cv2.HISTCMP_BHATTACHARYYA)
    return {
        "color": float(np.clip(color_sim, 0, 1)),
        "texture": float(np.clip(features.cosine_similarity(q.radial, r.radial), 0, 1)),
        "hog": float(np.clip(features.cosine_similarity(q.hog, r.hog), 0, 1)),
    }


def sift_score(q: PillAnalysis, r: PillAnalysis, ratio: float) -> tuple[float, int, list[cv2.DMatch]]:
    """Slower clue: SIFT matches that also agree on one rotation/scale/shift."""
    assert q.kp_pts is not None and r.kp_pts is not None
    good = features.ratio_test_matches(q.desc, r.desc, cv2.NORM_L2, ratio)
    inliers = features.geometric_inliers(q.kp_pts, r.kp_pts, good)
    return min(1.0, inliers / SIFT_FULL_SCORE), inliers, good


def combine(parts: dict[str, float]) -> float:
    return sum(WEIGHTS[k] * parts.get(k, 0.0) for k in WEIGHTS)


def identify(query: PillAnalysis, database: Any, settings: Optional[Settings] = None) -> list[Match]:
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
    for _pre, row, ref, parts in quick[:s.shortlist]:
        sc, inliers, _ = sift_score(query, ref, s.sift_ratio)
        parts = dict(parts, sift=sc)
        matches.append(Match(row=row, score=combine(parts), parts=parts, inliers=inliers))
    matches.sort(key=lambda m: m.score, reverse=True)
    return matches[:s.top_k]


def verdict(matches: list[Match]) -> tuple[str, str]:
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


def draw_match_view(query: PillAnalysis, ref: PillAnalysis, ratio: float = 0.75, max_lines: int = 40) -> np.ndarray:
    """Side-by-side picture with lines joining matching SIFT keypoints (RGB)."""
    assert query.crop is not None and ref.crop is not None
    good = features.ratio_test_matches(query.desc, ref.desc, cv2.NORM_L2, ratio)
    good = sorted(good, key=lambda m: m.distance)[:max_lines]
    out = cv2.drawMatches(query.crop, query.keypoints(), ref.crop, ref.keypoints(), good, None,  # type: ignore[call-overload]
                          matchColor=(0, 220, 0), singlePointColor=(200, 120, 0),
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
