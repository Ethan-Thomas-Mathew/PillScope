import numpy as np
import pytest

from pillscope import features


def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert features.cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert features.cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_cosine_similarity_zero_vector_is_zero_not_nan():
    a = np.zeros(4, dtype=np.float32)
    b = np.ones(4, dtype=np.float32)
    assert features.cosine_similarity(a, b) == 0.0


def test_ratio_test_matches_handles_missing_descriptors():
    assert features.ratio_test_matches(None, None) == []
    empty = np.zeros((0, 128), dtype=np.float32)
    assert features.ratio_test_matches(empty, empty) == []


def test_geometric_inliers_needs_at_least_three_matches():
    assert features.geometric_inliers(np.zeros((5, 2), dtype=np.float32), np.zeros((5, 2), dtype=np.float32), []) == 0


def test_hog_vector_is_unit_length_or_zero(pill_photo):
    import cv2

    gray = cv2.cvtColor(pill_photo, cv2.COLOR_BGR2GRAY)
    v = features.hog_vector(gray)
    norm = float(np.linalg.norm(v))
    assert norm == pytest.approx(1.0, abs=1e-3) or norm == 0.0
