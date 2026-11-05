import numpy as np
import pytest

from pillscope import color


def test_classify_color_name_basics():
    assert color.classify_color_name(0, 0, 255) == "white"
    assert color.classify_color_name(0, 0, 10) == "black"
    assert color.classify_color_name(0, 10, 128) == "gray"
    assert color.classify_color_name(0, 200, 200) == "red"
    assert color.classify_color_name(115, 200, 200) == "blue"


def test_color_signature_on_plain_pill(pill_photo):
    info = color.color_signature(pill_photo)
    assert 0 <= info["hue"] <= 180
    assert info["colorName"]  # never empty
    assert 0 <= info["saturation"] <= 255
    assert 0 <= info["value"] <= 255


def test_color_signature_respects_mask(pill_photo):
    full_mask = np.ones(pill_photo.shape[:2], dtype=bool)
    no_mask = color.color_signature(pill_photo, mask=None)
    with_mask = color.color_signature(pill_photo, mask=full_mask)
    assert no_mask["colorName"] == with_mask["colorName"]


def test_lab_histogram_normalizes_to_one(pill_photo):
    hist = color.lab_histogram(pill_photo)
    assert hist.sum() == pytest.approx(1.0, abs=1e-4)


def test_lab_histogram_empty_mask_returns_zeros(pill_photo):
    empty_mask = np.zeros(pill_photo.shape[:2], dtype=bool)
    hist = color.lab_histogram(pill_photo, mask=empty_mask)
    assert hist.sum() == 0.0
