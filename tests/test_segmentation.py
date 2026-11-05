import numpy as np
from helpers import make_pill_photo

from pillscope import segmentation


def test_segment_pill_finds_the_circle():
    img = make_pill_photo(shape="circle")
    mask = segmentation.segment_pill(img)
    frac = mask.mean()
    assert 0.1 < frac < 0.9  # found something reasonable, not nothing / not everything


def test_segment_pill_falls_back_to_whole_image_on_a_blank_photo():
    blank = np.full((100, 100, 3), 200, dtype=np.uint8)  # no pill at all
    mask = segmentation.segment_pill(blank)
    assert mask.all()


def test_describe_shape_round():
    img = make_pill_photo(shape="circle")
    mask = segmentation.segment_pill(img)
    info = segmentation.describe_shape(mask)
    assert info["shape"] == "round"
    assert info["circularity"] > 0.7
    assert info["area"] > 0


def test_describe_shape_on_empty_mask():
    mask = np.zeros((50, 50), dtype=bool)
    info = segmentation.describe_shape(mask)
    assert info["shape"] == "other"
    assert info["area"] == 0


def test_crop_to_mask_is_square_and_the_right_size():
    img = make_pill_photo(shape="circle", size=240)
    mask = segmentation.segment_pill(img)
    crop, crop_mask = segmentation.crop_to_mask(img, mask, size=96)
    assert crop.shape == (96, 96, 3)
    assert crop_mask.shape == (96, 96)
    assert crop_mask.dtype == bool
    assert crop_mask.any()  # something survived the erosion
