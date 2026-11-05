import cv2
import pytest
from helpers import make_pill_photo

from pillscope.config import Settings
from pillscope.database import PillDatabase


@pytest.fixture
def empty_db(tmp_path):
    return PillDatabase(
        csv_path=tmp_path / "db.csv",
        gallery_dir=tmp_path / "gallery",
        cache_path=tmp_path / "cache.pkl",
    )


def test_next_id_starts_at_one(empty_db):
    assert empty_db.next_id() == 1


def test_add_image_then_ensure_features(empty_db, tmp_path):
    photo_path = tmp_path / "pill.jpg"
    cv2.imwrite(str(photo_path), make_pill_photo())

    row = empty_db.add_image(photo_path, name="Test Pill")
    assert row["id"] == 1
    assert row["name"] == "Test Pill"
    assert empty_db.find(1) is not None
    assert empty_db.find(999) is None

    # add_image analyses the photo immediately, so it's already in entries()
    assert len(empty_db.entries()) == 1
    # a never-fingerprinted db reloads the (empty, on-disk) cache first, so the
    # freshly-added pill - not yet on disk - is recomputed once; still correct,
    # just not free
    assert empty_db.ensure_features(Settings()) == 1
    assert len(empty_db.entries()) == 1


def test_delete_removes_the_row(empty_db, tmp_path):
    photo_path = tmp_path / "pill.jpg"
    cv2.imwrite(str(photo_path), make_pill_photo())
    row = empty_db.add_image(photo_path)
    empty_db.delete(row["id"])
    assert empty_db.find(row["id"]) is None


def test_save_and_reload_round_trips(empty_db, tmp_path):
    photo_path = tmp_path / "pill.jpg"
    cv2.imwrite(str(photo_path), make_pill_photo())
    empty_db.add_image(photo_path, name="Round Trip Pill")
    empty_db.save()

    reloaded = PillDatabase(csv_path=empty_db.csv_path, gallery_dir=empty_db.gallery_dir, cache_path=empty_db.cache_path)
    assert len(reloaded.rows) == 1
    assert reloaded.rows[0]["name"] == "Round Trip Pill"


def test_missing_required_fields():
    row = {"name": "", "shape": "round", "imprintCode": "", "dosageMg": "", "manufacturer": "Acme"}
    missing = PillDatabase.missing_required_fields(row)
    assert "name" in missing
    assert "shape" not in missing
    assert "manufacturer" not in missing


def test_duplicate_imprints_matches_case_insensitively(empty_db):
    empty_db.rows = [
        {"id": "1", "imprintCode": "ABC", "dosageMg": "10"},
        {"id": "2", "imprintCode": "abc", "dosageMg": "10"},
        {"id": "3", "imprintCode": "XYZ", "dosageMg": "5"},
    ]
    dupes = empty_db.duplicate_imprints()
    assert len(dupes) == 1
    assert {dupes[0][0]["id"], dupes[0][1]["id"]} == {"1", "2"}


def test_color_mismatch_detects_disagreement():
    row = {"colorName": "blue", "hue": 15, "saturation": 200, "value": 200}  # actually orange
    assert PillDatabase.color_mismatch(row) == "orange"


def test_color_mismatch_agrees_when_names_overlap():
    row = {"colorName": "gray, cyan speckles", "hue": 0, "saturation": 10, "value": 128}  # gray
    assert PillDatabase.color_mismatch(row) is None
