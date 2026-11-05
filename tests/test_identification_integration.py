"""End-to-end checks against the real pill database shipped in data/ -
the strongest regression test: it exercises the exact pipeline this project
is about, not a synthetic stand-in."""
import pytest

from pillscope.config import SAMPLES_DIR, Settings
from pillscope.database import PillDatabase
from pillscope.io_utils import imread
from pillscope.pipeline import analyze, identify, verdict


@pytest.fixture(scope="module")
def db():
    database = PillDatabase()
    if not database.rows:
        pytest.skip("data/pill_database.csv is empty - run scripts/build_database.py first")
    database.ensure_features(Settings())
    return database


def test_a_known_pill_photographed_at_an_angle_is_identified(db):
    img = imread(SAMPLES_DIR / "sample_042_rotated30.jpg")
    matches = identify(analyze(img), db)
    kind, _ = verdict(matches)
    assert kind == "match"
    assert matches[0].row["name"] == "Pill 042"


def test_a_pill_left_out_of_the_database_is_not_matched(db):
    img = imread(SAMPLES_DIR / "sample_unknown_pill252_not_in_database.jpg")
    matches = identify(analyze(img), db)
    kind, _ = verdict(matches)
    assert kind != "match"
