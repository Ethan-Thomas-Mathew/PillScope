import pytest

from pillscope import pipeline
from pillscope.pipeline import Match


def test_weights_sum_to_one():
    assert sum(pipeline.WEIGHTS.values()) == pytest.approx(1.0)


def test_combine_with_perfect_clues():
    parts = dict.fromkeys(pipeline.WEIGHTS, 1.0)
    assert pipeline.combine(parts) == pytest.approx(1.0)


def test_combine_with_missing_clue_defaults_to_zero():
    parts = {"sift": 1.0}  # color/texture/hog missing
    assert pipeline.combine(parts) == pytest.approx(pipeline.WEIGHTS["sift"])


def test_verdict_on_empty_database():
    kind, sentence = pipeline.verdict([])
    assert kind == "unknown"
    assert "empty" in sentence.lower()


def test_verdict_confident_match():
    m = Match(row={"name": "Pill 001"}, score=0.9, parts={}, inliers=25)
    kind, sentence = pipeline.verdict([m])
    assert kind == "match"
    assert "Pill 001" in sentence


def test_verdict_possible_match():
    m = Match(row={"name": "Pill 002"}, score=0.5, parts={}, inliers=12)
    kind, _ = pipeline.verdict([m])
    assert kind == "possible"


def test_verdict_no_reliable_match():
    m = Match(row={"name": "Pill 003"}, score=0.2, parts={}, inliers=2)
    kind, _ = pipeline.verdict([m])
    assert kind == "unknown"


def test_analyze_runs_end_to_end(pill_photo):
    a = pipeline.analyze(pill_photo)
    assert a.mask.any()
    assert a.crop.shape[:2] == (192, 192)  # default work_size
    assert a.lab_hist.sum() == pytest.approx(1.0, abs=1e-3)
    assert len(a.kp_pts) == len(a.desc) == len(a.kp_size) == len(a.kp_angle)


def test_identify_on_an_empty_database_returns_nothing(pill_photo):
    class FakeDB:
        @staticmethod
        def entries():
            return []

    a = pipeline.analyze(pill_photo)
    assert pipeline.identify(a, FakeDB()) == []
