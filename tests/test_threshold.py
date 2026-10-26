"""S8 threshold tests: cost math checked by hand, and the threshold falls as missed defaults get costlier."""
import numpy as np
import pytest

from risk.threshold import best_threshold, expected_cost

Y = np.array([1, 0, 1, 0, 0])
P = np.array([0.9, 0.8, 0.2, 0.1, 0.05])


def test_expected_cost_counts_by_hand():
    # Threshold 0.5 declines the loans scored 0.9 (a default: caught) and 0.8 (good: false decline).
    # The default scored 0.2 is approved, so it is missed.
    r = expected_cost(Y, P, threshold=0.5, ratio=10)
    assert r["defaults_caught"] == 1
    assert r["false_declines"] == 1
    assert r["missed_defaults"] == 1
    assert r["cost"] == 10 * 1 + 1
    assert r["decline_rate"] == pytest.approx(0.4)


def test_threshold_above_every_score_approves_everyone():
    r = expected_cost(Y, P, threshold=1.01, ratio=10)
    assert r["declined"] == 0
    assert r["missed_defaults"] == 2
    assert r["cost"] == 10 * 2


def test_higher_cost_ratio_never_raises_the_threshold():
    rng = np.random.default_rng(0)
    p = rng.random(5000) * 0.2
    y = (rng.random(5000) < p).astype(int)
    thresholds = [best_threshold(y, p, ratio) for ratio in [2, 10, 50]]
    assert thresholds[0] >= thresholds[1] >= thresholds[2]