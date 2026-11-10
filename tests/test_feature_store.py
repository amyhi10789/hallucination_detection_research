import numpy as np
import pytest

from hallucination_detection.feature_store import load_feature_bundle, save_feature_bundle


def test_feature_bundle_round_trip(tmp_path):
    expected = {3: np.arange(12, dtype=np.float32).reshape(3, 4)}
    save_feature_bundle(tmp_path, expected, ["a", "b", "c"])

    actual, row_ids = load_feature_bundle(tmp_path)

    assert row_ids == ["a", "b", "c"]
    np.testing.assert_array_equal(actual[3], expected[3])


def test_feature_bundle_rejects_misaligned_rows(tmp_path):
    with pytest.raises(ValueError, match="record table"):
        save_feature_bundle(tmp_path, {1: np.zeros((2, 4))}, ["only-one"])

