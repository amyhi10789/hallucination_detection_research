from hallucination_detection.utils import read_json, slugify, stable_int, write_json


def test_stable_int_is_deterministic():
    assert stable_int("question", 7) == stable_int("question", 7)
    assert stable_int("question", 7) != stable_int("question", 8)


def test_json_helpers_and_slug(tmp_path):
    path = write_json(tmp_path / "nested" / "value.json", {"ok": True})
    assert read_json(path) == {"ok": True}
    assert slugify("Pilot Run #1") == "pilot-run-1"

