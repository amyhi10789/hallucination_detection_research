from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from hallucination_detection.utils import read_json, sha256_file, write_json


def save_feature_bundle(
    directory: str | Path,
    features: dict[int, np.ndarray],
    row_ids: Iterable[str],
) -> dict:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    ids = [str(value) for value in row_ids]
    manifest: dict = {"row_ids": ids, "layers": {}}

    for layer, values in sorted(features.items()):
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"Layer {layer} features must be a 2D array, got {array.shape}.")
        if len(array) != len(ids):
            raise ValueError(
                f"Layer {layer} has {len(array)} rows but the record table has {len(ids)} rows."
            )
        filename = f"layer_{int(layer):03d}.npy"
        path = target / filename
        np.save(path, array, allow_pickle=False)
        manifest["layers"][str(int(layer))] = {
            "file": filename,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": sha256_file(path),
        }

    write_json(target / "manifest.json", manifest)
    return manifest


def load_feature_bundle(directory: str | Path) -> tuple[dict[int, np.ndarray], list[str]]:
    source = Path(directory)
    manifest = read_json(source / "manifest.json")
    features: dict[int, np.ndarray] = {}
    for layer, details in manifest["layers"].items():
        path = source / details["file"]
        if sha256_file(path) != details["sha256"]:
            raise ValueError(f"Feature file checksum does not match its manifest: {path}")
        array = np.load(path, allow_pickle=False)
        if list(array.shape) != details["shape"]:
            raise ValueError(f"Feature file shape does not match its manifest: {path}")
        features[int(layer)] = array
    return features, [str(value) for value in manifest["row_ids"]]

