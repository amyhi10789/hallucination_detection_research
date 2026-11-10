from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


PACKAGES = (
    "datasets",
    "joblib",
    "matplotlib",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "torch",
    "transformers",
)


def collect_environment(repo_root: str | Path) -> dict[str, Any]:
    versions: dict[str, str] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"

    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": versions,
        "git_commit": _git_value(repo_root, ["rev-parse", "HEAD"]),
        "git_dirty": bool(_git_value(repo_root, ["status", "--porcelain"])),
    }
    try:
        import torch

        metadata["torch_runtime"] = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpu_names": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        metadata["torch_runtime"] = {"available": False}
    return metadata


def _git_value(repo_root: str | Path, arguments: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=Path(repo_root),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = completed.stdout.strip()
    return value or None

