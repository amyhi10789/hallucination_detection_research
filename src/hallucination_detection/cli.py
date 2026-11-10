from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from hallucination_detection.config import ConfigError, load_config
from hallucination_detection.metadata import collect_environment
from hallucination_detection.pipeline import run_experiment
from hallucination_detection.paraphrase_drafting import draft_paraphrases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hallucination-detection",
        description="Run leakage-safe hidden-state hallucination-detector experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="Validate a TOML profile.")
    validate.add_argument("--config", required=True, type=Path)

    doctor = subparsers.add_parser("doctor", help="Check configuration and local runtime.")
    doctor.add_argument("--config", type=Path, default=Path("configs/smoke.toml"))

    run = subparsers.add_parser("run", help="Execute inference, probes, analysis, and report.")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument(
        "--resume-run",
        type=Path,
        help="Existing run directory whose cached inference may be reused.",
    )

    draft = subparsers.add_parser(
        "draft-paraphrases",
        help="Generate unreviewed paraphrase candidates for human validation.",
    )
    draft.add_argument("--config", required=True, type=Path)
    draft.add_argument("--output", required=True, type=Path)
    draft.add_argument("--limit", type=int)
    draft.add_argument("--max-new-tokens", type=int, default=128)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate-config":
            config = load_config(args.config)
            print(f"Valid configuration: {args.config.resolve()}")
            print(f"Profile: {config.run.profile}; model: {config.model.model_id}")
            return 0
        if args.command == "doctor":
            return _doctor(args.config)
        if args.command == "run":
            run_dir = run_experiment(args.config, resume_run=args.resume_run)
            print(f"RESULTS_DIR={run_dir.resolve()}")
            print(f"REPORT={run_dir.resolve() / 'report.md'}")
            return 0
        if args.command == "draft-paraphrases":
            output = draft_paraphrases(
                args.config,
                args.output,
                limit=args.limit,
                max_new_tokens=args.max_new_tokens,
            )
            print(f"CANDIDATES={output}")
            print(
                "Review every row, change semantic_status to validated only when meaning and "
                "the correct answer are unchanged, then save reviewed rows as "
                "data/curated_paraphrases.csv."
            )
            return 0
    except (ConfigError, OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 1


def _doctor(config_path: Path) -> int:
    config = load_config(config_path)
    repo_root = config_path.resolve().parent.parent
    required = [
        "datasets",
        "joblib",
        "matplotlib",
        "numpy",
        "pandas",
        "sklearn",
        "torch",
        "transformers",
    ]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    environment = collect_environment(repo_root)
    print(json.dumps(environment, indent=2, default=str))
    print(f"\nConfiguration: {config_path.resolve()}")
    print(f"Dataset: {config.data.dataset_id}@{config.data.revision}")
    print(f"Model: {config.model.model_id}@{config.model.revision}")
    if config.paraphrases.mode == "curated":
        path = config.paraphrases.curated_path
        if path is not None and not path.is_absolute():
            path = repo_root / path
        row_count = _csv_data_rows(path) if path is not None else 0
        if row_count == 0:
            print(
                "WARNING: the full profile needs human-validated rows in "
                f"{path}; it will intentionally stop until they are supplied."
            )
    if missing:
        print(f"MISSING DEPENDENCIES: {', '.join(missing)}", file=sys.stderr)
        return 2
    print("Runtime check passed.")
    return 0


def _csv_data_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig") as handle:
        return max(sum(1 for line in handle if line.strip()) - 1, 0)


if __name__ == "__main__":
    raise SystemExit(main())

