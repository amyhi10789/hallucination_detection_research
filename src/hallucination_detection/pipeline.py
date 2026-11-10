from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd

from hallucination_detection.config import ExperimentConfig, load_config
from hallucination_detection.data import load_questions
from hallucination_detection.evaluation import (
    evaluate_standard_slices,
    paired_prediction_changes,
    summarize_paired_changes,
)
from hallucination_detection.feature_store import load_feature_bundle, save_feature_bundle
from hallucination_detection.inference import InferenceResult, make_inference_runner
from hallucination_detection.metadata import collect_environment
from hallucination_detection.paraphrases import build_variants
from hallucination_detection.probes import fit_probe_suite, predict_probe_suite, save_probe
from hallucination_detection.prompting import add_prompts
from hallucination_detection.reporting import generate_report
from hallucination_detection.splits import split_by_question, split_summary
from hallucination_detection.statistics import (
    bootstrap_change_statistics,
    directional_excess_volatility_test,
    stable_margin_analysis,
)
from hallucination_detection.utils import (
    read_json,
    sha256_file,
    slugify,
    utc_timestamp,
    write_json,
)


def run_experiment(
    config_path: str | Path,
    *,
    resume_run: str | Path | None = None,
) -> Path:
    source_config = Path(config_path).resolve()
    repo_root = source_config.parent.parent
    config = _resolve_paths(load_config(source_config), repo_root)
    run_dir = _create_run_directory(config, resume_run)
    fingerprint = _config_fingerprint(config)
    previous_manifest = run_dir / "manifest.json"
    if resume_run and previous_manifest.exists():
        previous = read_json(previous_manifest)
        previous_fingerprint = previous.get("config_fingerprint")
        if previous_fingerprint and previous_fingerprint != fingerprint:
            raise ValueError(
                "The resume directory was created with a different configuration. "
                "Use its original TOML profile or start a new run."
            )
    logger = _configure_logging(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "status": "running",
        "profile": config.run.profile,
        "config_source": str(source_config),
        "config_fingerprint": fingerprint,
        "stages": {},
    }
    write_json(manifest_path, manifest)
    write_json(run_dir / "config.resolved.json", asdict(config))
    shutil.copy2(source_config, run_dir / "config.source.toml")

    try:
        logger.info("Loading questions")
        questions = load_questions(config)
        data_dir = run_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        questions.to_csv(data_dir / "questions.csv", index=False)
        manifest["stages"]["questions"] = {"status": "complete", "rows": len(questions)}
        write_json(manifest_path, manifest)

        logger.info("Creating prompt variants")
        variants = add_prompts(build_variants(questions, config.paraphrases))
        variants = split_by_question(
            variants,
            train_fraction=config.split.train_fraction,
            validation_fraction=config.split.validation_fraction,
            test_fraction=config.split.test_fraction,
            heldout_subjects=config.split.heldout_subjects,
            seed=config.run.seed,
        )
        variants.to_csv(data_dir / "variants.csv", index=False)
        variants[["question_id", "subject", "split"]].drop_duplicates().to_csv(
            data_dir / "split_assignments.csv", index=False
        )
        split_summary(variants).to_csv(data_dir / "split_summary.csv", index=False)
        manifest["stages"]["variants"] = {
            "status": "complete",
            "rows": len(variants),
            "variants": sorted(variants["variant"].unique().tolist()),
        }
        write_json(manifest_path, manifest)

        inference = _load_or_run_inference(config, variants, run_dir, resume_run, logger)
        manifest["stages"]["inference"] = {
            "status": "complete",
            "rows": len(inference.records),
            "layers": sorted(inference.features),
            "parse_failures": int((~inference.records["parse_valid"].astype(bool)).sum()),
        }
        write_json(manifest_path, manifest)

        logger.info("Training canonical, augmented, and invariant probes")
        predictions = _train_and_predict(config, inference, run_dir)
        prediction_dir = run_dir / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(prediction_dir / "detector_predictions.csv", index=False)
        manifest["stages"]["probes"] = {
            "status": "complete",
            "prediction_rows": len(predictions),
            "methods": sorted(predictions["method"].unique().tolist()),
        }
        write_json(manifest_path, manifest)

        logger.info("Computing metrics and clustered-bootstrap comparisons")
        analysis = _analyze(config, inference.records, predictions, run_dir)
        manifest["stages"]["analysis"] = {
            "status": "complete",
            "hypothesis": analysis["hypothesis"],
        }
        write_json(manifest_path, manifest)

        environment = collect_environment(repo_root)
        run_metadata = {
            "run_name": config.run.name,
            "profile": config.run.profile,
            "seed": config.run.seed,
            "dataset": asdict(config.data),
            "model": inference.model_metadata,
            "environment": environment,
        }
        write_json(run_dir / "run_metadata.json", run_metadata)

        report_profile = (
            config.run.profile
            if config.run.profile in {"smoke", "pilot", "full"}
            else "smoke"
        )
        limitations = [
            "Deterministic generation gives one binary outcome per question variant; error rates are aggregate quantities across questions.",
            "The invariant-projection probe is a proposed method and must be compared on the same held-out questions as both baselines.",
        ]
        if config.paraphrases.mode == "templates":
            limitations.append(
                "The automatic variants change instruction wrappers while leaving the question stem unchanged; they are prompt-format controls, not full semantic paraphrases."
            )
        generate_report(
            run_dir,
            profile=report_profile,
            predictions=predictions,
            tracking=analysis["change_intervals"],
            layer_metrics=analysis["report_layer_metrics"],
            drift_metrics=analysis["change_summary"],
            calibration=analysis["calibration"],
            summary={
                "hypothesis": analysis["report_hypothesis"],
                "metrics": analysis["headline_metrics"],
            },
            metadata=run_metadata,
            limitations=limitations,
        )

        manifest["status"] = "complete"
        manifest["artifacts"] = _artifact_manifest(run_dir)
        write_json(manifest_path, manifest)
        (Path(config.run.output_root) / "LATEST.txt").write_text(
            str(run_dir.resolve()) + "\n", encoding="utf-8"
        )
        logger.info("Run complete: %s", run_dir)
        return run_dir
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_json(manifest_path, manifest)
        logger.exception("Experiment failed")
        raise


def _resolve_paths(config: ExperimentConfig, repo_root: Path) -> ExperimentConfig:
    output_root = config.run.output_root
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    curated_path = config.paraphrases.curated_path
    if curated_path is not None and not curated_path.is_absolute():
        curated_path = repo_root / curated_path
    cache_dir = config.data.cache_dir
    if cache_dir is not None and not cache_dir.is_absolute():
        cache_dir = repo_root / cache_dir
    return replace(
        config,
        run=replace(config.run, output_root=output_root),
        data=replace(config.data, cache_dir=cache_dir),
        paraphrases=replace(config.paraphrases, curated_path=curated_path),
    )


def _create_run_directory(config: ExperimentConfig, resume_run: str | Path | None) -> Path:
    if resume_run is not None:
        run_dir = Path(resume_run).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {run_dir}")
        return run_dir
    output_root = Path(config.run.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    base = output_root / f"{utc_timestamp()}_{slugify(config.run.name)}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}_{suffix}")
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _configure_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"hallucination_detection.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def _load_or_run_inference(
    config: ExperimentConfig,
    variants: pd.DataFrame,
    run_dir: Path,
    resume_run: str | Path | None,
    logger: logging.Logger,
) -> InferenceResult:
    inference_dir = run_dir / "inference"
    records_path = inference_dir / "responses.csv"
    feature_dir = inference_dir / "features"
    metadata_path = inference_dir / "model_metadata.json"
    if resume_run and records_path.exists() and (feature_dir / "manifest.json").exists():
        logger.info("Reusing cached inference artifacts")
        records = pd.read_csv(records_path)
        features, row_ids = load_feature_bundle(feature_dir)
        if records["variant_id"].astype(str).tolist() != row_ids:
            raise ValueError("Cached features are not aligned with cached response rows.")
        metadata = read_json(metadata_path) if metadata_path.exists() else {"cached": True}
        return InferenceResult(records=records, features=features, model_metadata=metadata)

    runner = make_inference_runner(config)
    try:
        result = runner.run(variants)
    finally:
        close = getattr(runner, "close", None)
        if callable(close):
            close()
    inference_dir.mkdir(parents=True, exist_ok=True)
    result.records.to_csv(records_path, index=False)
    save_feature_bundle(feature_dir, result.features, result.records["variant_id"])
    write_json(metadata_path, result.model_metadata)
    return result


def _train_and_predict(
    config: ExperimentConfig,
    inference: InferenceResult,
    run_dir: Path,
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for layer, feature_values in sorted(inference.features.items()):
        frame = inference.records.copy()
        frame["hidden_state"] = list(feature_values)
        train = frame.loc[frame["split"] == "train"].copy()
        if train.empty:
            raise ValueError("No training questions were assigned; increase the sample size.")
        probes = fit_probe_suite(
            train,
            explained_variance=config.probe.invariant_variance,
            max_components=config.probe.max_invariant_components,
            C=config.probe.c,
            class_weight=config.probe.class_weight,
            max_iter=config.probe.max_iter,
            seed=config.run.seed,
        )
        predicted = predict_probe_suite(frame, probes)
        predicted["layer"] = int(layer)
        predicted["model"] = config.model.model_id
        predicted["seed"] = int(config.run.seed)
        predicted["parse_success"] = predicted["parse_valid"].astype(bool)
        predicted = predicted.drop(columns=["hidden_state"])
        outputs.append(predicted)
        if config.output.save_models:
            for method, probe in probes.items():
                save_probe(
                    probe,
                    run_dir / "models" / f"layer_{layer:03d}" / f"{method}.joblib",
                )
    return pd.concat(outputs, ignore_index=True)


def _analyze(
    config: ExperimentConfig,
    generations: pd.DataFrame,
    predictions: pd.DataFrame,
    run_dir: Path,
) -> dict[str, Any]:
    tables = run_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_standard_slices(
        predictions, n_bins=config.evaluation.calibration_bins
    )
    metrics.to_csv(tables / "all_metrics.csv", index=False)

    paired = paired_prediction_changes(predictions)
    paired.to_csv(tables / "question_level_changes.csv", index=False)
    change_summary = summarize_paired_changes(paired)
    change_summary.to_csv(tables / "change_summary.csv", index=False)

    analysis_pairs = paired.loc[paired["split"].isin(["test", "heldout"])].copy()
    if analysis_pairs.empty:
        analysis_pairs = paired.copy()
    bootstrap_groups = ["model", "seed", "layer", "method", "split", "variant"]
    change_intervals = bootstrap_change_statistics(
        analysis_pairs,
        group_cols=bootstrap_groups,
        n_resamples=config.evaluation.bootstrap_replicates,
        confidence_level=config.evaluation.confidence_level,
        seed=config.run.seed,
    )
    change_intervals.to_csv(tables / "change_intervals.csv", index=False)

    primary_layer = _resolve_primary_layer(
        config.evaluation.primary_layer,
        sorted(int(value) for value in predictions["layer"].unique()),
    )
    primary_pairs = analysis_pairs.loc[analysis_pairs["layer"] == primary_layer].copy()
    stable = stable_margin_analysis(
        primary_pairs,
        contrast_cols=["method", "split", "variant"],
        margin=config.evaluation.stable_error_margin,
        n_resamples=config.evaluation.bootstrap_replicates,
        seed=config.run.seed,
    )
    stable.to_csv(tables / "stable_error_analysis.csv", index=False)

    baseline_pairs = primary_pairs.loc[
        (primary_pairs["method"] == "canonical_logistic")
        & (primary_pairs["split"] == "test")
    ].copy()
    if baseline_pairs.empty:
        baseline_pairs = primary_pairs.loc[
            primary_pairs["method"] == "canonical_logistic"
        ].copy()
    hypothesis = directional_excess_volatility_test(
        baseline_pairs,
        contrast_cols=["subject", "variant"],
        n_resamples=config.evaluation.bootstrap_replicates,
        confidence_level=config.evaluation.confidence_level,
        seed=config.run.seed,
    )
    write_json(tables / "hypothesis_test.json", hypothesis)

    report_layer_metrics = metrics.loc[
        (metrics["scope"] == "overall")
        & (metrics["method"] == "canonical_logistic")
        & (metrics["split"] == "test")
    ].copy()
    if report_layer_metrics.empty:
        report_layer_metrics = metrics.loc[
            (metrics["scope"] == "overall")
            & (metrics["method"] == "canonical_logistic")
            & (metrics["split"] == "heldout")
        ].copy()
    if report_layer_metrics.empty:
        report_layer_metrics = metrics.loc[
            (metrics["scope"] == "overall")
            & (metrics["method"] == "canonical_logistic")
        ].copy()

    calibration_rows = predictions.loc[
        (predictions["layer"] == primary_layer)
        & (predictions["method"] == "canonical_logistic")
        & (predictions["split"] == "test")
    ].copy()
    if calibration_rows.empty:
        calibration_rows = predictions.loc[
            (predictions["layer"] == primary_layer)
            & (predictions["method"] == "canonical_logistic")
            & (predictions["split"] == "heldout")
        ].copy()
    calibration = _calibration_table(
        calibration_rows,
        bins=config.evaluation.calibration_bins,
    )

    generation_slice = generations.loc[
        (generations["variant"] == "canonical")
        & generations["split"].isin(["test", "heldout"])
    ]
    primary_metric_slice = report_layer_metrics.loc[
        report_layer_metrics["layer"] == primary_layer
    ]
    headline_metrics = {
        "canonical_llm_error_rate": (
            float(generation_slice["is_error"].mean()) if len(generation_slice) else None
        ),
        "parse_failure_rate": float((~generations["parse_valid"].astype(bool)).mean()),
        "primary_layer": primary_layer,
        "baseline_test_auroc": _first_finite(primary_metric_slice.get("auroc")),
        "baseline_test_brier": _first_finite(primary_metric_slice.get("brier")),
        "mean_excess_volatility": hypothesis["mean_excess_volatility"],
    }
    primary_contrasts = summarize_paired_changes(
        baseline_pairs,
        group_cols=["subject", "variant"],
    )
    contrast_gaps = (
        primary_contrasts["detector_score_change"]
        - primary_contrasts["actual_error_rate_change"]
    )
    report_hypothesis = {
        "supported": hypothesis["supports_hypothesis"],
        **hypothesis,
        "effect_size": hypothesis["mean_excess_volatility"],
        "pair_count": int(len(baseline_pairs)),
        "mean_actual_change": float(primary_contrasts["actual_error_rate_change"].mean()),
        "mean_detector_change": float(primary_contrasts["detector_score_change"].mean()),
        "mean_excess_detector_change": float(contrast_gaps.mean()),
        "median_excess_detector_change": float(contrast_gaps.median()),
        "fraction_pairs_detector_changed_more": float((contrast_gaps > 0).mean()),
    }
    return {
        "metrics": metrics,
        "paired": paired,
        "change_summary": change_summary,
        "change_intervals": change_intervals,
        "stable": stable,
        "hypothesis": hypothesis,
        "report_hypothesis": report_hypothesis,
        "report_layer_metrics": report_layer_metrics,
        "calibration": calibration,
        "headline_metrics": headline_metrics,
    }


def _first_finite(series: Any) -> float | None:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.iloc[0]) if len(values) else None


def _resolve_primary_layer(spec: int | float, available: list[int]) -> int:
    if not available:
        raise ValueError("No hidden-state layers are available for analysis.")
    if isinstance(spec, float):
        requested = max(1, int(round(spec * max(available))))
    else:
        requested = int(spec)
    if requested not in available:
        raise ValueError(
            f"Primary layer {requested} was not extracted; available layers are {available}."
        )
    return requested


def _config_fingerprint(config: ExperimentConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _calibration_table(predictions: pd.DataFrame, *, bins: int) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=["bin_lower", "bin_upper", "mean_predicted_error", "observed_error_rate", "count"]
        )
    frame = predictions[["error_probability", "is_error"]].copy()
    frame["bin"] = pd.cut(
        frame["error_probability"],
        bins=[index / bins for index in range(bins + 1)],
        include_lowest=True,
    )
    grouped = frame.groupby("bin", observed=True).agg(
        mean_predicted_error=("error_probability", "mean"),
        observed_error_rate=("is_error", "mean"),
        count=("is_error", "size"),
    )
    result = grouped.reset_index()
    result.insert(0, "bin_lower", result["bin"].map(lambda value: float(value.left)))
    result.insert(1, "bin_upper", result["bin"].map(lambda value: float(value.right)))
    return result.drop(columns="bin")


def _artifact_manifest(run_dir: Path) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(run_dir).as_posix()
        details: dict[str, Any] = {"bytes": path.stat().st_size}
        if path.stat().st_size <= 100 * 1024 * 1024:
            details["sha256"] = sha256_file(path)
        artifacts[relative] = details
    return artifacts

