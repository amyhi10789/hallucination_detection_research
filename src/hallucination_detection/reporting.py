"""Create human-readable and machine-readable experiment reports.

The reporting layer intentionally depends only on pandas, NumPy, and
matplotlib.  It accepts already-computed result frames so that statistical
analysis remains separate from presentation and can be tested independently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


_PROFILES = {"smoke", "pilot", "full"}
_ACTUAL_CHANGE_COLUMNS = (
    "actual_error_rate_change",
    "actual_change",
    "error_rate_change",
    "absolute_error_change",
    "delta_error",
)
_DETECTOR_CHANGE_COLUMNS = (
    "detector_score_change",
    "detector_change",
    "prediction_change",
    "absolute_detector_change",
    "absolute_prediction_change",
    "delta_detector",
)
_SCORE_COLUMNS = (
    "predicted_error_probability",
    "detector_score",
    "error_probability",
    "y_prob",
    "probability",
    "score",
)
_LABEL_COLUMNS = ("is_error", "incorrect", "error", "y_true", "label")


@dataclass(frozen=True)
class ReportArtifacts:
    """Paths produced by :func:`generate_report`."""

    report: Path
    summary: Path
    tables: Mapping[str, Path]
    figures: Mapping[str, Path]

    def as_dict(self) -> dict[str, Any]:
        return {
            "report": self.report,
            "summary": self.summary,
            "tables": dict(self.tables),
            "figures": dict(self.figures),
        }


def _first_column(frame: pd.DataFrame | None, candidates: Sequence[str]) -> str | None:
    if frame is None:
        return None
    return next((name for name in candidates if name in frame.columns), None)


def _ordered_columns(frame: pd.DataFrame) -> pd.DataFrame:
    preferred = (
        "question_id",
        "pair_id",
        "subject",
        "split",
        "variant",
        "paraphrase_id",
        "method",
        "layer",
        "layer_index",
        "gold_answer",
        "predicted_answer",
        "parse_success",
        "is_error",
        "detector_score",
        "predicted_error_probability",
        "actual_error_rate_change",
        "detector_score_change",
    )
    first = [name for name in preferred if name in frame.columns]
    remaining = [name for name in frame.columns if name not in first]
    return frame.loc[:, first + remaining]


def _write_table(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ordered_columns(frame).to_csv(path, index=False, float_format="%.8g")


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="both", color="#D9DEE8", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _tracking_figure(frame: pd.DataFrame, path: Path) -> bool:
    actual = _first_column(frame, _ACTUAL_CHANGE_COLUMNS)
    detector = _first_column(frame, _DETECTOR_CHANGE_COLUMNS)
    if actual is None or detector is None:
        return False

    method = _first_column(frame, ("method", "detector", "training_method"))
    columns = [actual, detector] + ([method] if method else [])
    plot = frame.loc[:, columns].copy()
    plot[actual] = pd.to_numeric(plot[actual], errors="coerce")
    plot[detector] = pd.to_numeric(plot[detector], errors="coerce")
    plot = plot.dropna(subset=[actual, detector])
    if plot.empty:
        return False

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    if method:
        for name, group in plot.groupby(method, dropna=False, sort=True):
            ax.scatter(
                group[actual],
                group[detector],
                s=42,
                alpha=0.72,
                edgecolors="white",
                linewidths=0.45,
                label=str(name),
            )
    else:
        ax.scatter(
            plot[actual],
            plot[detector],
            s=42,
            alpha=0.72,
            color="#3568B8",
            edgecolors="white",
            linewidths=0.45,
        )

    maximum = float(np.nanmax(plot[[actual, detector]].to_numpy()))
    limit = max(maximum * 1.08, 0.05)
    ax.plot([0, limit], [0, limit], linestyle="--", color="#555B66", label="Perfect tracking")
    if method and plot[method].nunique(dropna=False) > 1:
        ax.legend(title="Method", frameon=False)
    else:
        ax.legend(frameon=False)
    ax.set_xlim(left=0, right=limit)
    ax.set_ylim(bottom=0, top=limit)
    ax.set_xlabel("Absolute change in observed LLM error rate")
    ax.set_ylabel("Absolute change in detector score")
    ax.set_title("Do detector changes track actual error-rate changes?")
    _style_axes(ax)
    _save_figure(fig, path)
    return True


def _layer_figure(frame: pd.DataFrame, path: Path) -> bool:
    layer = _first_column(frame, ("layer", "layer_index", "layer_fraction"))
    if layer is None:
        return False

    metric_candidates = (
        "auroc",
        "auprc",
        "balanced_accuracy",
        "accuracy",
        "f1",
        "brier",
        "ece",
    )
    metrics = [name for name in metric_candidates if name in frame.columns]
    if not metrics:
        return False

    plot = frame.loc[:, [layer, *metrics]].copy()
    for name in metrics:
        plot[name] = pd.to_numeric(plot[name], errors="coerce")
    numeric_layer = pd.to_numeric(plot[layer], errors="coerce")
    if numeric_layer.notna().all():
        plot["_x"] = numeric_layer
        plot = plot.sort_values("_x")
        tick_labels = None
    else:
        categories = list(dict.fromkeys(plot[layer].astype(str)))
        category_map = {value: index for index, value in enumerate(categories)}
        plot["_x"] = plot[layer].astype(str).map(category_map)
        tick_labels = categories

    fig, ax = plt.subplots(figsize=(7.6, 5.1))
    plotted = 0
    for metric in metrics:
        valid = plot.dropna(subset=["_x", metric])
        if valid.empty:
            continue
        ax.plot(valid["_x"], valid[metric], marker="o", linewidth=2, label=metric.upper())
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return False
    if tick_labels is not None:
        ax.set_xticks(range(len(tick_labels)), tick_labels, rotation=30, ha="right")
    values = plot[metrics].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size and finite.min() >= 0 and finite.max() <= 1:
        ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Metric value")
    ax.set_title("Detector performance by hidden-state layer")
    ax.legend(frameon=False, ncols=min(plotted, 3))
    _style_axes(ax)
    _save_figure(fig, path)
    return True


def _drift_figure(frame: pd.DataFrame, path: Path) -> bool:
    actual = _first_column(frame, _ACTUAL_CHANGE_COLUMNS)
    detector = _first_column(frame, _DETECTOR_CHANGE_COLUMNS)
    if actual is None or detector is None:
        return False
    method = _first_column(frame, ("method", "detector", "training_method"))

    plot = frame.loc[:, [actual, detector] + ([method] if method else [])].copy()
    plot[actual] = pd.to_numeric(plot[actual], errors="coerce")
    plot[detector] = pd.to_numeric(plot[detector], errors="coerce")
    plot = plot.dropna(subset=[actual, detector])
    if plot.empty:
        return False
    if method is None:
        plot["_method"] = "All detector pairs"
        method = "_method"
    grouped = plot.groupby(method, dropna=False)[[actual, detector]].mean().sort_index()

    x = np.arange(len(grouped))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7.0, 1.15 * len(grouped) + 3.0), 5.0))
    ax.bar(x - width / 2, grouped[actual], width, label="Actual error-rate change", color="#5B8FD1")
    ax.bar(x + width / 2, grouped[detector], width, label="Detector-score change", color="#E38B5B")
    ax.set_xticks(x, [str(value) for value in grouped.index], rotation=25, ha="right")
    ax.set_ylabel("Mean absolute change")
    ax.set_title("Observed performance change vs. detector drift")
    ax.legend(frameon=False)
    _style_axes(ax)
    _save_figure(fig, path)
    return True


def _derive_calibration(predictions: pd.DataFrame | None) -> pd.DataFrame | None:
    score = _first_column(predictions, _SCORE_COLUMNS)
    label = _first_column(predictions, _LABEL_COLUMNS)
    if predictions is None or score is None or label is None:
        return None
    frame = predictions.loc[:, [score, label]].copy()
    frame[score] = pd.to_numeric(frame[score], errors="coerce")
    frame[label] = pd.to_numeric(frame[label], errors="coerce")
    frame = frame.dropna()
    frame = frame.loc[frame[score].between(0, 1) & frame[label].between(0, 1)]
    if frame.empty:
        return None

    edges = np.linspace(0.0, 1.0, 11)
    frame["_bin"] = pd.cut(frame[score], bins=edges, include_lowest=True, duplicates="drop")
    grouped = frame.groupby("_bin", observed=True)
    result = grouped.agg(
        mean_predicted_error=(score, "mean"),
        observed_error_rate=(label, "mean"),
        count=(label, "size"),
    ).reset_index()
    result.insert(0, "bin_lower", result["_bin"].map(lambda value: float(value.left)))
    result.insert(1, "bin_upper", result["_bin"].map(lambda value: float(value.right)))
    return result.drop(columns="_bin")


def _calibration_figure(frame: pd.DataFrame, path: Path) -> bool:
    predicted = _first_column(
        frame,
        ("mean_predicted_error", "mean_prediction", "predicted_probability", "detector_score"),
    )
    observed = _first_column(
        frame,
        ("observed_error_rate", "fraction_positive", "observed_rate", "actual_rate"),
    )
    if predicted is None or observed is None:
        return False
    count = _first_column(frame, ("count", "n", "sample_size"))
    columns = [predicted, observed] + ([count] if count else [])
    plot = frame.loc[:, columns].copy()
    plot[predicted] = pd.to_numeric(plot[predicted], errors="coerce")
    plot[observed] = pd.to_numeric(plot[observed], errors="coerce")
    plot = plot.dropna(subset=[predicted, observed]).sort_values(predicted)
    if plot.empty:
        return False
    if count:
        sizes = 28 + 18 * np.sqrt(pd.to_numeric(plot[count], errors="coerce").fillna(1).clip(lower=1))
    else:
        sizes = 60

    fig, ax = plt.subplots(figsize=(6.2, 5.3))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#555B66", label="Perfect calibration")
    ax.plot(plot[predicted], plot[observed], color="#3568B8", linewidth=1.8)
    ax.scatter(
        plot[predicted],
        plot[observed],
        s=sizes,
        color="#3568B8",
        alpha=0.8,
        edgecolors="white",
        linewidths=0.5,
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Mean predicted probability of error")
    ax.set_ylabel("Observed error rate")
    ax.set_title("Detector calibration")
    ax.legend(frameon=False)
    _style_axes(ax)
    _save_figure(fig, path)
    return True


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    numeric = pd.to_numeric(series, errors="coerce")
    mapped = series.astype("string").str.strip().str.lower().map(
        {"true": True, "yes": True, "valid": True, "false": False, "no": False, "invalid": False}
    )
    return mapped.fillna(numeric.fillna(0).astype(bool)).astype(bool)


def _parse_failure_count(predictions: pd.DataFrame | None) -> tuple[int | None, float | None]:
    if predictions is None or predictions.empty:
        return None, None
    # Detector predictions repeat each generation across layers and methods.
    # Count parse failures once per generated response when an ID is available.
    if "variant_id" in predictions:
        predictions = predictions.drop_duplicates("variant_id")
    if "parse_success" in predictions:
        failures = ~_bool_series(predictions["parse_success"])
    elif "parse_failure" in predictions:
        failures = _bool_series(predictions["parse_failure"])
    else:
        parsed = _first_column(predictions, ("parsed_answer", "predicted_answer", "parsed_choice"))
        if parsed is None:
            return None, None
        values = predictions[parsed].astype("string").str.strip().str.upper()
        failures = predictions[parsed].isna() | ~values.isin(list("ABCD"))
    count = int(failures.sum())
    return count, count / len(predictions)


def _sample_sizes(
    predictions: pd.DataFrame | None,
    tracking: pd.DataFrame | None,
    layer_metrics: pd.DataFrame | None,
) -> dict[str, int]:
    sizes: dict[str, int] = {}
    if predictions is not None:
        sizes["prediction_rows"] = int(len(predictions))
        if "variant_id" in predictions:
            sizes["generation_rows"] = int(predictions["variant_id"].nunique())
        for output_name, candidates in (
            ("questions", ("question_id", "base_question_id", "group_id")),
            ("subjects", ("subject", "domain")),
            ("variants", ("variant", "paraphrase_id", "prompt_variant")),
        ):
            column = _first_column(predictions, candidates)
            if column:
                sizes[output_name] = int(predictions[column].nunique(dropna=True))
    if tracking is not None:
        sizes["tracking_pairs"] = int(len(tracking))
    if layer_metrics is not None:
        layer = _first_column(layer_metrics, ("layer", "layer_index", "layer_fraction"))
        if layer:
            sizes["layers_evaluated"] = int(layer_metrics[layer].nunique(dropna=True))
    return sizes


def _hypothesis_result(
    tracking: pd.DataFrame | None,
    supplied: Mapping[str, Any],
) -> dict[str, Any]:
    provided = supplied.get("hypothesis")
    result: dict[str, Any] = dict(provided) if isinstance(provided, Mapping) else {}
    if "supported" not in result and "hypothesis_supported" in supplied:
        result["supported"] = supplied["hypothesis_supported"]
    if "supported" not in result and "supports_hypothesis" in result:
        result["supported"] = result["supports_hypothesis"]
    if "supported" not in result and "supports_hypothesis" in supplied:
        result["supported"] = supplied["supports_hypothesis"]

    actual = _first_column(tracking, _ACTUAL_CHANGE_COLUMNS)
    detector = _first_column(tracking, _DETECTOR_CHANGE_COLUMNS)
    if tracking is not None and actual and detector:
        pair = tracking.loc[:, [actual, detector]].apply(pd.to_numeric, errors="coerce").dropna()
        if not pair.empty:
            gaps = pair[detector] - pair[actual]
            result.setdefault("pair_count", int(len(pair)))
            result.setdefault("mean_actual_change", float(pair[actual].mean()))
            result.setdefault("mean_detector_change", float(pair[detector].mean()))
            result.setdefault("mean_excess_detector_change", float(gaps.mean()))
            result.setdefault("median_excess_detector_change", float(gaps.median()))
            result.setdefault("fraction_pairs_detector_changed_more", float((gaps > 0).mean()))

    supported = result.get("supported")
    if supported is True:
        result.setdefault("status", "supported")
        result.setdefault("statement", "The prespecified analysis supports the hypothesis.")
    elif supported is False:
        result.setdefault("status", "not_supported")
        result.setdefault("statement", "The prespecified analysis does not support the hypothesis.")
    elif "mean_excess_detector_change" in result:
        consistent = result["mean_excess_detector_change"] > 0
        result.setdefault("status", "descriptively_consistent" if consistent else "descriptively_inconsistent")
        result.setdefault(
            "statement",
            (
                "The descriptive result is consistent with the hypothesis, but no confirmatory "
                "test result was supplied."
                if consistent
                else "The descriptive result is not consistent with the hypothesis; no confirmatory test result was supplied."
            ),
        )
    else:
        result.setdefault("status", "not_evaluated")
        result.setdefault("statement", "The available artifacts do not evaluate the hypothesis.")
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
        return None
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "Not available"
        return f"{value:.4f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(rows: Sequence[tuple[str, Any]]) -> list[str]:
    output = ["| Item | Value |", "|---|---:|"]
    output.extend(f"| {label} | {_fmt(value)} |" for label, value in rows)
    return output


def _profile_note(profile: str) -> str:
    return {
        "smoke": "Pipeline-validation run only. Do not treat smoke-run estimates as scientific evidence.",
        "pilot": "Preliminary run intended to refine the protocol and estimate variability before the full study.",
        "full": "Full configured study run. Interpret results together with the limitations and run metadata below.",
    }[profile]


def _build_report(
    *,
    profile: str,
    generated_at: str,
    sample_sizes: Mapping[str, int],
    parse_failures: int | None,
    parse_failure_rate: float | None,
    hypothesis: Mapping[str, Any],
    summary: Mapping[str, Any],
    metadata: Mapping[str, Any],
    limitations: Sequence[str],
    table_paths: Mapping[str, Path],
    figure_paths: Mapping[str, Path],
    run_dir: Path,
) -> str:
    lines = [
        f"# Hallucination-Detector Experiment Report — {profile.upper()}",
        "",
        f"> **Run classification:** {_profile_note(profile)}",
        "",
        f"Generated at `{generated_at}`.",
        "",
        "## Research Question",
        "",
        "How accurately do pre-generation hidden-state hallucination detectors track changes in LLM "
        "error rates caused by meaning-preserving question paraphrases, especially when actual error "
        "rates remain stable?",
        "",
        "## Run Overview",
        "",
    ]
    overview_rows: list[tuple[str, Any]] = [("Profile", profile.upper())]
    overview_rows.extend((name.replace("_", " ").title(), value) for name, value in sample_sizes.items())
    overview_rows.extend(
        [
            ("Parse failures", parse_failures),
            ("Parse failure rate", parse_failure_rate),
        ]
    )
    lines.extend(_markdown_table(overview_rows))

    lines.extend(["", "## Hypothesis Result", "", str(hypothesis.get("statement", "Not evaluated.")), ""])
    evidence_keys = (
        "pair_count",
        "mean_actual_change",
        "mean_detector_change",
        "mean_excess_detector_change",
        "median_excess_detector_change",
        "fraction_pairs_detector_changed_more",
        "mean_excess_volatility",
        "one_sided_lower_bound",
        "bootstrap_probability_nonpositive",
        "effect_size",
        "p_value",
        "ci_low",
        "ci_high",
    )
    evidence = [(key.replace("_", " ").title(), hypothesis[key]) for key in evidence_keys if key in hypothesis]
    if evidence:
        lines.extend(_markdown_table(evidence))
    if hypothesis.get("supported") is None:
        lines.extend(
            [
                "",
                "This label is descriptive unless the summary explicitly supplies a prespecified "
                "statistical test and `supported` decision.",
            ]
        )

    metrics = summary.get("metrics")
    if isinstance(metrics, Mapping) and metrics:
        lines.extend(["", "## Key Metrics", ""])
        lines.extend(_markdown_table([(str(key), value) for key, value in metrics.items()]))

    if figure_paths:
        lines.extend(["", "## Figures", ""])
        figure_titles = {
            "tracking_scatter": "Detector tracking",
            "layer_metrics": "Layer comparison",
            "detector_drift_by_method": "Detector drift by method",
            "calibration": "Calibration",
        }
        for name, path in figure_paths.items():
            relative = path.relative_to(run_dir).as_posix()
            title = figure_titles.get(name, name.replace("_", " ").title())
            lines.extend([f"### {title}", "", f"![{title}]({relative})", ""])

    lines.extend(["## Data Quality", ""])
    if parse_failures is None:
        lines.append("Parse-failure information was not available in the prediction table.")
    else:
        lines.append(
            f"{parse_failures} of {sample_sizes.get('generation_rows', sample_sizes.get('prediction_rows', 0))} generated responses "
            f"({parse_failure_rate:.2%}) could not be parsed into a valid answer. Parse failures must "
            "remain visible in downstream accuracy and robustness analyses."
        )

    lines.extend(["", "## Reproducibility Metadata", ""])
    if metadata:
        lines.extend(_markdown_table([(str(key), _json_safe(value)) for key, value in metadata.items()]))
    else:
        lines.append("No run metadata was supplied.")

    lines.extend(["", "## Artifacts", "", "### Tables", ""])
    if table_paths:
        for name, path in table_paths.items():
            relative = path.relative_to(run_dir).as_posix()
            lines.append(f"- [{name.replace('_', ' ').title()}]({relative})")
    else:
        lines.append("- No tabular artifacts were available.")
    lines.extend(["", "### Figures", ""])
    if figure_paths:
        for name, path in figure_paths.items():
            relative = path.relative_to(run_dir).as_posix()
            lines.append(f"- [{name.replace('_', ' ').title()}]({relative})")
    else:
        lines.append("- No figures could be generated from the available columns.")
    lines.extend(["", "- [Machine-readable summary](summary.json)", "", "## Limitations", ""])
    lines.extend(f"- {item}" for item in limitations)
    lines.extend(
        [
            "",
            "## Interpretation Reminder",
            "",
            "Detector-score movement is not itself proof of model uncertainty. Conclusions should be "
            "based on the paired change analysis, uncertainty estimates, held-out evaluations, and "
            "documented paraphrase-validity checks.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_report(
    run_dir: str | Path,
    *,
    profile: str,
    predictions: pd.DataFrame | None = None,
    tracking: pd.DataFrame | None = None,
    layer_metrics: pd.DataFrame | None = None,
    drift_metrics: pd.DataFrame | None = None,
    calibration: pd.DataFrame | None = None,
    summary: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    limitations: Sequence[str] | None = None,
) -> ReportArtifacts:
    """Write tables, figures, ``summary.json``, and a Markdown report.

    Parameters are deliberately dataframe-based.  The function recognizes the
    canonical column names documented by the project as well as a small set of
    common aliases.  Missing optional inputs simply omit the corresponding
    figure rather than failing the full run.
    """

    normalized_profile = profile.strip().lower()
    if normalized_profile not in _PROFILES:
        raise ValueError(f"profile must be one of {sorted(_PROFILES)}, got {profile!r}")

    root = Path(run_dir).resolve()
    table_dir = root / "tables"
    figure_dir = root / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, Path] = {}
    supplied_frames = {
        "predictions": predictions,
        "tracking_pairs": tracking,
        "layer_metrics": layer_metrics,
        "drift_metrics": drift_metrics,
    }
    for name, frame in supplied_frames.items():
        if frame is not None:
            path = table_dir / f"{name}.csv"
            _write_table(path, frame)
            tables[name] = path

    calibration_frame = calibration if calibration is not None else _derive_calibration(predictions)
    if calibration_frame is not None:
        calibration_path = table_dir / "calibration.csv"
        _write_table(calibration_path, calibration_frame)
        tables["calibration"] = calibration_path

    figures: dict[str, Path] = {}
    if tracking is not None:
        path = figure_dir / "tracking_scatter.png"
        if _tracking_figure(tracking, path):
            figures["tracking_scatter"] = path
    if layer_metrics is not None:
        path = figure_dir / "layer_metrics.png"
        if _layer_figure(layer_metrics, path):
            figures["layer_metrics"] = path
    drift_source = drift_metrics if drift_metrics is not None else tracking
    if drift_source is not None:
        path = figure_dir / "detector_drift_by_method.png"
        if _drift_figure(drift_source, path):
            figures["detector_drift_by_method"] = path
    if calibration_frame is not None:
        path = figure_dir / "calibration.png"
        if _calibration_figure(calibration_frame, path):
            figures["calibration"] = path

    supplied_summary = dict(summary or {})
    run_metadata = dict(metadata or {})
    sizes = _sample_sizes(predictions, tracking, layer_metrics)
    parse_count, parse_rate = _parse_failure_count(predictions)
    hypothesis = _hypothesis_result(tracking, supplied_summary)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    default_limitations = [
        "Results apply only to the selected models, questions, prompt variants, layers, and generation settings.",
        "Automatically generated paraphrases require semantic-validity checks before confirmatory analysis.",
        "A detector can correlate with errors without identifying a causal or model-internal notion of uncertainty.",
    ]
    if normalized_profile == "smoke":
        default_limitations.insert(0, "The smoke profile is intentionally small and cannot support research conclusions.")
    elif normalized_profile == "pilot":
        default_limitations.insert(0, "The pilot profile is exploratory; analysis choices should be frozen before the full run.")
    report_limitations = list(dict.fromkeys([*default_limitations, *(limitations or [])]))

    report_path = root / "report.md"
    summary_path = root / "summary.json"
    artifacts_payload = {
        "report": report_path.relative_to(root).as_posix(),
        "summary": summary_path.relative_to(root).as_posix(),
        "tables": {name: path.relative_to(root).as_posix() for name, path in tables.items()},
        "figures": {name: path.relative_to(root).as_posix() for name, path in figures.items()},
    }
    summary_payload = {
        **supplied_summary,
        "schema_version": 1,
        "profile": normalized_profile,
        "generated_at_utc": generated_at,
        "sample_sizes": sizes,
        "parse_failures": {"count": parse_count, "rate": parse_rate},
        "hypothesis": hypothesis,
        "metadata": run_metadata,
        "limitations": report_limitations,
        "artifacts": artifacts_payload,
    }
    summary_path.write_text(
        json.dumps(_json_safe(summary_payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report_text = _build_report(
        profile=normalized_profile,
        generated_at=generated_at,
        sample_sizes=sizes,
        parse_failures=parse_count,
        parse_failure_rate=parse_rate,
        hypothesis=hypothesis,
        summary=supplied_summary,
        metadata=run_metadata,
        limitations=report_limitations,
        table_paths=tables,
        figure_paths=figures,
        run_dir=root,
    )
    report_path.write_text(report_text, encoding="utf-8")

    return ReportArtifacts(
        report=report_path,
        summary=summary_path,
        tables=tables,
        figures=figures,
    )

