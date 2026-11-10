"""Typed, validated experiment configuration loaded from TOML files."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
import tomllib
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def _section_values(
    cls: type[Any], raw: Mapping[str, Any] | None, section: str
) -> dict[str, Any]:
    values = dict(raw or {})
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ConfigError(f"Unknown key(s) in [{section}]: {', '.join(unknown)}")
    return values


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{name} must be a string or list of strings")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise ConfigError(f"{name} cannot contain empty values")
    return result


@dataclass(frozen=True, slots=True)
class RunConfig:
    name: str = "experiment"
    profile: str = "default"
    seed: int = 42
    output_root: Path = Path("results")
    resume: bool = True

    def __post_init__(self) -> None:
        _nonempty(self.name, "run.name")
        _nonempty(self.profile, "run.profile")
        if self.seed < 0:
            raise ConfigError("run.seed must be non-negative")
        object.__setattr__(self, "output_root", Path(self.output_root))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> RunConfig:
        return cls(**_section_values(cls, raw, "run"))


@dataclass(frozen=True, slots=True)
class DataConfig:
    backend: str = "huggingface"
    dataset_id: str = "cais/mmlu"
    dataset_config: str = "all"
    split: str = "test"
    revision: str | None = None
    subjects: tuple[str, ...] = ()
    max_questions: int | None = None
    cache_dir: Path | None = None
    trust_remote_code: bool = False
    synthetic_size: int = 48

    def __post_init__(self) -> None:
        _nonempty(self.dataset_id, "data.dataset_id")
        _nonempty(self.dataset_config, "data.dataset_config")
        _nonempty(self.split, "data.split")
        if self.backend not in {"huggingface", "synthetic"}:
            raise ConfigError("data.backend must be 'huggingface' or 'synthetic'")
        if self.max_questions is not None and self.max_questions <= 0:
            raise ConfigError("data.max_questions must be positive or omitted")
        if self.synthetic_size <= 0:
            raise ConfigError("data.synthetic_size must be positive")
        object.__setattr__(self, "subjects", _string_tuple(self.subjects, "data.subjects"))
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> DataConfig:
        values = _section_values(cls, raw, "data")
        if "subjects" in values:
            values["subjects"] = _string_tuple(values["subjects"], "data.subjects")
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ParaphraseConfig:
    mode: str = "templates"
    variants: tuple[str, ...] = ("canonical", "direct", "exam")
    curated_path: Path | None = None
    require_validated: bool = False

    def __post_init__(self) -> None:
        allowed = {"none", "templates", "curated", "templates+curated"}
        if self.mode not in allowed:
            raise ConfigError(f"paraphrases.mode must be one of {sorted(allowed)}")
        object.__setattr__(
            self,
            "variants",
            _string_tuple(self.variants, "paraphrases.variants"),
        )
        if self.curated_path is not None:
            object.__setattr__(self, "curated_path", Path(self.curated_path))
        if "curated" in self.mode and self.curated_path is None:
            raise ConfigError(
                "paraphrases.curated_path is required when mode includes curated data"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> ParaphraseConfig:
        values = _section_values(cls, raw, "paraphrases")
        if "variants" in values:
            values["variants"] = _string_tuple(values["variants"], "paraphrases.variants")
        return cls(**values)

    @property
    def template_names(self) -> tuple[str, ...]:
        """Compatibility name used by the variant builder."""

        return self.variants

    @property
    def include_canonical(self) -> bool:
        return "canonical" in self.variants


@dataclass(frozen=True, slots=True)
class ModelConfig:
    backend: str = "huggingface"
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    batch_size: int = 1
    max_input_tokens: int = 1_024
    max_new_tokens: int = 8
    do_sample: bool = False
    temperature: float = 0.0
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        _nonempty(self.model_id, "model.model_id")
        if self.backend not in {"huggingface", "synthetic"}:
            raise ConfigError("model.backend must be 'huggingface' or 'synthetic'")
        if self.batch_size <= 0:
            raise ConfigError("model.batch_size must be positive")
        if self.max_new_tokens <= 0:
            raise ConfigError("model.max_new_tokens must be positive")
        if self.max_input_tokens <= 0:
            raise ConfigError("model.max_input_tokens must be positive")
        if self.temperature < 0:
            raise ConfigError("model.temperature must be non-negative")
        if self.do_sample and self.temperature <= 0:
            raise ConfigError("model.temperature must be positive when do_sample=true")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> ModelConfig:
        return cls(**_section_values(cls, raw, "model"))


@dataclass(frozen=True, slots=True)
class FeaturesConfig:
    layers: tuple[int | float, ...] = (0.25, 0.5, 0.75, 1.0)
    pooling: str = "last_prompt_token"
    cache: bool = True

    def __post_init__(self) -> None:
        layers = self.layers
        if isinstance(layers, list):
            layers = tuple(layers)
            object.__setattr__(self, "layers", layers)
        if not isinstance(layers, tuple) or not layers:
            raise ConfigError("features.layers must be a non-empty list")
        for layer in layers:
            if isinstance(layer, bool) or not isinstance(layer, (int, float)):
                raise ConfigError("features.layers entries must be integers or fractions")
            if isinstance(layer, float) and not 0 < layer <= 1:
                raise ConfigError("fractional features.layers entries must be in (0, 1]")
        if len(set(layers)) != len(layers):
            raise ConfigError("features.layers cannot contain duplicates")
        if self.pooling not in {"last_prompt_token", "mean_prompt"}:
            raise ConfigError(
                "features.pooling must be 'last_prompt_token' or 'mean_prompt'"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> FeaturesConfig:
        values = _section_values(cls, raw, "features")
        if isinstance(values.get("layers"), list):
            values["layers"] = tuple(values["layers"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class SplitConfig:
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    group_by: str = "question_id"
    stratify_by: str = "subject"
    heldout_subjects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        fractions = (
            self.train_fraction,
            self.validation_fraction,
            self.test_fraction,
        )
        if any(value <= 0 or value >= 1 for value in fractions):
            raise ConfigError("split fractions must each be between 0 and 1")
        if abs(sum(fractions) - 1.0) > 1e-9:
            raise ConfigError("split fractions must sum to 1.0")
        _nonempty(self.group_by, "split.group_by")
        _nonempty(self.stratify_by, "split.stratify_by")
        object.__setattr__(
            self,
            "heldout_subjects",
            _string_tuple(self.heldout_subjects, "split.heldout_subjects"),
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> SplitConfig:
        values = _section_values(cls, raw, "split")
        if "heldout_subjects" in values:
            values["heldout_subjects"] = _string_tuple(
                values["heldout_subjects"], "split.heldout_subjects"
            )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    classifier: str = "logistic_regression"
    c: float = 1.0
    max_iter: int = 2_000
    class_weight: str | None = "balanced"
    invariant_variance: float = 0.8
    max_invariant_components: int = 16

    def __post_init__(self) -> None:
        if self.classifier != "logistic_regression":
            raise ConfigError("probe.classifier currently supports logistic_regression only")
        if self.c <= 0:
            raise ConfigError("probe.c must be positive")
        if self.max_iter <= 0:
            raise ConfigError("probe.max_iter must be positive")
        if self.class_weight not in {None, "balanced"}:
            raise ConfigError("probe.class_weight must be 'balanced' or omitted")
        if not 0 < self.invariant_variance <= 1:
            raise ConfigError("probe.invariant_variance must be in (0, 1]")
        if self.max_invariant_components <= 0:
            raise ConfigError("probe.max_invariant_components must be positive")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> ProbeConfig:
        return cls(**_section_values(cls, raw, "probe"))


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    bootstrap_replicates: int = 1_000
    confidence_level: float = 0.95
    stable_error_margin: float = 0.05
    min_group_size: int = 2
    calibration_bins: int = 10
    primary_layer: int | float = 1.0

    def __post_init__(self) -> None:
        if self.bootstrap_replicates <= 0:
            raise ConfigError("evaluation.bootstrap_replicates must be positive")
        if not 0 < self.confidence_level < 1:
            raise ConfigError("evaluation.confidence_level must be between 0 and 1")
        if not 0 <= self.stable_error_margin <= 1:
            raise ConfigError(
                "evaluation.stable_error_margin must be between 0 and 1"
            )
        if self.min_group_size < 2:
            raise ConfigError("evaluation.min_group_size must be at least 2")
        if self.calibration_bins < 2:
            raise ConfigError("evaluation.calibration_bins must be at least 2")
        if isinstance(self.primary_layer, float) and not 0 < self.primary_layer <= 1:
            raise ConfigError("fractional evaluation.primary_layer must be in (0, 1]")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> EvaluationConfig:
        return cls(**_section_values(cls, raw, "evaluation"))


@dataclass(frozen=True, slots=True)
class OutputConfig:
    save_features: bool = True
    save_models: bool = True
    save_generations: bool = True
    save_bootstrap_samples: bool = False
    table_format: str = "csv"

    def __post_init__(self) -> None:
        if self.table_format not in {"csv", "parquet"}:
            raise ConfigError("output.table_format must be 'csv' or 'parquet'")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> OutputConfig:
        return cls(**_section_values(cls, raw, "output"))


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    run: RunConfig
    data: DataConfig
    paraphrases: ParaphraseConfig
    model: ModelConfig
    features: FeaturesConfig
    split: SplitConfig
    probe: ProbeConfig
    evaluation: EvaluationConfig
    output: OutputConfig

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ExperimentConfig:
        allowed = {
            "run",
            "data",
            "paraphrases",
            "model",
            "features",
            "split",
            "probe",
            "evaluation",
            "output",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ConfigError(f"Unknown top-level TOML section(s): {', '.join(unknown)}")
        for key, value in raw.items():
            if not isinstance(value, Mapping):
                raise ConfigError(f"[{key}] must be a TOML table")
        return cls(
            run=RunConfig.from_mapping(raw.get("run")),
            data=DataConfig.from_mapping(raw.get("data")),
            paraphrases=ParaphraseConfig.from_mapping(raw.get("paraphrases")),
            model=ModelConfig.from_mapping(raw.get("model")),
            features=FeaturesConfig.from_mapping(raw.get("features")),
            split=SplitConfig.from_mapping(raw.get("split")),
            probe=ProbeConfig.from_mapping(raw.get("probe")),
            evaluation=EvaluationConfig.from_mapping(raw.get("evaluation")),
            output=OutputConfig.from_mapping(raw.get("output")),
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> ExperimentConfig:
        config_path = Path(path)
        try:
            with config_path.open("rb") as handle:
                raw = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"Could not load configuration {config_path}: {exc}") from exc
        return cls.from_mapping(raw)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment TOML file."""

    return ExperimentConfig.from_toml(path)
