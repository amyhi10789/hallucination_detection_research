import json

from hallucination_detection.pipeline import run_experiment


def test_synthetic_pipeline_writes_complete_result_bundle(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = config_dir / "tiny.toml"
    config.write_text(
        """
[run]
name = "tiny"
profile = "smoke"
seed = 9
output_root = "results"

[data]
backend = "synthetic"
dataset_id = "synthetic/mmlu"
dataset_config = "all"
revision = "local"
split = "test"
max_questions = 24
subjects = ["science", "history", "ethics"]

[paraphrases]
mode = "templates"
variants = ["canonical", "direct", "exam"]
require_validated = false

[model]
backend = "synthetic"
model_id = "synthetic"
revision = "local"
device = "cpu"
dtype = "float32"
batch_size = 8
max_input_tokens = 256
max_new_tokens = 4

[features]
layers = [0.5, 1.0]
pooling = "last_prompt_token"

[split]
train_fraction = 0.60
validation_fraction = 0.20
test_fraction = 0.20
heldout_subjects = []

[probe]
c = 1.0
max_iter = 500
class_weight = "balanced"
invariant_variance = 0.80
max_invariant_components = 3

[evaluation]
bootstrap_replicates = 5
confidence_level = 0.95
stable_error_margin = 0.10
calibration_bins = 5
primary_layer = 1.0

[output]
save_features = true
save_models = true
save_bootstrap_samples = false
""".strip()
        + "\n",
        encoding="utf-8",
    )

    run_dir = run_experiment(config)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "complete"
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "tables" / "hypothesis_test.json").is_file()
    assert (run_dir / "figures" / "tracking_scatter.png").is_file()
    assert (run_dir / "predictions" / "detector_predictions.csv").is_file()

