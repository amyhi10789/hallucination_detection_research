# Robust Hallucination Detection from Pre-Generation Hidden States

> **Project status:** The experimental pipeline is implemented and tested. No real-model results are claimed until the smoke, pilot, and full profiles have been run and the full study's paraphrases have been human-validated.

## Overview

Large language models can produce incorrect answers with high confidence. A hallucination detector should warn us when an answer is likely to be wrong, but a detector may instead learn shortcuts tied to a prompt's wording or subject area.

This project investigates whether classifiers trained on an LLM's internal neural representations can predict incorrect responses **before the response is generated**. We will test whether those predictions remain reliable across meaning-preserving paraphrases, subject domains, and transformer layers. We then plan to develop and evaluate a detection method designed to be more consistent across reworded questions and unfamiliar domains.

For this study, *hallucination* has a narrow, measurable meaning: an incorrect response to a controlled question-answering task. This does not cover every type of LLM hallucination.

## Research Question

> How accurately do pre-generation hidden-state hallucination detectors track changes in LLM error rates caused by meaning-preserving question paraphrases, compared with their prediction changes when error rates remain stable?

**In simpler terms:** When the same question is worded differently, how well do AI hallucination detectors predict changes in how often the AI gives wrong answers?

## Hypothesis

> Hidden-state hallucination detectors will exhibit larger prediction changes than actual error-rate changes across equivalent questions, indicating sensitivity to question wording beyond changes in the LLM's performance.

In other words, we predict that detector scores will shift across paraphrases even when the LLM's actual performance stays similar.

## Why This Matters

A detector is useful only if its confidence reflects the model's real likelihood of making an error. If harmless wording changes cause large detector shifts without corresponding changes in model accuracy, then the detector may be measuring surface-level prompt features rather than genuine uncertainty. Such a detector could appear accurate in a familiar benchmark while failing on real-world prompts or new subject areas.

## Experimental Pipeline

1. **Build question groups.** Select questions from multiple subject areas and create meaning-preserving paraphrases for each one.
2. **Measure model performance.** Run open-weight LLMs under controlled generation settings and label each response as correct or incorrect.
3. **Extract internal representations.** Record a defined hidden-state representation before answer generation at multiple transformer layers.
4. **Train detectors.** Fit classifiers that use the frozen hidden states to estimate the probability of an incorrect response.
5. **Test robustness.** Evaluate detector behavior on held-out paraphrases, subject domains, and model layers.
6. **Improve the method.** Compare the canonical and paraphrase-augmented probes with an invariant-projection probe that learns and removes dominant within-question wording directions before classification.

The central comparison is:

```text
actual change    = |error rate for wording A - error rate for wording B|
detector change  = |mean detector score A  - mean detector score B|
```

The hypothesis is supported if detector changes remain large for equivalent question pairs whose actual error-rate changes are small.

## Evaluation Principles

The implemented protocol follows these principles:

- Keep every original question and all of its paraphrases in the same data split to prevent semantic leakage.
- Compare in-domain performance with held-out-paraphrase and held-out-domain performance.
- Report both error prediction quality and probability calibration, not accuracy alone.
- Analyze results by subject, paraphrase type, model, and transformer layer.
- Compare detector-score changes directly with observed error-rate changes.
- Use fixed configurations, documented random seeds, and uncertainty estimates for reported metrics.
- Audit ambiguous correctness labels rather than silently treating them as clear-cut examples.

Reported metrics include AUROC and AUPRC for error detection; Brier score, log loss, and expected calibration error for probability quality; and paired change, score drift, detector flips, tracking mismatch, and excess volatility for paraphrase robustness. The TOML profiles freeze the dataset, model revisions, sample sizes, stable-error margin, layers, and bootstrap counts before a run begins.

## Roadmap

- [x] Define the research question and hypothesis
- [x] Select pinned MMLU and Qwen2.5 defaults for smoke, pilot, and full profiles
- [x] Implement response generation, correctness labeling, and hidden-state extraction
- [x] Implement canonical and paraphrase-augmented logistic-probe baselines
- [x] Implement the invariant-projection detector
- [x] Implement paraphrase, held-out-domain, and layer evaluations
- [x] Generate organized tables, figures, metadata, and a Markdown report automatically
- [ ] Human-review and approve the full study's question paraphrases
- [ ] Run the full profile and interpret the resulting evidence

## Getting Started

Python 3.11 or 3.12 is recommended. On Windows PowerShell, run:

```powershell
.\scripts\setup.ps1
.\scripts\run_tests.ps1
.\scripts\run_synthetic.ps1
```

The synthetic profile requires no internet, model download, or GPU. It verifies the complete workflow and creates a sample report under `results/` without producing scientific evidence.

Next, run the real-model smoke profile:

```powershell
.\scripts\run_smoke.ps1
```

The first real run downloads the public model and dataset from Hugging Face. The smoke profile uses `Qwen/Qwen2.5-0.5B-Instruct` and 24 MMLU questions, so it can run on CPU, although a GPU is faster. Check the machine and configuration at any time with:

```powershell
.\scripts\doctor.ps1 -Profile smoke
```

### Run Profiles

| Profile | Purpose | Model | Questions | Paraphrases | Bootstrap draws |
|---|---|---|---:|---|---:|
| `synthetic` | Offline pipeline verification only | Synthetic | 48 | Wrapper controls | 50 |
| `smoke` | Small real-model functional check | Qwen2.5-0.5B-Instruct | 24 | Wrapper controls | 200 |
| `pilot` | Preliminary protocol and variability check | Qwen2.5-1.5B-Instruct | 400 | Wrapper controls | 1,000 |
| `full` | Confirmatory study | Qwen2.5-1.5B-Instruct | 3,000 | Human-validated stem paraphrases | 2,000 |

Run the larger automatic profile with:

```powershell
.\scripts\run_pilot.ps1
```

Wrapper controls change the surrounding instruction while leaving the question stem and choices untouched. They are guaranteed meaning-preserving but are not a substitute for genuine stem paraphrases.

### Prepare the Full Study's Paraphrases

Draft candidates with the configured local model:

```powershell
.\scripts\draft_paraphrases.ps1 -Profile full
```

This creates `data/paraphrase_candidates_full.csv`. Review every row without looking at detector results. Confirm that facts, numbers, entities, negation, scope, choices, and the correct answer are unchanged. Change `semantic_status` to `validated` only after review, then save the approved rows as `data/curated_paraphrases.csv`.

The full profile intentionally refuses to run with missing or unreviewed paraphrases:

```powershell
.\scripts\run_full.ps1
```

This human review is part of the research protocol, not a software limitation: semantic similarity alone cannot establish that a rewrite preserves the correct answer.

### Direct Python Usage

The PowerShell scripts call the same CLI available on every platform:

```text
python -m hallucination_detection validate-config --config configs/smoke.toml
python -m hallucination_detection doctor --config configs/smoke.toml
python -m hallucination_detection run --config configs/smoke.toml
```

## What Each Run Produces

Every run receives an immutable timestamped directory and `results/LATEST.txt` points to the newest completed run:

```text
results/<timestamp>_<profile>/
├── report.md                         # Start here
├── summary.json                      # Machine-readable headline results
├── manifest.json                     # Status, artifact sizes, and checksums
├── config.source.toml
├── config.resolved.json
├── run_metadata.json                 # Versions, hardware, Git state, model revision
├── data/                              # Questions, prompts, and leakage-safe splits
├── inference/                         # Raw generations and hidden-state arrays
├── models/                            # Saved probes for every selected layer
├── predictions/                       # Row-level detector probabilities
├── tables/                            # Metrics, paired changes, CIs, hypothesis test
├── figures/                           # Tracking, layer, drift, and calibration plots
└── logs/run.log
```

The primary hypothesis test uses a question-cluster bootstrap. It reports the excess detector volatility effect size, a one-sided lower confidence bound, and whether that bound is above zero. Smoke and pilot results are labeled non-confirmatory in their reports.

Before the confirmatory run, read the [experimental protocol](docs/PROTOCOL.md), use the [paraphrase review checklist](docs/PARAPHRASE_REVIEW.md), and interpret the artifacts with the [results guide](docs/RESULTS_GUIDE.md).

## Implementation Notes

- Hidden states come from a prompt-only forward pass at the final prompt token, before any answer token is generated.
- All variants of an original question remain in the same train, validation, test, or held-out-domain split.
- Invalid generations remain visible as parse failures and errors rather than being silently dropped.
- The canonical probe trains only on canonical prompts; the augmented and invariant probes train on every training variant.
- The invariant probe estimates within-question wording directions with SVD, projects them out, and then fits a logistic classifier.
- Model weights, Hugging Face caches, large feature arrays, and generated run folders are not committed.

The default public resources are [CAIS MMLU](https://huggingface.co/datasets/cais/mmlu) and [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) / [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct). Exact revisions are pinned in each TOML profile and recorded again in every run.

## Team

- Amy Wang
- Alyssa Pan

Student researchers at Eastside Preparatory School.

## License

A license has not yet been selected. Until one is added, no permission is granted to copy, modify, or redistribute the project materials.
