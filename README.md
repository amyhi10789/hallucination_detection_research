# Robust Hallucination Detection from Pre-Generation Hidden States

> **Project status:** Research design and implementation are in progress. This repository does not yet contain a runnable experimental pipeline or published results.

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

## Planned Method

1. **Build question groups.** Select questions from multiple subject areas and create meaning-preserving paraphrases for each one.
2. **Measure model performance.** Run open-weight LLMs under controlled generation settings and label each response as correct or incorrect.
3. **Extract internal representations.** Record a defined hidden-state representation before answer generation at multiple transformer layers.
4. **Train detectors.** Fit classifiers that use the frozen hidden states to estimate the probability of an incorrect response.
5. **Test robustness.** Evaluate detector behavior on held-out paraphrases, subject domains, and model layers.
6. **Improve the method.** Use the observed failure modes to design a detector or training procedure that emphasizes paraphrase consistency and domain generalization.

### Paired paraphrase design

The experimental unit is an underlying factual question, not an individual prompt string. Each selected PopQA fact is expanded into four prompts:

1. the dataset's original wording;
2. a lexical substitution;
3. a syntactic reordering; and
4. a concise request for the same information.

Every variant preserves the subject, requested relation, and accepted answers. Two team members review meaning equivalence before model collection. All variants from one fact remain in the same train/validation/test split.

Before sampling, the pilot excludes the `color` relation and ambiguous single-word subjects for non-geographic relations. This rule is applied before any model outputs are observed; it prevents questions such as “Who composed Time?” from conflating wording sensitivity with uncertainty about which work named *Time* is intended. The exclusion is recorded in the committed configuration and manifest.

When PopQA supplies a disambiguated Wikipedia title, the sampler uses it in every wording—for example, `One Day (2011 film)` instead of only `One Day`. This preserves the target fact while reducing accidental entity ambiguity across the entire paraphrase group.

This paired design supports two distinct measurements:

- **behavioral sensitivity:** whether the model's answer changes from correct to incorrect across equivalent wordings;
- **detector sensitivity:** how much the pre-generation detector score changes across those wordings, including cases where answer correctness does not change.

The pilot uses 5 base facts × 4 wordings = 20 prompts. The full study will scale the number of base facts without changing the four-variant protocol.

The central comparison is:

```text
actual change    = |error rate for wording A - error rate for wording B|
detector change  = |mean detector score A  - mean detector score B|
```

The hypothesis is supported if detector changes remain large for equivalent question pairs whose actual error-rate changes are small.

## Evaluation Principles

The final protocol will be selected before results are analyzed. It will follow these principles:

- Keep every original question and all of its paraphrases in the same data split to prevent semantic leakage.
- Compare in-domain performance with held-out-paraphrase and held-out-domain performance.
- Report both error prediction quality and probability calibration, not accuracy alone.
- Analyze results by subject, paraphrase type, model, and transformer layer.
- Compare detector-score changes directly with observed error-rate changes.
- Use fixed configurations, documented random seeds, and uncertainty estimates for reported metrics.
- Audit ambiguous correctness labels rather than silently treating them as clear-cut examples.

Candidate metrics include AUROC and AUPRC for error detection, calibration error or Brier score for probability quality, and paired change or consistency measures for paraphrase robustness. Exact metrics, thresholds, datasets, and models remain to be finalized.

## Roadmap

- [x] Define the research question and hypothesis
- [ ] Finalize datasets, base models, and paraphrase-generation protocol
- [ ] Implement response generation, correctness labeling, and hidden-state extraction
- [ ] Establish simple and hidden-state detector baselines
- [ ] Run paraphrase, domain, and layer generalization studies
- [ ] Develop and evaluate the improved detection method
- [ ] Release reproducible configurations, results, and a research report

## Getting Started

The pilot pipeline is implemented for Windows PowerShell. From the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create the deterministic PopQA pilot:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_pilot.py
```

Before running the model, both researchers should inspect `data/processed/pilot_questions.jsonl` and change each `meaning_review_status` from `pending` to `approved` only when all four wordings request the same fact. Then collect answers and prompt-only hidden states:

```powershell
.\.venv\Scripts\python.exe scripts\collect_pilot.py
.\.venv\Scripts\python.exe scripts\verify_artifacts.py
.\.venv\Scripts\python.exe scripts\export_review_sheet.py
```

For a plumbing-only run before human approval, `collect_pilot.py --allow-unreviewed` is available, but data from that run must not be used as final research data.

To test one complete four-wording group before the full pilot, use:

```powershell
.\.venv\Scripts\python.exe scripts\collect_pilot.py --fact-limit 1 --allow-unreviewed --output-dir artifacts\smoke
.\.venv\Scripts\python.exe scripts\verify_artifacts.py --responses artifacts\smoke\responses.jsonl
```

Run the lightweight unit tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The configuration is recorded in `config/pilot.json`. Generated model weights, hidden-state tensors, review exports, and caches are ignored by Git. The selected question manifest is kept under `data/processed` so the team can commit and reproduce it.

The collector explicitly overrides the model card's sampling and repetition settings (`do_sample=false`, `temperature=1`, `top_p=1`, `top_k=0`, `repetition_penalty=1`). This makes generation genuinely greedy and allows the raw final-prompt logits to be checked directly against the first generated token.

### Expected pilot artifacts

- `data/processed/pilot_questions.jsonl`: five fact groups and their 20 wordings;
- `data/processed/pilot_manifest.json`: seed, relation counts, popularity bands, and fixed fact IDs;
- `artifacts/pilot/responses.jsonl`: prompts, answers, automatic labels, model metadata, and verification fields;
- `artifacts/pilot/hidden_states/*.safetensors`: one `[layers + 1, hidden_size]` tensor per prompt;
- `outputs/pilot_review.csv`: a human-review worksheet with separate reviewer and adjudication fields.

The first model download may be several gigabytes and may take time. The recommended pilot model is `Qwen/Qwen2.5-0.5B-Instruct`; its Hugging Face commit is pinned in `config/pilot.json` and recorded with each response.

### Implementation boundaries

The current code implements dataset sampling, controlled paraphrases, answer generation, pre-generation hidden-state extraction, conservative automatic labels, review export, and artifact verification. Probe training and final statistical analysis will be implemented only after the pilot passes review, because their inputs and split protocol must first be frozen.

Large model weights, caches, and generated tensors must not be committed to the repository.

## Team

- Amy Wang
- Alyssa Pan

Student researchers at Eastside Preparatory School.

## License

A license has not yet been selected. Until one is added, no permission is granted to copy, modify, or redistribute the project materials.
