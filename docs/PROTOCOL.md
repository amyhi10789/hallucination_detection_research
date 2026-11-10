# Experimental Protocol

This document records the intended confirmatory analysis before the full run. Any change made after inspecting pilot or full detector outcomes should be dated and described rather than silently replacing this protocol.

## Independent Unit

The original MMLU question is the independent sampling unit. Its canonical prompt and all paraphrases remain in the same split and in the same bootstrap cluster.

For question `i` and wording `v`:

- `y_iv = 1` when the generated answer is incorrect or unparseable, otherwise `0`;
- `p_iv = P(error)` from a detector using only a pre-generation hidden state.

Error rates are averages across questions. A single deterministic question response is a binary outcome, not an error rate.

## Frozen Defaults

- Dataset: `cais/mmlu`, configuration `all`, test split, pinned in `configs/full.toml`.
- Model: `Qwen/Qwen2.5-1.5B-Instruct`, pinned in `configs/full.toml`.
- Generation: greedy decoding, at most eight new tokens, answer restricted by instruction to A–D.
- Representation: final real prompt token at the assistant-answer boundary from four predeclared layer fractions.
- Split: 60% train, 20% validation, 20% test within non-held-out subjects; named subjects are held out entirely.
- Paraphrases: three human-validated stem rewrites per question, with choices and choice order fixed.
- Primary detector/layer: canonical logistic probe at the layer identified by `evaluation.primary_layer`.
- Stable-error equivalence margin: five percentage points.
- Full-run uncertainty: 2,000 question-cluster bootstrap resamples.

## Detectors

1. `canonical_logistic`: standardized logistic regression trained only on canonical prompts.
2. `augmented_logistic`: standardized logistic regression trained on canonical and paraphrased prompts.
3. `invariant_projection`: estimates dominant within-question wording directions on training data, removes those directions, and fits logistic regression to the projected representations.

The improved method is successful only if it reduces tracking mismatch or detector drift without materially degrading held-out error detection and calibration.

## Primary Estimands

For each canonical-to-paraphrase contrast:

```text
delta_error = mean(y_paraphrase - y_canonical)
delta_prediction = mean(p_paraphrase - p_canonical)
tracking_mismatch = |delta_prediction - delta_error|
excess_volatility = |delta_prediction| - |delta_error|
```

The directional hypothesis is supported only when the one-sided cluster-bootstrap lower confidence bound for mean excess volatility is above zero. Report the effect size and interval even when the result does not support the hypothesis.

## Secondary Outcomes

- AUROC and AUPRC;
- Brier score, log loss, and expected calibration error;
- mean absolute within-question detector-score drift;
- correctness and detector-decision flip rates;
- stable-error contrasts whose 90% bootstrap interval lies inside the equivalence margin;
- performance by subject, wording, split, method, and layer;
- held-out-domain performance.

Layer-by-layer and subject-by-subject findings are exploratory unless separately corrected for multiple comparisons.

## Data-Quality Rules

- Preserve parser failures and report them separately; do not drop them.
- Validate paraphrases before examining detector results.
- Reject rewrites that change a fact, number, entity, negation, scope, answer choice, or correct answer.
- Do not reorder answer choices in the primary study.
- Fit scaling, wording projections, classifiers, and any threshold using training or validation data only.
- Never select a layer or method using test-set results and then describe that selection as prespecified.

## Interpretation

Discrimination does not establish calibration, robustness, causality, or a mechanistic uncertainty signal. Conclusions apply only to the tested models, benchmark, prompt construction, generation settings, and operational definition of hallucination.

