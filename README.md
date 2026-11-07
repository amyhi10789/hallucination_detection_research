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

There is currently nothing to install or run. Once the first implementation is committed, this section will document:

- supported Python and hardware requirements;
- environment setup and dependencies;
- dataset preparation;
- model access and storage requirements;
- commands for training, evaluation, and reproducing figures.

Large model weights, caches, and restricted datasets will not be committed to the repository.

## Team

- Amy Wang
- Alyssa Pan

Student researchers at Eastside Preparatory School.

## License

A license has not yet been selected. Until one is added, no permission is granted to copy, modify, or redistribute the project materials.
