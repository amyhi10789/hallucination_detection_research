# Results Guide

Open `results/LATEST.txt`, then begin with the referenced run's `report.md`.

## Headline Files

- `report.md`: readable summary, figures, hypothesis decision, limitations, and links.
- `summary.json`: the same headline information for scripts or later aggregation.
- `tables/hypothesis_test.json`: primary effect size and cluster-bootstrap result.
- `tables/change_intervals.csv`: actual and predicted changes with confidence intervals.
- `tables/stable_error_analysis.csv`: contrasts meeting the predefined stable-error rule.
- `tables/all_metrics.csv`: discrimination and calibration across methods, layers, splits, variants, and subjects.

## Diagnostic Files

- `inference/responses.csv`: generated answers, parser status, and correctness.
- `predictions/detector_predictions.csv`: detector probabilities for every layer and method.
- `tables/question_level_changes.csv`: paired canonical-to-paraphrase outcomes.
- `data/split_assignments.csv`: evidence that question groups did not cross splits.
- `run_metadata.json`: model, package, hardware, seed, and Git information.
- `manifest.json`: completion state, artifact sizes, and checksums.

## Reading the Primary Result

A positive mean excess volatility means detector scores moved more than observed LLM error rates. The stated hypothesis is supported only if the one-sided lower confidence bound is above zero. A positive point estimate with an interval crossing zero is inconclusive, not confirmation.

Compare the invariant method with both logistic baselines. Lower drift is useful only if held-out AUROC, AUPRC, and calibration remain acceptable.

Smoke results verify code. Pilot results refine the protocol. Only a completed full run with validated paraphrases should be used for confirmatory claims.

