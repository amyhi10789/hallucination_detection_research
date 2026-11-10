# Data

The experiment downloads public benchmark data through Hugging Face and records the exact dataset revision in each run. Raw datasets and model caches are not committed.

`curated_paraphrases.csv` is the review gate for the full experiment. Each row must contain:

- `question_id`: the stable ID produced by the preparation step;
- `variant`: a unique name such as `paraphrase_1`;
- `paraphrased_question`: the reviewed rewrite of the question stem;
- `semantic_status`: `validated` only after a human confirms that facts, numbers, negation, answer choices, and the correct answer are unchanged;
- `reviewer_notes`: optional notes about edits or uncertainty.

Do not mark model-generated paraphrases as validated without human review.

