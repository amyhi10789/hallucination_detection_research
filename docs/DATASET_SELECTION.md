# Dataset Selection

## Decision

Use **PopQA as the primary dataset** for the first experiment. Use SQuAD 2.0 later as a separate context-grounded abstention experiment, and TruthfulQA as an adversarial robustness check. HaluEval is useful as a downstream comparison dataset, but it is not the cleanest source for collecting this project's prompt-only hidden states.

## Selection criteria

The primary dataset should have:

1. factual questions that can be sent directly to a decoder-only language model;
2. short, objectively scorable answers;
3. answer aliases or another reliable grading mechanism;
4. enough variation to obtain both correct and incorrect model answers;
5. enough rows for train/validation/test splits; and
6. an accessible official source and research-friendly license.

## Shortlist

| Dataset | What it contains | Strength for this project | Main drawback | Recommended role |
|---|---|---|---|---|
| [PopQA](https://github.com/AlexTMallen/adaptive-retrieval) | About 14K entity-centric, open-domain questions derived from Wikidata; answer aliases, relation types, and popularity fields | Short answers, easy sampling, automatic scoring, and natural difficulty variation | Templated wording; some facts may have changed since construction | **Primary dataset** |
| [TriviaQA](https://nlp.cs.washington.edu/triviaqa/) | 110K open-domain question-answer pairs, with a much larger reading-comprehension release | Diverse natural trivia questions and answer aliases | Larger download and noisier questions; evidence and grading pipeline are more involved | External validation after PopQA |
| [TruthfulQA](https://github.com/sylinrl/TruthfulQA) | 817 adversarial questions built around common misconceptions | Directly targets truthful versus imitative false answers | Small, only one validation-style set, and open-ended scoring is difficult | Adversarial robustness check |
| [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) | Wikipedia passages with answerable questions and over 50K adversarially unanswerable questions | Excellent for testing whether a model abstains when context does not support an answer | Measures context faithfulness, not closed-book factual recall | Separate grounded-hallucination experiment |
| [HaluEval](https://github.com/RUCAIBox/HaluEval) | 35K generated and human-annotated examples; QA subset pairs knowledge and questions with right and hallucinated answers | Explicit positive/negative hallucination examples | The candidate answers were created separately; they do not directly label what our chosen model will generate from its pre-answer state | Detector comparison or later extension |

## Why PopQA wins for the first pass

The official PopQA repository describes 14K question-answer pairs and exposes fields including `question`, `possible_answers`, relation type, Wikidata identifiers, and entity popularity. That lets us sample across popularity rather than taking the first 20 rows, which could accidentally contain only easy or only hard questions.

The pilot uses 5 fixed underlying facts, each expanded into 4 meaning-preserving wordings, for 20 total prompts:

- 2 facts from the lower third of subject popularity;
- 1 fact from the middle third; and
- 2 facts from the upper third.

Within each band, spread facts across relation types. Each fact receives the original, lexical, syntactic, and concise wording variants. Use PopQA's disambiguated Wikipedia title when it is available, exclude ambiguous single-word non-geographic subjects before sampling, and have both researchers approve meaning equivalence. Save the selected fact IDs and seed in a manifest so every team member runs exactly the same prompts.

The first 20 prompts test the pipeline and paired design, not statistical performance. If the later collection needs more examples in an outcome class, draw additional complete four-wording fact groups using a recorded rule; do not hand-pick prompts after seeing individual outputs.

## Important caveats

### A wrong answer is not automatically a hallucination

String mismatch can occur because of punctuation, capitalization, alternate names, or an incomplete alias list. Normalize answers for an initial exact-match score, then manually review mismatches. Preserve `abstain` separately from `incorrect`.

### Facts can drift

PopQA is derived from a snapshot of Wikidata. Relations such as occupation, employer, political office, or team membership can change. Flag time-sensitive relations and either exclude them or score them against the dataset's historical snapshot while clearly stating that choice.

### Public data may be in a model's training corpus

The goal is to predict whether this model will answer correctly, not to claim unseen-knowledge generalization. Still, public benchmark contamination must be reported as a limitation. A later extension can evaluate on newly constructed or post-training facts.

### Do not mix experimental settings

Closed-book PopQA, passage-grounded SQuAD, and candidate-response HaluEval operationalize different kinds of hallucination. Results should be reported separately rather than pooled into one score.

## Sources

- Mallen et al., [When Not to Trust Language Models: Investigating Effectiveness and Limitations of Parametric and Non-Parametric Memories](https://aclanthology.org/2023.acl-long.546/)
- Joshi et al., [TriviaQA](https://nlp.cs.washington.edu/triviaqa/)
- Lin et al., [TruthfulQA](https://openai.com/index/truthfulqa/)
- Rajpurkar, Jia, and Liang, [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)
- Li et al., [HaluEval](https://aclanthology.org/2023.emnlp-main.397/)
