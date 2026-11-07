# Research Plan

## Research question

Can prompt-only hidden states from a decoder-only transformer predict whether the model's subsequently generated answer to a factual question will be correct?

## Hypotheses

- **H1:** A linear classifier trained on the last prompt token's hidden state can predict answer correctness above a majority-class baseline on held-out questions.
- **H2:** Predictive performance differs by layer, with middle or later layers outperforming the embedding and earliest layers.
- **H3:** Hidden-state probes add signal beyond simple output-confidence baselines such as first-token entropy or answer log probability.

H2 is exploratory unless the layer range is committed before the final evaluation.

## Phase 1: freeze the pilot

### Actions

1. Select one open-weights decoder-only instruction model that fits the available hardware.
2. Pin its model and tokenizer revision.
3. Select 5 PopQA facts stratified by subject popularity and relation type, then create 4 controlled wordings per fact (20 prompts total).
4. Freeze one prompt template, for example: `Answer the question briefly. If you do not know, say "I don't know."\nQuestion: {question}\nAnswer:`
5. Use greedy decoding and a small maximum number of new tokens.

### Completion criteria

- The same command produces the same 5 fact IDs and 20 wording variants twice.
- Two researchers approve that every four-prompt group preserves the same subject, relation, and accepted answers.
- Both correct and incorrect outputs appear. Abstentions remain separately labeled.
- Each row has a stable dataset ID and all accepted answer aliases.

## Phase 2: learn and implement hidden-state extraction

For a Hugging Face causal language model, call the model on the tokenized **prompt only** with `output_hidden_states=True`, `return_dict=True`, and inference mode enabled. The returned `hidden_states` is normally a tuple containing the embedding output followed by each transformer's layer output. Each tensor has shape:

`[batch_size, sequence_length, hidden_size]`

For a single prompt, extract `hidden_states[layer][0, final_prompt_token, :]` for every layer and stack the vectors into:

`[number_of_layers_plus_embedding, hidden_size]`

Then generate the answer. Do not append the gold answer or generated answer when computing the prompt-only representation.

Reference: [Hugging Face model outputs](https://huggingface.co/docs/transformers/main_classes/output).

### Completion criteria

- The saved state contains one vector for every expected layer.
- The selected token is the final non-padding prompt token.
- Tensors are detached, moved to CPU, and saved with documented dtype and shape.
- Generation uses the same prompt token IDs as extraction.

## Phase 3: verify correctness of the extraction

Use several independent checks:

1. **Shape check:** number of returned states equals the configured number of layers plus the embedding state.
2. **Token-position check:** decoding the selected token ID shows it is the last token in the rendered prompt, accounting for chat-template control tokens.
3. **First-token check:** with greedy decoding, the argmax of the prompt forward pass's final logits should equal the first generated token when no additional logits processors change it.
4. **Repeatability check:** in evaluation mode, repeated prompt-only forward passes produce matching hidden states within numerical tolerance.
5. **Leakage check:** changing the reference answer without changing the prompt cannot change the extracted state, because the reference answer is never an input.
6. **Round-trip check:** load a saved tensor and confirm its shape, dtype, question ID, layer indexing, and prompt hash.

## Phase 4: label answers

Use a two-stage labeler:

1. normalized exact match or alias containment against PopQA's accepted answers;
2. manual review for all pilot examples, then for all uncertain cases and a random audit sample at scale.

Store the automatic and reviewed labels separately. Create a short written rubric before full annotation. If reviewers disagree, keep the disagreement and adjudication fields rather than silently overwriting them.

## Phase 5: scale data collection

The initial 20 prompts only test the pipeline. Scale by adding complete four-wording fact groups toward at least hundreds of examples in each outcome class. Sample facts before generation, preserve popularity and relation metadata, and never rebalance the test set based on probe performance.

Split by underlying `fact_id` before training probes. Keep the original wording, all paraphrases, answers, and hidden-state artifacts from one fact in the same split.

## Phase 6: establish baselines

At minimum, compare each hidden-state probe with:

- majority-class prediction;
- random prediction with the observed class frequency;
- subject/entity popularity alone;
- first-token entropy or maximum probability;
- generated-answer mean log probability, noting that this last baseline is post-generation and therefore answers a different practical question.

Use a regularized logistic regression as the first probe. Fit one probe per layer rather than concatenating every layer at first; this makes the result interpretable and greatly reduces dimensionality.

## Phase 7: evaluate

Report:

- class counts and abstention rate;
- AUROC and average precision for binary correct/incorrect evaluation;
- balanced accuracy and F1 at a threshold selected only on validation data;
- confidence intervals from bootstrap resampling by question;
- performance by layer, popularity band, and relation type;
- comparison with all baselines.

If abstentions are frequent, report a separate three-class analysis or exclude them from the binary analysis with the exclusion count stated explicitly.

## Phase 8: interpret carefully

A successful probe shows that correctness information is statistically recoverable from a representation. It does **not** by itself show that a specific neuron causes hallucination, that the model consciously "knows" it will be wrong, or that the signal generalizes to other models and datasets.

The final report should cover dataset age, benchmark contamination, scoring errors, prompt sensitivity, model specificity, class imbalance, multiple layer comparisons, and the distinction between factual error and hallucination.

## Suggested team checkpoints

1. **Dataset review:** approve PopQA, label definitions, exclusions, and the 5-fact/20-prompt paired sampling rule.
2. **Pilot review:** inspect every prompt, answer, label, and tensor shape.
3. **Protocol freeze:** pin code, model revision, prompt, decoding settings, and evaluation metrics before scaling.
4. **Midpoint audit:** manually inspect a random sample of saved records and labels.
5. **Final analysis:** run the predeclared evaluation once on the untouched test split.
