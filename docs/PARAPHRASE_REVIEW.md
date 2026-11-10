# Paraphrase Review Checklist

Review candidates before looking at model answers or detector scores. For each row in `data/paraphrase_candidates_full.csv`:

1. Read the original and rewritten question side by side.
2. Confirm that every fact, number, unit, date, name, and technical term is preserved.
3. Confirm that negation and qualifiers such as *except*, *least*, *most*, *false*, and *incorrect* are preserved.
4. Confirm that the requested task and scope are unchanged.
5. Check the four fixed answer choices and verify that the original correct choice remains correct.
6. Rewrite the candidate manually if it is unclear, too similar, or grammatically broken.
7. Record useful concerns or edits in `reviewer_notes`.
8. Set `semantic_status` to `validated` only after every check passes; otherwise use `rejected` or leave it `unreviewed`.

The full pipeline requires exactly one validated row for every configured question/variant pair. It will reject duplicate, missing, or unreviewed rows.

For stronger review, have both researchers independently review a subset and report agreement. Resolve disagreements without viewing experimental outcomes.

