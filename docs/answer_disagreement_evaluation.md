# Sampled answer disagreement on the POPE pilot

## Paired experiment

This evaluation reuses the exact 30 question/image pairs from the deterministic
POPE baseline, entropy evaluation, embedding-distance evaluation, and
augmentation-consistency evaluation. The script verifies the question and
image identities, category, ground-truth label, raw and normalized baseline
prediction, correctness/error label, and the saved augmentation score before
running sampled inference. It also validates the pairs against the local POPE
annotations. The sample contains 10 Random, 10 Popular, and 10 Adversarial
questions. All error metrics use the original deterministic baseline labels;
sampled responses do not change those labels.

For each of the 30 image/question pairs, SmolVLM generated five responses with
`do_sample=true`, temperature `0.7`, top-p `0.9`, one beam, and
`max_new_tokens=32`. Responses were normalized using the same yes/no parser as
the baseline. Unparseable or ambiguous answers were assigned a third category.
Rows are processed in the stable order returned by the paired join (question
ID and normalized image ID, lexicographically). The per-question base seed is
`2027 + sample_index * 5`; each of that question's five generations uses its
base seed plus the zero-based response index. Thus all 150 generations have
recorded, nonoverlapping seeds.

The frozen model was `HuggingFaceTB/SmolVLM-500M-Instruct`, revision
`a7da5b986cb59b408707209984f360a5f4ad7e47`, using Transformers `5.22.1`,
PyTorch `2.15.0`, MPS, and float16. No inference was repeated on original
baseline inputs, and the saved baseline predictions were not modified.

## Score and metrics

For the five sampled answers, let `p_yes`, `p_no`, and `p_ambiguous` be their
empirical proportions. The score is normalized categorical entropy:

\[
D = -\frac{1}{\log 3}\sum_{c \in \{yes,no,ambiguous\}} p_c\log p_c,
\]

with zero-proportion terms omitted. It ranges from 0 for unanimous responses to
1 for an equal split among all three categories. Ambiguous answers count as
their own response class rather than being discarded or treated as yes/no.
This measures response variation under the chosen sampling settings. It is
neither model-token entropy nor a probability of error or hallucination.

The positive class for evaluation is an incorrect or ambiguous **original
baseline answer**. AUROC measures ranking of those baseline errors; AUPRC is
non-interpolated average precision. Risk–coverage curves remove the highest
score groups first and report the error rate among retained questions; ties
are excluded as a complete group. All five scores below use the same 30 labels.

## Results

The fixed baseline made 24 correct predictions and six incorrect or ambiguous
predictions (20% error prevalence, including one ambiguous baseline answer).
The comparison was:

| Risk score | Mean, correct | Mean, incorrect/ambiguous | AUROC | AUPRC (average precision) |
| --- | ---: | ---: | ---: | ---: |
| Predictive entropy | 0.0654 | 0.3252 | 0.8264 | 0.5913 |
| Cosine distance | 0.0151 | 0.0189 | 0.6528 | 0.4857 |
| Nearest-neighbor distance | 16.5812 | 18.4053 | 0.6806 | 0.4972 |
| Augmentation consistency | 0.0069 | 0.0000 | 0.4792 | 0.2000 |
| Sampled answer disagreement | 0.0255 | 0.1780 | 0.6424 | 0.3278 |

In this pilot, sampled answer disagreement detected errors **worse than
predictive entropy**: AUROC was `0.6424` versus `0.8264`, and AUPRC was
`0.3278` versus `0.5913`. Its AUPRC was above the sample's no-skill prevalence
of `0.20`, but only six examples were errors. The median disagreement score was
zero in both outcome groups, despite the higher error-group mean. The score
took only three values: 27 questions scored `0`, one scored `0.4555`, and two
scored `0.6126`.

The complete risk–coverage curves for all five methods are in the saved SVG and
JSON. For answer disagreement, tied groups produce only these nonempty points:

| Coverage retained | Retained questions | Baseline error rate |
| ---: | ---: | ---: |
| 100.0% | 30 | 20.0% |
| 93.3% | 28 | 17.9% |
| 90.0% | 27 | 14.8% |

The next threshold removes the 27 examples tied at zero, leaving zero coverage,
where risk is undefined. Therefore this score has no tie-preserving operating
point at 80%, 50%, or 33.3% coverage. For comparison, the prior entropy curve
had error rates 8.3%, 6.7%, and 10.0% at those coverages; cosine and
nearest-neighbor distance each had 12.5%, 13.3%, and 10.0%. The earlier
augmentation score had no such operating points either: 29 of its 30 scores
were tied at zero.

## Stable sampled answers on baseline errors

Four of the six original error cases received the same normalized answer in all
five generations, and that answer remained wrong. “Unanimous” here describes
the sampled labels only; no calibrated confidence or answer probability was
measured.

| Question ID | Category | Ground truth | Original baseline label | Five sampled labels | Disagreement score |
| ---: | --- | --- | --- | --- | ---: |
| 1144 | Adversarial | no | yes | yes × 5 | 0.0000 |
| 2079 | Adversarial | yes | no | no × 5 | 0.0000 |
| 2254 | Popular | no | yes | yes × 5 | 0.0000 |
| 2448 | Adversarial | no | yes | yes × 5 | 0.0000 |
| 21 | Random | yes | no | no × 4, ambiguous × 1 | 0.4555 |
| 2270 | Random | no | ambiguous | ambiguous × 3, no × 2 | 0.6126 |

Question 2243 was a correct baseline answer (`yes`) with three sampled `yes`
and two sampled `no` responses, producing a score of `0.6126`. Thus high
disagreement can also occur on a correct answer, while four incorrect answers
were perfectly stable under these five samples. These cases show why sampled
consistency cannot guarantee correctness.

## Saved artifacts

The evaluator saves outputs under Git-ignored
`data/results/answer_disagreement_evaluation/`:

- `pope_answer_disagreement_30_seed_2027_scores.jsonl`: the paired identifiers,
  original baseline prediction and error label, existing four uncertainty
  scores, all five raw and normalized sampled responses, each response seed,
  sampling parameters, and the disagreement score and proportions;
- `pope_answer_disagreement_30_seed_2027_comparison.json`: model and sample
  metadata, all five metric tables, and complete risk–coverage curve points;
- `pope_answer_disagreement_30_seed_2027_comparison.csv`: the five-method
  comparison table;
- `pope_answer_disagreement_30_seed_2027_risk_coverage.svg`: the five-method
  risk–coverage plot.

## Reproduction

With the POPE pilot annotations, 100 pilot images, the existing result files,
and the pinned model already available locally, run from the repository root:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/evaluate_answer_disagreement.py \
  --temperature 0.7 \
  --top-p 0.9 \
  --seed 2027 \
  --max-new-tokens 32 \
  --device mps
```

This performs exactly five sampled generations for each of the fixed 30 pairs
(150 total), reads the prior baseline, entropy, embedding, and augmentation
outputs, and writes results under `data/results/answer_disagreement_evaluation/`.
Offline mode prevents a model-weight download. The script stops if any input
pair, baseline prediction, correctness label, annotation, or augmentation score
does not match the paired sources.

Run checks from the project root with:

```bash
PYTHONPATH=src .venv/bin/pytest -q
PYTHONPATH=src .venv/bin/ruff check src scripts tests
```

## Limitations

- Five generations give a coarse empirical distribution. Different seeds,
  temperatures, top-p values, hardware, and runtime versions can change it.
- Only 30 questions and six error labels were evaluated. AUROC, AUPRC, and
  risk–coverage estimates are uncertain at this sample size and should not be
  treated as evidence of broad generalization.
- The target is POPE baseline answer correctness, not independently reviewed
  hallucination claims. Ambiguous answers count as errors by the inherited
  evaluation convention.
- The parser maps full responses to three labels and ignores textual variation
  within a label. Four erroneous answers remained unanimous in this run.
- A stable wrong answer can score zero; a varied correct answer can score high.
  The score is not a calibrated probability of error or hallucination.
- This experiment uses one model revision, one POPE image split, and one
  sampling configuration. It does not evaluate distribution shifts or
  calibration.
