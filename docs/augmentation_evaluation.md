# Augmentation-consistency evaluation on the POPE pilot

## Experiment and paired sample

This exploratory evaluation uses the exact 30 POPE question/image pairs from
the earlier baseline, predictive-entropy, and embedding-distance experiments.
The evaluator joins those result files by question ID and image ID, checks the
question, category, ground truth, original answer, and correctness labels, and
verifies the prepared POPE annotations. The sample has 10 Random, 10 Popular,
and 10 Adversarial questions, with sampling and generation seed `2027`. The
original predictions and labels are read from the existing result files and
remain unchanged. No new model calls were made on the original images.

For each image, the existing Pillow augmentation function produced six
deterministic views: brightness factors 0.90 and 1.10, contrast factors 0.90
and 1.10, and one horizontal and one vertical translation up to 1% of the
image dimension. Translation signs use NumPy PCG64 with seed `2027`; actual
offsets, fill color, view dimensions, and transform parameters are recorded
per view. Images retain their original dimensions and transformed images are
kept in memory only. The run made 180 model calls (six per question).

The model was `HuggingFaceTB/SmolVLM-500M-Instruct`, revision
`a7da5b986cb59b408707209984f360a5f4ad7e47`, with Transformers `5.22.1`,
PyTorch `2.15.0`, MPS, and float16. Generation was greedy (`do_sample=false`),
with one beam, `max_new_tokens=32`, and seed `2027`.

## Score and metrics

For the saved original normalized answer \(\hat y_0\) and the six transformed
view answers \(\hat y_k\), the score is

\[
U_{aug} = \frac{1}{6}\sum_{k=1}^{6}\mathbf{1}[\hat y_k \ne \hat y_0].
\]

Answers are normalized to `yes`, `no`, or `null` for ambiguous/unparseable
responses. `null` is a distinct state, so a transformed answer changing between
`yes`/`no` and `null` counts as disagreement. The score ranges from zero (all
views match the saved original answer) to one (all six differ). It measures
answer instability under these selected image changes; it is not an estimated
probability of correctness or hallucination. Repeating an incorrect answer
still gives a score of zero.

The error target is the same as the prior evaluations: any incorrect or
ambiguous original answer is positive. AUROC ranks those errors against correct
answers; AUPRC is non-interpolated average precision. Risk–coverage curves
exclude the highest-scoring tie group first, then report error rate among the
remaining answers. All four methods below use the exact same paired labels.

## Measured results

SmolVLM answered 24 of 30 questions correctly. Six original predictions were
incorrect or ambiguous (20% prevalence), including one ambiguous answer. The
table reports mean uncertainty among correct and incorrect/ambiguous answers
and error-detection ranking metrics.

| Score (higher treated as more risk) | Mean, correct | Mean, incorrect or ambiguous | AUROC | AUPRC (average precision) |
| --- | ---: | ---: | ---: | ---: |
| Predictive entropy | 0.0654 | 0.3252 | 0.8264 | 0.5913 |
| Cosine distance | 0.0151 | 0.0189 | 0.6528 | 0.4857 |
| Nearest-neighbor distance | 16.5812 | 18.4053 | 0.6806 | 0.4972 |
| Augmentation consistency | 0.0069 | 0.0000 | 0.4792 | 0.2000 |

The no-skill average-precision reference is the error prevalence, `0.20`.
Augmentation consistency performed worse than predictive entropy in this pilot:
its AUROC was `0.4792` versus `0.8264`, and its AUPRC was `0.2000` versus
`0.5913`. Its score was zero for 29 questions and `1/6` for one question. That
one nonzero score belonged to a correct answer. All six incorrect or ambiguous
answers received the same answer on all six transformed views as on the saved
original prediction, so the score did not flag any of them. This is an observed
failure on this pilot, not evidence that augmentation consistency will always
perform poorly.

The prior methods' selected risk–coverage points and the available
augmentation-consistency operating points are:

| Method | Error rate at 100% coverage | At 80% | At 50% | At 33.3% | Other available point |
| --- | ---: | ---: | ---: | ---: | --- |
| Predictive entropy | 20.0% | 8.3% | 6.7% | 10.0% | — |
| Cosine distance | 20.0% | 12.5% | 13.3% | 10.0% | — |
| Nearest-neighbor distance | 20.0% | 12.5% | 13.3% | 10.0% | — |
| Augmentation consistency | 20.0% | — | — | — | 20.7% at 96.7% coverage |

For augmentation consistency, the only nonzero score can be excluded as a
single complete score group, leaving 29 questions and an error rate of `6/29 =
20.7%`. The other 29 scores are tied at zero; excluding that tied group leaves
zero coverage, where risk is undefined. Therefore, the curve has no
tie-preserving operating points at 80%, 50%, or 33.3% coverage. The generated
SVG contains the full risk–coverage curves for the four methods.

## Outputs

The evaluator writes these local artifacts under the Git-ignored
`data/results/augmentation_evaluation/` directory:

- `pope_augmentation_consistency_30_seed_2027_scores.jsonl`: one row per
  question, including the original labels, all six transformed raw and
  normalized answers, timing, view names and parameters, and uncertainty score;
- `pope_augmentation_consistency_30_seed_2027_comparison.json`: sample/model
  metadata, aggregate scores, and complete risk–coverage points;
- `pope_augmentation_consistency_30_seed_2027_comparison.csv`: four-method
  comparison table;
- `pope_augmentation_consistency_30_seed_2027_risk_coverage.svg`: plot of the
  four risk–coverage curves.

## Reproduction

From the repository root, with the existing POPE annotations, pilot images, and
model checkpoint already available locally, run:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/evaluate_augmentation_pilot.py \
  --device mps \
  --seed 2027
```

`HF_HUB_OFFLINE=1` prevents a weight download. The script checks the fixed
sample identities, original predictions and labels, model metadata, and
category counts before it makes inference calls. It writes only to
`data/results/augmentation_evaluation/` by default.

Run the automated checks from the repository root:

```bash
PYTHONPATH=src .venv/bin/pytest -q
PYTHONPATH=src .venv/bin/ruff check src scripts tests
```

## Limitations

- There are only 30 questions and six error labels, including one ambiguous
  answer. AUROC, AUPRC, and risk–coverage estimates are unstable at this size.
- POPE answer correctness is used as the error target; no independent
  claim-level hallucination review was performed.
- Brightness, contrast, and small translations may preserve the evidence, in
  which case even an incorrect answer can remain stable. A transform can also
  hide a small object near an edge. The views were not separately adjudicated
  for evidence preservation.
- The method maps full generated responses to `yes`, `no`, or ambiguous and
  does not measure wording or verbal confidence. Stability does not guarantee
  correctness, and this score is not a calibrated hallucination probability.
- This is one model revision, one POPE validation-image pilot, and one fixed
  augmentation set. It does not establish generalization to other models,
  images, transformations, or distribution shifts.
- Six additional generations per question add inference time. The saved
  per-view durations are for this local run and are hardware- and runtime-
  dependent.
