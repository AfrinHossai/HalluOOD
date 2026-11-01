# Binary entropy evaluation on the POPE pilot

## Experiment

This evaluation used the same 30 question IDs as the prior POPE baseline:
10 each from the Random, Popular, and Adversarial categories, selected with
PCG64 seed `2027`. The evaluation script checked that the entropy run and
baseline contained the same category, question, and image identities. All 30
generated predictions also matched the baseline predictions.

The frozen checkpoint was `HuggingFaceTB/SmolVLM-500M-Instruct` at revision
`a7da5b986cb59b408707209984f360a5f4ad7e47`, using Transformers `5.22.1`,
PyTorch `2.15.0`, MPS, and float16. Generation matched the baseline settings:
greedy decoding, seed `2027`, one beam, and `max_new_tokens=32`. Binary entropy
was recorded for all 30 questions, with no inference or scoring failures.

The error target is any incorrect or ambiguous answer, following the baseline's
all-question accuracy convention. Entropy is the normalized entropy of the
conditional Yes/No candidate-sequence distribution; larger values are treated
as higher error risk. This is not a calibrated probability of error or
hallucination.

## Measurements

The model answered 24 of 30 questions correctly (80%); six answers were
incorrect, including one ambiguous response. Mean and median normalized entropy
were higher among the incorrect answers:

| Outcome | Questions | Mean entropy | Median entropy |
| --- | ---: | ---: | ---: |
| Correct | 24 | 0.0654 | 0.0133 |
| Incorrect or ambiguous | 6 | 0.3252 | 0.1662 |

Using entropy to rank errors gave **AUROC 0.8264** and **AUPRC 0.5913**. The
incorrect-answer prevalence was **0.20**, the no-skill reference level for
average precision on this sample. Scores did not perfectly separate the
outcomes: at least one correct answer had very high entropy, and at least one
incorrect answer had very low entropy.

| Coverage retained | High-entropy questions excluded | Error rate among retained |
| ---: | ---: | ---: |
| 100% | 0 | 20.0% |
| 80% | 6 | 8.3% |
| 50% | 15 | 6.7% |
| 33.3% | 20 | 10.0% |

The curve excludes complete tied-score groups at each threshold; at zero
coverage the error rate is undefined. The plotted curve and per-question
measurements are saved locally under the Git-ignored `data/results/` directory:

- `pope_entropy_30_seed_2027.jsonl` — model responses and entropy scores;
- `pope_entropy_30_seed_2027_evaluated.jsonl` — rows with correctness and
  normalized entropy;
- `pope_entropy_30_seed_2027_evaluation.json` — aggregate metrics and curve
  points;
- `pope_entropy_30_seed_2027_risk_coverage.svg` — risk–coverage chart.

## Reproduction

With the prepared pilot data, COCO pilot images, and pinned model cached locally,
run from the repository root:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/run_baseline.py \
  --sample-size 30 \
  --seed 2027 \
  --device mps \
  --max-new-tokens 32 \
  --record-uncertainty \
  --output data/results/pope_entropy_30_seed_2027.jsonl \
  --cache-dir "$HOME/.cache/huggingface/halluood"

PYTHONPATH=src .venv/bin/python scripts/evaluate_entropy.py
```

The evaluator checks the 30 question/image identities and model, device,
precision, software versions, seed, and generation settings against the
baseline result before calculating metrics. Entropy scoring adds two
teacher-forced model forward passes per question.

## Interpretation and limitations

In this particular sample, higher entropy was associated with incorrect
predictions: the error group had higher mean and median entropy, and AUROC was
above 0.5. The risk–coverage curve also shows lower error rates after excluding
the highest-entropy cases at 80% and 50% coverage. The error rate rises again at
33.3% coverage, so the relationship is not perfectly ordered.

These 30 questions are a small, single-model POPE pilot, not held-out evidence
of generalization. AUROC and AUPRC are uncertain with only six incorrect cases;
the category-specific groups are smaller still. The target combines ordinary
answer errors and an ambiguous answer, rather than labeling visual
hallucinations through independent claim review. The binary candidate entropy
measures relative likelihood of the fixed `Yes` and `No` strings, not the
full-vocabulary token entropy specified as the primary entropy definition in
the research plan. It should be treated as an exploratory signal that needs
larger held-out testing across models and shifts.
