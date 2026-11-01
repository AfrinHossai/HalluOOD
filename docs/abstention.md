# Predictive-entropy abstention

## Method

The selective-prediction helper in `src/halluood/selective.py` accepts the
normalized binary predictive entropy already produced by the entropy scorer.
It answers when entropy is at or below a chosen threshold and abstains when it
is above the threshold. The entropy is the normalized entropy of the model's
conditional likelihoods for the fixed `Yes` and `No` answer sequences. It is an
uncertainty score, not a probability that an answer is wrong.

Threshold selection uses only entropy values from a calibration subset. For a
target coverage, it considers each observed calibration score as a cutoff and
chooses the cutoff whose empirical answered fraction is closest to the target.
Every item with a tied score receives the same decision. If two attainable
coverages are equally close, the lower coverage is selected. Consequently, ties
can make the achieved calibration coverage differ from the requested target.
The selection API receives only evaluation image IDs for the leakage check; it
does not receive evaluation scores or correctness labels. Correctness labels
are used only afterward to report selective risk.

The procedure matches **coverage**, not a target risk. It does not provide a
formal calibration or selective-risk guarantee.

## Reproducible local pilot

The real smoke run used the existing entropy evaluation sample of 30 POPE
questions on 30 distinct images. It selected 20 calibration questions using
the PCG64 category-balanced sampler and seed `2027`, with one question per
image: seven Random, seven Popular, and six Adversarial. The sampler excluded
all 30 evaluation image IDs before selecting calibration rows. The two sets
had zero image overlap. Calibration questions and scores are saved locally in
the Git-ignored `data/results/abstention/` directory; calibration labels are
not included in that output.

The calibration entropy scores came from
`HuggingFaceTB/SmolVLM-500M-Instruct`, revision
`a7da5b986cb59b408707209984f360a5f4ad7e47`, using Transformers `5.22.1`,
PyTorch `2.15.0`, MPS, and float16. The final evaluation predictions and
correctness labels were reused without rerunning or changing the baseline.
There were six incorrect or ambiguous answers among the 30 evaluation rows
(20% error rate before abstention).

| Target calibration coverage | Entropy threshold | Achieved calibration coverage | Evaluation coverage | Evaluation selective risk | Evaluation abstention rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50% | 0.035745 | 50% | 70% | 4.8% (1/21) | 30% |
| 70% | 0.063853 | 70% | 73.3% | 4.5% (1/22) | 26.7% |
| 80% | 0.243382 | 80% | 90% | 14.8% (4/27) | 10% |
| 90% | 0.387414 | 90% | 90% | 14.8% (4/27) | 10% |

The realized evaluation coverage differs from target calibration coverage
because the evaluation entropy distribution differs from the 20-image
calibration distribution. On this small sample, the first two thresholds
excluded five of the six errors while also abstaining on several correct
answers. At the 80% and 90% calibration targets, four errors remained among
the 27 answered items. These are descriptive pilot measurements, not evidence
of performance on other splits, models, or distribution shifts.

Outputs from the run:

- `data/results/abstention/pope_abstention_calibration_20_seed_2027.jsonl` —
  selected calibration questions, entropy scores, scoring metadata, and no
  ground-truth labels;
- `data/results/abstention/pope_abstention_20cal_30eval_seed_2027.json` —
  thresholds, split identifiers and image IDs, overlap check, and evaluation
  metrics and decisions.

Reproduce from the repository root after the local annotations, 100 pilot COCO
images, cached checkpoint, and existing entropy evaluation files are present:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/evaluate_abstention.py \
  --evaluation data/results/pope_entropy_30_seed_2027_evaluated.jsonl \
  --annotations-dir data/processed \
  --image-dir data/images/val2014 \
  --output-dir data/results/abstention \
  --calibration-size 20 \
  --seed 2027 \
  --target-coverages 0.5,0.7,0.8,0.9 \
  --cache-dir "$HOME/.cache/huggingface/halluood" \
  --device mps
```

`HF_HUB_OFFLINE=1` prevents network access. The selected records, calibration
seed, model revision, scoring settings, thresholds, dataset identifiers, and
image IDs are written to the ignored results directory.

## Limitations

- The calibration set has only 20 images and the final evaluation has 30. The
  observed risk can vary substantially with a few questions.
- Coverage-based thresholding does not use error labels and does not promise a
  desired error rate. Selective risk is an observed held-out metric only.
- The small calibration sample is from the same POPE pilot source and does not
  represent the distribution shifts in the broader research plan.
- Image-level separation prevents the same image from crossing the split, but
  does not rule out semantic or near-duplicate overlap across datasets.
- Entropy can be high on a correct answer and low on an incorrect one. An
  abstention rule may therefore reject useful answers or retain errors.
- The current evaluator requires one evaluation question per image so the
  image-ID split and recorded evaluation identity remain exact.
