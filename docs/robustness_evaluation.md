# Expanded POPE evaluation and robustness study

## Scope and status

This evaluation expands the existing local POPE pilot to all 1,800 available
questions: 600 each in the Random, Popular, and Adversarial question
categories, over the same 100 COCO validation images. It evaluates deterministic
SmolVLM answers and binary predictive entropy on clean images. It also compares
those results with the exact 30-question pilot, estimates image-grouped
confidence intervals, tests entropy-based abstention on a disjoint image split,
and runs the existing controlled image perturbations on a small fixed subset.

The target called an **error** below is any incorrect, ambiguous, or failed
yes/no response. This is a POPE answer-correctness proxy, not an independently
annotated claim-level hallucination label. The entropy score is conditional
entropy over the fixed candidate answer sequences `Yes` and `No`; it is not
full-vocabulary token entropy and is not a probability of hallucination.

## Dataset, model, and decoding

The runner validated the three prepared annotation files before inference. Each
category contained 600 unique questions on the same 100 images, with six
questions per image and category. Every referenced image existed locally and
was readable. Each raw result row preserves the source question, image ID,
category, raw and normalized ground truth, raw and normalized model response,
parse status, correctness, binary entropy details, and timings.

The model is `HuggingFaceTB/SmolVLM-500M-Instruct`, pinned to revision
`a7da5b986cb59b408707209984f360a5f4ad7e47`. Decoding matches the baseline:
greedy generation, seed `2027`, one beam, and `max_new_tokens=32`. The run uses
the locally cached Transformers integration, MPS, and float16. No model weights,
images, annotations, or raw results are added to Git.

For each question, the runner records the original greedy response and then
computes binary entropy from the model's actual likelihoods for the complete
candidate sequences `Yes` and `No`. The two sequence log-likelihoods are
normalized over that binary candidate set. Entropy is divided by `log(2)` to
produce a score from zero to one. Ambiguous responses still receive an entropy
score when generation succeeds.

## Clean-image results

The 1,800-row run completed with 1,800 predictions and 1,800 entropy scores;
there were no prediction or entropy failures. MPS inference took 5,648 seconds
(about 94 minutes). The table reports accuracy
over all questions, ambiguous-response rate over all questions, and the
error-detection ranking of entropy among scored questions. Higher entropy is
treated as greater error risk. AUPRC is non-interpolated average precision and
must be compared with the error prevalence.

| POPE category | Questions | Accuracy (95% CI) | Ambiguous rate (95% CI) | Error prevalence | Mean entropy (95% CI) | AUROC (95% CI) | AUPRC (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 600 | 0.878 (0.853–0.903) | 0.053 (0.037–0.072) | 0.122 | 0.1248 (0.1079–0.1431) | 0.769 (0.711–0.818) | 0.344 (0.251–0.453) |
| Popular | 600 | 0.842 (0.812–0.868) | 0.010 (0.003–0.020) | 0.158 | 0.1657 (0.1460–0.1863) | 0.819 (0.781–0.857) | 0.419 (0.337–0.534) |
| Adversarial | 600 | 0.805 (0.775–0.835) | 0.028 (0.015–0.042) | 0.195 | 0.1981 (0.1772–0.2208) | 0.787 (0.748–0.824) | 0.457 (0.369–0.548) |
| Overall | 1,800 | 0.842 (0.819–0.866) | 0.031 (0.021–0.041) | 0.158 | 0.1629 (0.1470–0.1797) | 0.798 (0.763–0.830) | 0.410 (0.337–0.497) |

Image-grouped percentile intervals use 1,000 bootstrap replicates and seed
`2027`. Each replicate samples image IDs with replacement and carries every
question for a selected image together. The intervals quantify variation over
these image groups; they are not significance tests. They do not cover model
revision, dataset, or prompt uncertainty.

No failed bootstrap replicates occurred for these metrics. The AUPRC baseline
is the error prevalence in the same category: 0.122 Random, 0.158 Popular,
0.195 Adversarial, and 0.158 overall.

## Comparison with the original pilot

The original pilot contains 30 questions on 30 distinct images (10 per
category). It is nested inside the 1,800-row dataset. The expanded runner
checks that every pilot question, image, label, raw prediction, normalized
prediction, parse status, correctness label, and entropy value is reproduced
before calculating the comparison. The difference is descriptive; the samples
are not independent and no significance claim is made.

| Metric | Original 30-question pilot | Expanded 1,800-question evaluation | Difference |
| --- | ---: | ---: | ---: |
| Accuracy | 0.8000 | 0.8417 | +0.0417 |
| Ambiguous-response rate | 0.0333 | 0.0306 | −0.0028 |
| Mean normalized entropy | 0.1174 | 0.1629 | +0.0455 |
| AUROC | 0.8264 | 0.7983 | −0.0280 |
| AUPRC | 0.5913 | 0.4103 | −0.1809 |

The pilot's 10-per-category point estimates were volatile: Random AUROC/AUPRC
changed from 0.4375/0.3500 to 0.7693/0.3440; Popular from 1.0000/1.0000 to
0.8195/0.4186; and Adversarial from 0.9048/0.8056 to 0.7872/0.4570. These
descriptive changes are expected to be sensitive to the six or fewer errors in
each pilot stratum; the nested samples do not justify an independent-sample
significance test.

## Calibrated abstention and image separation

The existing 50-image embedding reference bank has 50 image IDs in common
with the required 100-image clean POPE evaluation. Predictive entropy does not
use the embedding bank, so the full 1,800-row entropy-only summary is retained
as requested and does not use any reference embeddings. The original 30-row
pilot comparison is checked to be disjoint from the reference IDs. To prevent
reference leakage in the new calibration and held-out selective evaluation,
all reference IDs are first excluded; the remaining 50 POPE images are split by
PCG64 seed `2027` into 10 calibration and 40 held-out evaluation
images. Every category and question for an image remains together. The split
manifest records all three image ID sets and overlap counts.

For each target coverage (50%, 70%, 80%, and 90%), the threshold is selected
only from calibration entropy values. The procedure chooses the observed
cutoff with closest empirical coverage, keeps ties together, and prefers lower
coverage on an equal-distance tie. The calibration function receives only
calibration image IDs and entropy values plus the held-out image IDs; it does
not receive evaluation scores or labels. Evaluation labels are used only after
the threshold is fixed to calculate coverage, selective risk, and abstention
rate. This is a coverage-matching procedure, not a formal calibration
guarantee.

| Target calibration coverage | Threshold | Achieved calibration coverage | Held-out coverage (95% CI) | Selective risk (95% CI) | Abstention rate (95% CI) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50% | 0.036220 | 50.6% | 53.8% (48.9–58.2%; 387/720) | 6.7% (3.5–10.5%; 26/387) | 46.3% (41.8–51.1%) |
| 70% | 0.125281 | 70.0% | 72.8% (68.8–76.1%; 524/720) | 7.6% (5.1–10.6%; 40/524) | 27.2% (23.9–31.2%) |
| 80% | 0.248957 | 80.6% | 82.5% (78.8–85.8%; 594/720) | 9.8% (6.9–13.2%; 58/594) | 17.5% (14.2–21.2%) |
| 90% | 0.645006 | 89.4% | 89.4% (86.5–92.1%; 644/720) | 11.6% (8.7–15.1%; 75/644) | 10.6% (7.9–13.5%) |

These selective intervals use 1,000 image-grouped bootstrap replicates on the
40 held-out images with each calibrated threshold frozen; they do not refit the
threshold inside a replicate.

Without abstention, the held-out 40-image group had 123 errors among 720
questions (17.1% risk). In this split, the selected thresholds yielded lower
observed selective risk at each reported coverage. This is a descriptive result
on one calibration/evaluation partition, not a formal guarantee.

## Controlled-shift robustness subset

The expanded shift experiment selects three image IDs deterministically from
the held-out evaluation images using PCG64 seed `2028`. It includes all 18
POPE questions for each selected image (six per category), for 54 clean
image-question pairs. Each pair is evaluated under all 12 existing conditions:
Gaussian blur, brightness reduction, Gaussian noise, and contrast reduction,
each at its existing severity levels 1, 2, and 3. This is 648 transformed
question evaluations in addition to the 54 clean references. The original
image files and POPE labels remain unchanged. Transformations happen in memory;
parameters and per-image-condition noise seeds are saved with every row.
The selected IDs were `000000191964`, `000000199940`, and `000000304819`.
The run completed all 648 generations and entropy scores without failures. The
logged per-row averages were 0.860 seconds for generation and 1.695 seconds for
binary entropy scoring, summing to 1,656 seconds of per-example inference time
(about 27.6 minutes, excluding model loading and summary/bootstrap work).

| Shift | Severity 1 | Severity 2 | Severity 3 |
| --- | --- | --- | --- |
| Gaussian blur | radius 0.5 px | radius 1.5 px | radius 3.0 px |
| Brightness reduction | factor 0.8 | factor 0.6 | factor 0.4 |
| Gaussian noise | standard deviation 5 | standard deviation 15 | standard deviation 30 |
| Contrast reduction | factor 0.8 | factor 0.6 | factor 0.4 |

For the fixed 54-pair subset, the report compares accuracy, mean normalized
entropy, AUROC, and AUPRC for clean images and every shifted condition. The
same original labels define errors throughout. In particular, synthetic
corruption can remove evidence needed to answer a question; no new human
answerability labels are available, so these measurements must not be read as
isolating model reliability from corruption-induced unanswerability.

The clean subset had 44/54 correct answers (accuracy 0.815) and a 0.185 error
prevalence. Image-grouped intervals below resample the three selected images,
keeping all 18 questions for each sampled image together. These intervals are
especially coarse because there are only three clusters. AUROC/AUPRC intervals
omit resamples with only one outcome class; the valid replicate count is shown.

| Condition | Accuracy (95% CI) | Mean entropy (95% CI) | Error prevalence |
| --- | ---: | ---: | ---: |
| Clean | 0.815 (0.722–0.907) | 0.180 (0.129–0.229) | 0.185 |
| Blur 1 | 0.815 (0.722–0.907) | 0.157 (0.103–0.195) | 0.185 |
| Blur 2 | 0.778 (0.556–0.926) | 0.233 (0.200–0.295) | 0.222 |
| Blur 3 | 0.685 (0.556–0.778) | 0.191 (0.160–0.231) | 0.315 |
| Brightness 1 | 0.815 (0.722–0.907) | 0.206 (0.110–0.260) | 0.185 |
| Brightness 2 | 0.759 (0.722–0.796) | 0.199 (0.130–0.244) | 0.241 |
| Brightness 3 | 0.704 (0.685–0.722) | 0.187 (0.152–0.234) | 0.296 |
| Noise 1 | 0.759 (0.556–0.907) | 0.192 (0.152–0.221) | 0.241 |
| Noise 2 | 0.704 (0.685–0.722) | 0.163 (0.103–0.213) | 0.296 |
| Noise 3 | 0.704 (0.556–0.796) | 0.238 (0.134–0.316) | 0.296 |
| Contrast 1 | 0.815 (0.722–0.907) | 0.212 (0.088–0.298) | 0.185 |
| Contrast 2 | 0.759 (0.722–0.796) | 0.191 (0.095–0.243) | 0.241 |
| Contrast 3 | 0.759 (0.722–0.796) | 0.170 (0.095–0.209) | 0.241 |

| Condition | AUROC (95% CI; valid replicates) | AUPRC (95% CI; valid replicates) | Error prevalence |
| --- | ---: | ---: | ---: |
| Clean | 0.602 (0.437–0.908; 976/1,000) | 0.264 (0.113–0.831; 976/1,000) | 0.185 |
| Blur 1 | 0.566 (0.433–0.862; 976/1,000) | 0.232 (0.109–0.687; 976/1,000) | 0.185 |
| Blur 2 | 0.540 (0.245–0.739; 976/1,000) | 0.263 (0.075–0.596; 976/1,000) | 0.222 |
| Blur 3 | 0.827 (0.512–0.964; 1,000/1,000) | 0.585 (0.502–0.847; 1,000/1,000) | 0.315 |
| Brightness 1 | 0.534 (0.384–0.877; 976/1,000) | 0.216 (0.103–0.698; 976/1,000) | 0.185 |
| Brightness 2 | 0.563 (0.492–0.815; 1,000/1,000) | 0.282 (0.269–0.711; 1,000/1,000) | 0.241 |
| Brightness 3 | 0.572 (0.354–0.757; 1,000/1,000) | 0.376 (0.273–0.614; 1,000/1,000) | 0.296 |
| Noise 1 | 0.728 (0.690–0.908; 976/1,000) | 0.568 (0.231–0.831; 976/1,000) | 0.241 |
| Noise 2 | 0.651 (0.538–0.765; 1,000/1,000) | 0.468 (0.351–0.674; 1,000/1,000) | 0.296 |
| Noise 3 | 0.605 (0.579–0.662; 1,000/1,000) | 0.364 (0.329–0.584; 1,000/1,000) | 0.296 |
| Contrast 1 | 0.532 (0.396–0.877; 976/1,000) | 0.214 (0.105–0.698; 976/1,000) | 0.185 |
| Contrast 2 | 0.610 (0.565–0.785; 1,000/1,000) | 0.304 (0.287–0.578; 1,000/1,000) | 0.241 |
| Contrast 3 | 0.604 (0.535–0.754; 1,000/1,000) | 0.300 (0.252–0.600; 1,000/1,000) | 0.241 |

In this small subset, accuracy fell under the strongest blur and brightness
conditions and under each noise level, while contrast severity 2 and 3 shared
the same accuracy. Mean entropy did not move monotonically with severity: it
fell across brightness and contrast levels, varied non-monotonically with
noise, and peaked at blur severity 2 rather than severity 3. Error ranking was
condition-dependent. AUROC ranged from 0.532 to 0.827; each point estimate was
above its condition's error prevalence for AUPRC, but the intervals are wide
and several AUROC intervals include chance. These results do not establish
that entropy remains reliably useful under degradation; they show mixed
pilot-scale behavior that needs more image groups.

## Method coverage at each scale

| Method | Scale evaluated in this task |
| --- | --- |
| Deterministic POPE prediction + binary entropy | All 1,800 questions on 100 images |
| Entropy-based calibrated abstention | 10 calibration and 40 held-out images, after excluding reference IDs |
| Controlled image shifts with binary entropy | 54 held-out clean pairs and 648 shifted evaluations |
| Embedding distance | Existing 30-question pilot only; not recalculated at expanded scale |
| Augmentation consistency | Existing 30-question pilot only |
| Sampled answer disagreement | Existing 30-question pilot only |

The earlier 30-question controlled-shift results remain a separate pilot and
are not pooled with the expanded 54-pair shift subset.

## Artifacts and reproduction

All generated artifacts are local and Git-ignored under
`data/results/robustness/`:

- `full_pope_predictions_seed_2027.jsonl` — resumable raw predictions and
  entropy details for all 1,800 questions;
- `full_pope_predictions_seed_2027_run_summary.json` — counts, model metadata,
  and failures;
- `expanded_robustness_summary_seed_2027.json` — clean/category metrics,
  pilot comparison, bootstrap intervals, abstention results, and split details;
- `expanded_image_bootstrap_ci_seed_2027.json` and
  `expanded_image_split_seed_2027.json` — complete interval and split manifests;
- `accuracy_by_category.svg`, `ambiguous_rate_by_category.svg`,
  `auroc_by_category.svg`, `auprc_by_category.svg`, and `risk_coverage.svg`;
- `expanded_shift_predictions_seed_2027.jsonl`,
  `expanded_shift_summary_seed_2027.json`,
  `expanded_shift_accuracy_vs_severity.svg`, and
  `expanded_shift_entropy_vs_severity.svg`.

Run from the repository root with the prepared annotations, local COCO images,
and pinned checkpoint already available. Offline mode prevents model downloads.
The full inference runner appends each completed row and resumes completed,
matching rows after interruption.

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/run_full_pope.py \
  --device mps --seed 2027 --max-new-tokens 32

PYTHONPATH=src .venv/bin/python scripts/analyze_robustness.py \
  --seed 2027 --bootstrap-replicates 1000 \
  --target-coverages 0.5,0.7,0.8,0.9

HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/evaluate_expanded_shifts.py \
  --device mps --seed 2027 --subset-seed 2028 --image-count 3

PYTHONPATH=src .venv/bin/pytest -q
.venv/bin/ruff check src scripts tests
```

## Limitations

- POPE correctness is a proxy for object-presence answer reliability, not
  independent annotation of unsupported claims. The ambiguous rate is reported
  separately; ambiguous and failed answers count as errors for accuracy.
- Binary Yes/No sequence entropy differs from full-vocabulary predictive
  entropy and can be influenced by candidate tokenization and sequence length.
  It is a ranking score, not a probability of hallucination.
- The 30-question comparison is nested in the expanded run, so its difference
  is descriptive. Image bootstrap intervals do not establish statistical
  significance or generalization to other datasets, models, or shifts.
- The expanded shift subset has only three images and 54 paired questions.
  Its estimates will be noisy. Blur, brightness, noise, and contrast are
  synthetic changes and do not represent all forms of distribution shift.
- Strong corruption can make some questions visually unanswerable. The
  original POPE labels are retained, and no item-level answerability audit was
  performed for the new three-image subset.
- The full entropy-only summary includes images that also appear in the
  separate embedding reference bank because all 100 POPE images were required.
  The bank is not used for these predictions or entropy scores. Abstention and
  expanded shift results exclude all reference IDs. The checkpoint's original
  pretraining overlap with COCO is not established by this experiment.
- Thresholds match observed calibration coverage only. The calibration split
  contains few image groups and no formal risk or calibration guarantee follows.
