# POPE reliability under controlled image shifts

## Experiment

This pilot reused exactly the 30 image-question pairs from the existing POPE
baseline and binary-entropy evaluation: 10 questions each from Random, Popular,
and Adversarial, on 30 distinct COCO images. Before inference, the evaluator
verified the baseline and entropy records had matching question IDs, image IDs,
questions, labels, predictions, and model settings. All 30 clean entropy
predictions matched the recorded baseline. Clean predictions and scores were
reused without rerunning or modifying them.

The experiment ran the frozen
`HuggingFaceTB/SmolVLM-500M-Instruct` checkpoint at revision
`a7da5b986cb59b408707209984f360a5f4ad7e47`, using Transformers `5.22.1`,
PyTorch `2.15.0`, MPS, float16, greedy generation, seed `2027`, one beam, and
`max_new_tokens=32`. It evaluated 30 questions under each of 12 shift
conditions (360 transformed examples total). Transformed images were produced
in memory; original COCO files and official POPE labels were not changed.

| Shift | Severity 1 | Severity 2 | Severity 3 |
| --- | --- | --- | --- |
| Gaussian blur | radius 0.5 px | radius 1.5 px | radius 3.0 px |
| Brightness reduction | factor 0.8 | factor 0.6 | factor 0.4 |
| Gaussian noise | standard deviation 5 | standard deviation 15 | standard deviation 30 |
| Contrast reduction | factor 0.8 | factor 0.6 | factor 0.4 |

Noise is sampled independently per pixel and channel from a zero-mean Gaussian
in the 0–255 pixel scale. Its per-image, per-condition seed is derived from the
experiment seed, image ID, shift type, and severity using SHA-256, then recorded
in each result row. The entropy score is normalized entropy over the model’s
conditional likelihoods for the candidate answer sequences `Yes` and `No`; it
is not a probability that the answer is wrong.

For each condition, accuracy counts incorrect, ambiguous, and failed answers
as errors, matching the prior all-question evaluation. AUROC and AUPRC use that
same answer-error label and normalized entropy as a risk ranking. This is a
secondary POPE answer-error analysis, not a direct human annotation of
hallucination claims. AUPRC is shown with error prevalence because its baseline
depends on prevalence.

## Measurements

| Condition | Accuracy | Mean normalized entropy | Error prevalence | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 0.800 | 0.1174 | 0.200 | 0.8264 | 0.5913 |
| Blur, severity 1 | 0.800 | 0.1066 | 0.200 | 0.8403 | 0.7302 |
| Blur, severity 2 | 0.833 | 0.1413 | 0.167 | 0.7120 | 0.3738 |
| Blur, severity 3 | 0.767 | 0.1667 | 0.233 | 0.8509 | 0.6068 |
| Brightness, severity 1 | 0.800 | 0.1126 | 0.200 | 0.8264 | 0.6496 |
| Brightness, severity 2 | 0.800 | 0.1130 | 0.200 | 0.8264 | 0.6597 |
| Brightness, severity 3 | 0.800 | 0.1295 | 0.200 | 0.8333 | 0.5708 |
| Noise, severity 1 | 0.767 | 0.1363 | 0.233 | 0.8758 | 0.8529 |
| Noise, severity 2 | 0.833 | 0.1219 | 0.167 | 0.8080 | 0.5968 |
| Noise, severity 3 | 0.833 | 0.1166 | 0.167 | 0.8000 | 0.3772 |
| Contrast, severity 1 | 0.800 | 0.1137 | 0.200 | 0.8333 | 0.6514 |
| Contrast, severity 2 | 0.800 | 0.1141 | 0.200 | 0.8264 | 0.5458 |
| Contrast, severity 3 | 0.833 | 0.0980 | 0.167 | 0.7520 | 0.3403 |

Every condition had 30 scored entropy values and 30 predictions. There were no
generation or entropy-scoring failures; 348 transformed responses parsed as
Yes/No and 12 were ambiguous. Mean generation time was about `0.920` seconds
per transformed example, with about `1.816` seconds for binary entropy scoring
on this MPS run.

## Interpretation

Entropy remained an error-ranking signal in this pilot: AUROC ranged from
`0.712` to `0.876`, above chance in all 12 conditions, and AUPRC exceeded each
condition’s error prevalence. Its ranking quality still varied by shift and
severity. For example, blur severity 2 had AUROC `0.712` and AUPRC `0.374`,
while noise severity 1 had AUROC `0.876` and AUPRC `0.853`.

Mean entropy did **not** rise consistently with severity. It increased across
the three blur levels and rose slightly at the strongest brightness reduction.
With Gaussian noise, mean entropy fell from `0.1363` at severity 1 to `0.1166`
at severity 3. With contrast reduction it fell to `0.0980` at severity 3,
below the clean mean of `0.1174`. Accuracy also showed no general monotonic
decline: it ranged from `0.767` to `0.833` across conditions, and it improved
relative to clean for some stronger noise and contrast conditions. A corruption
level therefore did not reliably predict either higher entropy or lower
accuracy in this sample.

The results suggest that binary entropy can continue to rank answer errors
under these particular synthetic shifts, but its behavior is condition-specific
and does not track visual severity in a uniform way. The sample is too small to
claim robust generalization or a calibrated error probability.

## Severe-shift answerability review

Before inference, clean/severity-3 contact sheets were generated for all 30
image-question pairs under each perturbation (120 shifted views) and visually
screened side by side. No view was completely blank or wholly obscured in this
screen. Blur, strong noise, and the 0.4 brightness/contrast factors reduced
fine detail; small or low-contrast objects could be difficult to judge in some
views. This was a qualitative screen, not independent item-level adjudication.
The original POPE labels were retained for every condition and no questions
were excluded. Metrics may therefore count errors on examples whose evidence
became ambiguous after corruption.

## Reproduction and outputs

With the prepared pilot images, prior baseline/entropy files, and pinned model
already cached locally, first generate the visual review sheets without
inference:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_distribution_shifts.py \
  --seed 2027 \
  --output-dir data/results/distribution_shift \
  --answerability-sheets-only
```

After reviewing the severity-3 sheets, run the full paired experiment. The
offline flag prevents model downloads; the script resumes completed rows if
interrupted.

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/evaluate_distribution_shifts.py \
  --baseline data/results/pope_baseline_smollm_500m_30_seed_2027.jsonl \
  --clean-evaluation data/results/pope_entropy_30_seed_2027_evaluated.jsonl \
  --image-dir data/images/val2014 \
  --cache-dir "$HOME/.cache/huggingface/halluood" \
  --output-dir data/results/distribution_shift \
  --seed 2027 \
  --device mps
```

Generated files are in Git-ignored `data/results/distribution_shift/`:

- `pope_controlled_shifts_predictions_seed_2027.jsonl` — 360 shifted raw
  predictions, labels, entropy details, per-row timings, and perturbation
  metadata;
- `pope_controlled_shifts_all_conditions_seed_2027.jsonl` — the 30 reused clean
  rows plus all shifted rows;
- `pope_controlled_shifts_evaluation_seed_2027.json` — per-condition metrics,
  parameters, model metadata, and output paths;
- `accuracy_vs_severity.svg` and `entropy_vs_severity.svg`;
- `answerability_review/` — the four clean-versus-severity-3 contact sheets.

## Limitations

- This is one 30-question sample from one model. AUROC/AUPRC estimates are
  uncertain, and the observed error rate varies from 16.7% to 23.3% by
  condition.
- The error target includes any incorrect or ambiguous answer; it does not
  distinguish an unsupported positive object claim from other answer errors.
- POPE object-presence labels are retained after image corruption, including
  views that may be difficult to answer. A future larger evaluation should add
  an independent, item-level answerability audit before interpreting severe
  conditions.
- Controlled blur, brightness, noise, and contrast are synthetic visual
  changes. They do not represent every OOD, domain, or compositional shift.
- Entropy is a ranking score, not a probability of hallucination. These pilot
  results do not establish calibration or generalization.
