# HalluOOD POPE baseline

## Scope

This command evaluates the frozen SmolVLM checkpoint on a small POPE pilot
sample. By default, it measures yes/no answer accuracy and counts ambiguous
responses without computing uncertainty scores or testing distribution shifts.
The opt-in `--record-uncertainty` flag also computes conditional binary answer
entropy from model logits.

The inputs are the three prepared files in `data/processed/`:

- `coco_pope_random_pilot.jsonl`
- `coco_pope_popular_pilot.jsonl`
- `coco_pope_adversarial_pilot.jsonl`

The sampler uses NumPy's PCG64 generator with seed `2027` by default. It gives
the three categories equal quotas where possible, groups questions by image,
prefers distinct image IDs across categories, and chooses one question row per
selected image/category group. On the prepared 100-image pilot, the default
sample is 30 questions, 10 from each category, with 30 distinct images.

## Reproduction

From the repository root, install the optional local-inference dependencies
and run the baseline:

```bash
python3 -m uv sync --extra inference
.venv/bin/python scripts/run_baseline.py \
  --sample-size 30 \
  --seed 2027 \
  --device auto
```

The script uses the pinned `HuggingFaceTB/SmolVLM-500M-Instruct` revision in
`src/halluood/model.py`, the prepared COCO pilot images under
`data/images/val2014/`, and the local Hugging Face cache. If model weights are
already cached, `HF_HUB_OFFLINE=1` can be prefixed to the run command to prevent
Hub requests. The checkpoint, COCO images, annotation files, and results are
local files and are excluded from Git by the `data/` ignore rule.

Useful options include `--annotations-dir`, `--image-dir`, `--output`,
`--cache-dir`, `--sample-size`, `--seed`, `--device`, and
`--max-new-tokens`. To include the binary answer entropy diagnostic, add
`--record-uncertainty`; this adds two teacher-forced model forward passes per
question. Run `scripts/run_baseline.py --help` for the defaults.

## Result files and answer parsing

The default per-question JSONL output is
`data/results/pope_baseline_smollm_500m_30_seed_2027.jsonl`. A companion
`*_summary.json` file contains overall and category metrics. Each row preserves
the raw generated response, normalized yes/no prediction (or `null`), raw and
normalized gold label, question and image identifiers, category, model/checkpoint
and library versions, generation settings, device, precision, and inference time.

The parser accepts a leading yes/no answer and clear direct presence/absence
phrases such as “There is no chair present in the image.” Mixed yes/no claims,
unanchored responses, and uncertain statements such as “cannot be determined”
are marked `ambiguous`; their raw response remains available in the JSONL. The
all-question accuracy denominator is the full sample, so ambiguous and failed
responses count as incorrect. Accuracy on parsed responses is also reported
separately. Failed inference rows are retained with a null prediction and an
error description.

When uncertainty recording is enabled, each successfully generated row also
contains `uncertainty_score` and `uncertainty_time_seconds`. The score object
records candidate log probabilities, actual token IDs and counts, normalized
Yes/No probabilities, entropy in nats, and entropy divided by `log(2)`.
Uncertainty-scoring errors are stored separately from generation errors and do
not change the generated prediction.

## Binary answer entropy

For the fixed candidate strings `Yes` and `No`, the adapter tokenizes each full
assistant answer with the loaded multimodal chat template. This supports
tokenizers where either answer uses multiple tokens. It verifies that each
candidate sequence begins with the same image-and-question generation prompt.
For each candidate token, a teacher-forced forward pass obtains the model's
actual next-token logits. A vocabulary-wide log-softmax gives that token's
log-probability; the adapter sums those values across the candidate's text
tokens. Trailing chat end markers are excluded. The two summed sequence log
probabilities are then normalized with a stable two-choice softmax to form a
distribution conditioned on the candidate set `{Yes, No}`. Shannon entropy is
reported in nats, with `H / log(2)` as a normalized value from 0 to 1. A
balanced candidate distribution has normalized entropy 1; a concentrated one
approaches 0.

This is **binary candidate-sequence entropy**, not full-vocabulary next-token
entropy. The research plan's primary predictive-entropy definition is the mean
full-vocabulary entropy over generated answer-token positions. The helper
`entropy_from_logits` supports full-vocabulary entropy calculations, but this
opt-in POPE path does not currently aggregate those values over generated
tokens. The binary score depends on the exact candidate strings, prompt
template, tokenizer, and checkpoint; comparing Yes and No sequence likelihoods
can also be affected by their different token counts. It measures relative
uncertainty between these two fixed answer candidates, **not the probability
that an answer is hallucinated**. Calibration and held-out evaluation would be
required to assess hallucination prediction.

## Interpretation

The three categories are POPE negative-sampling strategies, not distribution
shift conditions. Accuracy here is ordinary POPE yes/no answer accuracy; it is
not a hallucination-detection score or evidence that an uncertainty signal is
effective. The 30-question pilot is a smoke-test-sized baseline and is too small
to support broad claims about model quality or category difficulty.
