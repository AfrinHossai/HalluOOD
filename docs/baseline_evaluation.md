# HalluOOD POPE baseline

## Scope

This command evaluates the frozen SmolVLM checkpoint on a small POPE pilot
sample. It measures yes/no answer accuracy and counts ambiguous responses. This
baseline does not compute uncertainty scores or test distribution shifts.

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
`--max-new-tokens`. Run `scripts/run_baseline.py --help` for the defaults.

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

## Interpretation

The three categories are POPE negative-sampling strategies, not distribution
shift conditions. Accuracy here is ordinary POPE yes/no answer accuracy; it is
not a hallucination-detection score or evidence that an uncertainty signal is
effective. The 30-question pilot is a smoke-test-sized baseline and is too small
to support broad claims about model quality or category difficulty.
