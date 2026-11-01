# Sampled answer-disagreement uncertainty

## Definition

This method asks a frozen vision-language model the same image question several
times using controlled stochastic decoding. Each raw response is passed through
the existing POPE baseline answer normalizer and mapped to one of three labels:
`yes`, `no`, or `ambiguous`. Unparseable responses are kept in the ambiguous
group instead of being discarded or assigned to yes/no.

For `N` sampled responses, let `p_yes`, `p_no`, and `p_ambiguous` be the observed
proportions of those labels. The answer-disagreement score is normalized
empirical entropy:

\[
D = -\frac{1}{\log 3}\sum_{c \in \{yes,no,ambiguous\}} p_c\log p_c,
\]

where a zero-proportion term contributes zero. The score is between 0 and 1:
unanimous answers (including unanimous ambiguous answers) score 0, and an equal
mix of all three labels scores 1. The proportions and counts are saved with the
score. This is entropy of sampled, normalized responses; it is different from
the model's next-token or candidate-sequence entropy. It measures response
variation under the chosen decoding settings, not a probability of error or
hallucination. Repeatedly generating the same incorrect answer will still
produce a score of zero.

## Reproducible sampling

The new `SamplingSettings` are separate from the existing deterministic
`GenerationSettings`. Existing `generate_answer` calls remain greedy and
deterministic by default. Repeated sampling sets `do_sample=true`, uses one beam,
and records the number of generations, temperature, top-p, maximum new tokens,
model identifier/revision, base seed, and per-response seeds. Response `i` uses
`base_seed + i`, where `i` starts at zero; the torch random generator is seeded
before each generation. A runtime failure aborts the sampling call rather than
returning a partial list. Empty or wrong-length outputs are rejected before a
result file is written.

The one-pair runner reuses the question, image, and original answer already in
the saved POPE baseline. It verifies those fields against the prepared POPE
annotation, but does not regenerate, edit, or score the original baseline
prediction. Sampled responses and output metadata are saved locally under the
Git-ignored `data/results/` directory.

## Real-image smoke test

The smoke test used the existing Random-category baseline row for question
`2813`, image `COCO_val2014_000000396068.jpg`, “Is there a remote in the image?”
The ground-truth label was `yes`, and the unchanged saved baseline answer was
“Yes, there is a remote in the image.”

| Setting or observation | Value |
| --- | --- |
| Model | `HuggingFaceTB/SmolVLM-500M-Instruct` |
| Model revision | `a7da5b986cb59b408707209984f360a5f4ad7e47` |
| Runtime | Transformers 5.22.1, PyTorch 2.15.0, MPS, float16 |
| Sampled generations | 5 |
| Temperature / top-p | 0.7 / 0.9 |
| Base seed | 2027; response seeds 2027–2031 |
| Raw responses | “Yes, there is a remote in the image.” (all five) |
| Observed proportions | yes 1.0, no 0.0, ambiguous 0.0 |
| Answer-disagreement score | **0.0** |
| Total sampled inference time | 5.95 seconds |

The sampled answer distribution was unanimous and matched the saved baseline
answer. Since this is one image-question pair, it demonstrates that the sampled
generation path ran and recorded its settings; it does not establish that a
zero score indicates correctness or that the score predicts errors.

The raw per-response outputs and complete JSON record are stored at
`data/results/answer_disagreement_q2813_seed_2027.json`. The output includes
each raw response, normalized label, parser status, generation index and seed,
response counts and proportions, baseline answer provenance, model revision,
sampling settings, and total inference time.

## Reproduction

With the prepared POPE annotation, COCO pilot image, saved baseline result, and
model weights already available locally, run from the repository root:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/run_answer_disagreement.py \
  --question-id 2813 \
  --num-generations 5 \
  --temperature 0.7 \
  --top-p 0.9 \
  --seed 2027 \
  --max-new-tokens 32 \
  --device mps \
  --output data/results/answer_disagreement_q2813_seed_2027.json
```

Offline mode prevents downloading weights. To vary the question or sampling,
change the question ID, generation count, temperature, top-p, or seed; the
runner records the actual settings and derives the default output name from the
question ID and seed when `--output` is omitted. Run automated checks with:

```bash
PYTHONPATH=src .venv/bin/pytest -q
PYTHONPATH=src .venv/bin/ruff check src scripts tests
```

## Limitations

- Empirical proportions depend on the number of generations. Five responses
  provide a coarse estimate, and changing the temperature or top-p changes the
  distribution being measured.
- Sampling seeds improve reproducibility, but exact generated text can still
  depend on library versions, device kernels, and hardware.
- The answer normalizer reduces full text to three labels and can mark nuanced
  responses ambiguous. Wording differences within one label are ignored.
- A stable answer may be wrong, and a varied answer may include the correct
  label. The score is not calibrated to POPE errors or hallucinations.
- The smoke test covers one question, one image, one model revision, and one
  sampling configuration. It does not assess generalization or distribution
  shifts.
