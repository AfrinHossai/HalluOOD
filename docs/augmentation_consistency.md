# Augmentation-consistency uncertainty

## Method

The method measures whether a frozen vision-language model changes its
normalized yes/no answer when the same question is asked about mild views of
the same image. It reuses the POPE baseline answer parser and the existing
greedy-generation settings. It does not change or overwrite the baseline
prediction file.

For an original answer \(\hat y_0\) and \(K\) transformed-view answers, the
uncertainty score is:

\[
U_{aug} = \frac{1}{K}\sum_{k=1}^{K}
\mathbf{1}[\hat y_k \ne \hat y_0]
\]

Each parsed answer is normalized to yes or no. An ambiguous or unparseable
generated answer is stored as null and treated as a third answer state.
Consequently, a view that changes yes to no, or changes a parsed answer to
ambiguous, counts as disagreement. The score ranges from 0 (every view agrees
with the original) to 1 (every transformed view differs). It measures
augmentation sensitivity; it is **not** a model probability, binary entropy,
or direct estimate of hallucination risk. A model that repeats the same wrong
answer can receive a score of zero.

The transformations are deterministic Pillow operations:

- brightness factors 0.90 and 1.10;
- contrast factors 0.90 and 1.10;
- one horizontal and one vertical translation, each up to 1% of the matching
  image dimension. The translation signs are sampled using NumPy PCG64 with a
  recorded seed. The exposed strip is filled with the image's mean RGB color.

All views keep the original image dimensions. A translation is skipped when
the selected fraction would be less than one pixel on that axis. Images are
transformed in memory and are not written to disk. The run record stores each
transform's name and parameters, including the actual pixel offsets and fill
color. The original image is also inferred once, using the same question and
deterministic generation settings; the recorded baseline answer is retained
and compared with that fresh original answer.

## One-pair smoke test

The real smoke test used the existing POPE baseline row for question 2813:

| Field | Value |
| --- | --- |
| Category | Random |
| Image | COCO_val2014_000000396068.jpg |
| Question | Is there a remote in the image? |
| Ground truth | yes |
| Existing baseline response | “Yes, there is a remote in the image.” |
| Fresh original-image response | “Yes, there is a remote in the image.” |
| Original response matches baseline | yes |
| Transformed views | 6 |
| Disagreements | 0 of 6 |
| Augmentation-consistency uncertainty | **0.0** |

All six transformed images received the same normalized answer, yes. The
views used brightness 0.90 and 1.10, contrast 0.90 and 1.10, horizontal shift
+6 pixels on the 628-pixel width, and vertical shift -6 pixels on the 640-pixel
height. The exposed shift strips used mean RGB (47, 40, 37). Each view retained
the original 628 × 640 dimensions.

The run used HuggingFaceTB/SmolVLM-500M-Instruct,
revision a7da5b986cb59b408707209984f360a5f4ad7e47, Transformers 5.22.1,
PyTorch 2.15.0, MPS, float16, and greedy generation with seed 2027,
max_new_tokens=32, and one beam. It made seven calls: one on the original
image and one for each transformed view. The fresh original response matched
the saved baseline response; the baseline file itself was not changed.

The row-level smoke output is saved at
data/results/augmentation_consistency_q2813_seed_2027.json, under the
Git-ignored data/ directory. It includes raw responses, normalized answers,
inference timings, model metadata, transform configuration, and score.
Transformed image files are not persisted.

## Reproduction

With the existing POPE annotations, COCO pilot image, and pinned model already
available locally, run this from the repository root:

    HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python \
      scripts/evaluate_augmentation_consistency.py \
      --question-id 2813 \
      --seed 2027 \
      --device mps \
      --output data/results/augmentation_consistency_q2813_seed_2027.json

The script reads the selected question and original prediction from the
existing baseline output, verifies that the question, image, and ground truth
agree with the POPE annotation loader, and checks the loaded model and
generation metadata against the baseline row. Offline mode prevents downloading
model weights. The script evaluates one question only.

Run tests and lint from the repository root:

    PYTHONPATH=src .venv/bin/pytest -q
    PYTHONPATH=src .venv/bin/ruff check src scripts tests

## Limitations

- A low disagreement score does not imply correctness. A model can repeat the
  same unsupported answer on every view; a high score can reflect visual
  sensitivity even when one answer is correct.
- The answer parser reduces the generated response to yes, no, or an
  ambiguous state. It discards differences in explanation wording and confidence.
- Brightness, contrast, and small translations may leave the relevant evidence
  unchanged. Conversely, even a mild transform can affect a tiny object near an
  image boundary. This fixed transform set has not been audited for answer
  preservation on every POPE question.
- The smoke test is one question, one image, and one model. It is a
  reproducibility check, not evidence of predictive performance or
  generalization.
- Six extra generations per question increase inference time and cost. No
  probability-variation score or additional uncertainty method was evaluated.
