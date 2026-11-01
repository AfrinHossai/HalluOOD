# Embedding distance versus predictive entropy on the POPE pilot

## Question and scope

This exploratory comparison asks whether the existing image-distance scores rank
incorrect or ambiguous POPE answers differently from binary predictive entropy.
It reuses the exact 30 question/image pairs, SmolVLM predictions, and error labels
from the entropy experiment. The target is an **incorrect or ambiguous answer**;
it is not an independently adjudicated hallucination label. A large image
distance indicates that an image is less similar to the reference bank. It is
not a probability that the answer is wrong or hallucinated.

## Fixed sample and reference bank

The evaluation contains 30 distinct COCO validation images: 10 POPE Random, 10
Popular, and 10 Adversarial questions. Its sampling and greedy-generation seed
is `2027`. Before extracting features, the script checks every
question-ID/image-ID pair and verifies that the entropy and baseline records
agree on the question, category, ground truth, raw and normalized prediction,
and prediction status. Correctness is recalculated from the baseline answer;
ambiguous answers count as errors. All 30 entropy predictions matched the
baseline predictions.

The existing bank has 50 reference images represented by 960-dimensional
float32 embeddings. They were selected without replacement from the 70 local
pilot images remaining after excluding all 30 evaluation image IDs, using
NumPy PCG64 seed `2027`. The evaluation script re-creates this selection from
the local image files and checks that it matches the manifest, bank ordering,
image checksums, bank checksum, and reference-set identifier. The verified
reference-set ID is
`0380c46668a9e7cfada65abb70cdf72995d821b0ec30969af74ce1a54e5b2a82`; the
evaluation/reference ID overlap is zero.

The frozen feature encoder is
`HuggingFaceTB/SmolVLM-500M-Instruct` at revision
`a7da5b986cb59b408707209984f360a5f4ad7e47`. Its Idefics3 vision tower and
connector produce projected image-token features; the adapter averages those
features across image tokens and crops. The run used MPS, float16,
Transformers 5.22.1, and PyTorch 2.15.0, matching the bank manifest. The
evaluation image features were extracted once per image and paired to question
records by both question ID and image ID.

The scores are defined as follows:

- **Predictive entropy:** the existing normalized conditional entropy over the
  model's `Yes` and `No` answer candidate sequences. It reflects uncertainty
  between those two candidate answers, not full-vocabulary entropy.
- **Cosine distance:** the minimum `1 - cosine similarity` between the query
  embedding and an L2-normalized reference embedding. Larger values mean less
  similar to the closest reference image.
- **Nearest-neighbor distance:** the minimum Euclidean distance between the
  raw mean-pooled query vector and raw reference vectors. Its numeric scale
  depends on the encoder and representation.

For each score, AUROC and non-interpolated average precision (reported as
AUPRC) treat incorrect or ambiguous predictions as the positive class. The
sample error prevalence is 6/30 = 0.20, which is the no-skill average-precision
reference for this sample. Risk–coverage curves exclude the highest-score
examples first and report the error rate among the retained questions; tied
score values are always excluded as a group.

## Measurements

| Risk score (higher means higher risk) | Mean score, correct | Mean score, incorrect or ambiguous | AUROC | AUPRC (average precision) |
| --- | ---: | ---: | ---: | ---: |
| Predictive entropy | 0.0654 | 0.3252 | 0.8264 | 0.5913 |
| Cosine distance | 0.0151 | 0.0189 | 0.6528 | 0.4857 |
| Nearest-neighbor distance | 16.5812 | 18.4053 | 0.6806 | 0.4972 |

Entropy ranked errors better than either image-distance score on this sample.
For both embedding distances, the mean score was higher in the error group, but
the ranking metrics were weaker than entropy; this difference between group
means and ranking quality is possible with a small sample and overlapping
scores.

The discrete risk–coverage points below use complete score-tie groups. At 0%
coverage no risk is defined.

| Coverage retained | High-score questions excluded | Entropy error rate | Cosine-distance error rate | Nearest-neighbor error rate |
| ---: | ---: | ---: | ---: | ---: |
| 100% (30) | 0 | 20.0% | 20.0% | 20.0% |
| 80% (24) | 6 | 8.3% | 12.5% | 12.5% |
| 50% (15) | 15 | 6.7% | 13.3% | 13.3% |
| 33.3% (10) | 20 | 10.0% | 10.0% | 10.0% |

The full per-score curves are saved in the comparison JSON and plotted together
in the SVG. The CSV contains the comparison table, and the JSONL retains each
question ID, image ID, category, ground truth, prediction, correctness label,
entropy, both embedding distances, nearest reference IDs, and reference-set
ID. The extracted evaluation vectors are stored separately in an NPZ file.
These artifacts are in the Git-ignored `data/results/embedding_distance/`
directory.

The output files are `pope_embedding_entropy_30_seed_2027_scores.jsonl`,
`pope_embedding_entropy_30_seed_2027_evaluation_embeddings.npz`,
`pope_embedding_entropy_30_seed_2027_comparison.json`,
`pope_embedding_entropy_30_seed_2027_comparison.csv`, and
`pope_embedding_entropy_30_seed_2027_risk_coverage.svg`.

## Does distance add information beyond entropy?

As a limited complementarity check, the evaluator formed two label-independent
rank fusions: the average of each example's normalized average rank for entropy
and cosine distance, and the same fusion with nearest-neighbor distance. No
weights were selected using labels. It then compared each fusion with entropy
alone using 2,000 paired, image-level bootstrap replicates stratified by the
error label (seed `2027`). This is an exploratory diagnostic, not a fitted
predictor or a deployment procedure.

| Score | AUROC | AUROC difference versus entropy (95% bootstrap interval) | AUPRC | AUPRC difference versus entropy (95% bootstrap interval) |
| --- | ---: | ---: | ---: | ---: |
| Entropy alone | 0.8264 | — | 0.5913 | — |
| Entropy + cosine rank fusion | 0.8056 | -0.0208 (-0.1944, 0.1181) | 0.7101 | +0.1189 (-0.3109, 0.3806) |
| Entropy + nearest-neighbor rank fusion | 0.8194 | -0.0069 (-0.1494, 0.1181) | 0.6982 | +0.1070 (-0.3333, 0.3723) |

The fusion point estimates raise AUPRC but lower AUROC, and both bootstrap
intervals include zero for both metrics. Entropy and image-distance scores
also had near-zero Spearman rank correlations on these 30 items (cosine:
`-0.0238`; nearest neighbor: `-0.0438`). These observations do **not** show
that embedding distance genuinely improves error detection beyond entropy.
They leave open the possibility of complementary signal, but the pilot is too
small to distinguish that from sampling variation.

## Reproduction

From the repository root, with the prepared COCO pilot images, the existing
50-image embedding bank and manifest, and the pinned model available in the
local cache, run:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/evaluate_embedding_ood.py \
  --device mps \
  --bootstrap-replicates 2000 \
  --bootstrap-seed 2027
```

The offline flag prevents a model download. The script fails if its local
inputs are missing or if the manifest, reference-bank contents, deterministic
reference selection, predictions, or sample identities disagree. It only
extracts image features for the fixed 30 evaluation images and writes results
under `data/results/embedding_distance/`.

Run the automated checks with:

```bash
PYTHONPATH=src .venv/bin/pytest -q
PYTHONPATH=src .venv/bin/ruff check src scripts tests
```

## Limitations

- There are only 30 paired questions and six error labels, one of which is an
  ambiguous answer. AUROC, AUPRC, risk–coverage points, and bootstrap intervals
  are consequently unstable. The bootstrap intervals compare the rank-fusion
  diagnostic with entropy; they are not confidence intervals for every table
  entry.
- POPE answer correctness is only a proxy for hallucination here. No new
  claim-level or human hallucination annotation was done.
- Image distance measures visual familiarity relative to this small bank. A
  familiar image can still prompt an unsupported answer, while an unfamiliar
  image can be answered correctly. The scores are not calibrated probabilities
  of error, hallucination, or OOD status.
- The 50 reference images and 30 evaluation images are disjoint by COCO image
  ID, and their selection is reproducible. No scene-level or perceptual
  near-duplicate audit was performed.
- This uses one model revision, one encoder, one dataset split, and one small
  reference bank. It does not test distribution shifts or support claims about
  other models or settings.
- Raw Euclidean distances depend on the model, pooling, and feature scale. They
  should not be compared numerically across encoders without a separate
  prespecified normalization.
