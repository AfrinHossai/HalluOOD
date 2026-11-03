# HalluOOD: Training-Free OOD and Hallucination Calibration for Vision-Language Models

**HalluOOD is an exploratory research project for evaluating lightweight uncertainty signals on frozen vision-language models (VLMs).** It asks whether those signals can help rank unreliable image-question answers, including under controlled image degradation. This repository contains a reproducible evaluation pipeline and exploratory results; it is not a peer-reviewed or published study.

## Abstract

HalluOOD evaluates five training-free signals for yes/no object-presence answers: binary predictive entropy, cosine embedding distance, nearest-neighbor embedding distance, augmentation consistency, and sampled answer disagreement. Experiments use the POPE benchmark and one frozen checkpoint, `HuggingFaceTB/SmolVLM-500M-Instruct`. All five signals were compared on a 30-question pilot; predictive entropy alone was evaluated at expanded scale on 1,800 questions across 100 COCO validation images. A separate controlled-shift evaluation used 54 question-image pairs from three images. Results suggest that entropy ranks answer errors on this local clean sample, but low-entropy errors occur and uncertainty does not increase consistently with image degradation. Findings are exploratory and do not establish general hallucination detection or reliable performance under natural distribution shift.

**Start here:** [Final research report](docs/final_research_report.md) · [Failure analysis and figure guide](docs/failure_analysis.md) · [Evaluation reports](#evaluation-documentation)

## Research motivation and questions

VLMs can produce fluent answers unsupported by their images, and can behave differently when inputs are unfamiliar or degraded. Hallucination detection and out-of-distribution (OOD) detection are related but distinct: an unfamiliar image may be answered correctly, while an ordinary image may elicit an incorrect answer.

The central research question is:

> Which inexpensive, training-free uncertainty signals remain predictive of multimodal hallucination under different types and severities of distribution shift?

The project keeps model weights frozen. “Training-free” means no task-specific model fine-tuning; feature extraction, repeated inference, and calibration-set evaluation still incur computation. The experiments use POPE answer errors as a practical proxy, not independently annotated hallucination claims.

## Technical contributions

- A POPE loader and deterministic, image-grouped data preparation workflow.
- A version-pinned SmolVLM adapter that preserves raw responses and exposes candidate-answer sequence likelihoods for binary entropy.
- Implementations of five uncertainty signals, controlled image perturbations, and entropy-based abstention with image-disjoint calibration and evaluation sets.
- Evaluation utilities for accuracy, AUROC, average precision (AUPRC), risk–coverage, image-grouped bootstrap intervals, and reproducible plots.
- Per-example outputs and split metadata that retain question/image identifiers, predictions, score definitions, and run settings for local reproduction.

These are engineering and empirical contributions for a bounded evaluation. The uncertainty scores and selective-prediction approach build on established methods; this project does not claim a new general-purpose detector.

## Dataset and model

### Dataset

The experiments use the authors’ POPE object-presence questions for the **Random**, **Popular**, and **Adversarial** negative-sampling strategies. These categories are question-construction strata, not distribution-shift conditions. The expanded local evaluation contains 1,800 questions (600 per category) on 100 MS COCO 2014 validation images. The original pilot contains 30 questions on 30 images and is nested in that set. This is a subset of the standard POPE release, not a full benchmark evaluation.

POPE covers object existence, not every kind of multimodal hallucination. The main error label combines incorrect, ambiguous, and failed answers; it is not a separately human-verified hallucination label. See the [dataset and evaluation plan](docs/dataset_plan.md) for source and annotation details.

### Model

All recorded model experiments use `HuggingFaceTB/SmolVLM-500M-Instruct` at revision `a7da5b986cb59b408707209984f360a5f4ad7e47`. The expanded run used Transformers 5.22.1, PyTorch 2.15.0, Apple MPS, and float16. Deterministic runs use greedy decoding, one beam, seed 2027, and a 32-token limit. Model weights are not stored in this repository; inference requires the checkpoint in the local Hugging Face cache or network access for its first download.

## Evaluated uncertainty methods

| Method | Signal | Evaluation scale |
| --- | --- | --- |
| Predictive entropy | Entropy of model likelihoods normalized over candidate answer sequences `Yes` and `No`. This is binary candidate-sequence entropy, not full-vocabulary token entropy or a probability of hallucination. | 30-question pilot and expanded clean evaluation; also measured under controlled shifts |
| Cosine embedding distance | Minimum cosine distance from a frozen image embedding to a reference bank. Indicates image similarity to that bank, not answer correctness. | 30-question pilot |
| Nearest-neighbor embedding distance | Minimum Euclidean distance to a reference embedding. Its scale depends on the encoder representation. | 30-question pilot |
| Augmentation consistency | Fraction of six mild brightness, contrast, and small-translation views whose normalized answer differs from the original. | 30-question pilot |
| Sampled answer disagreement | Normalized entropy over five sampled `Yes`, `No`, and ambiguous responses. Repeated wrong answers can have zero disagreement. | 30-question pilot |

Only predictive entropy was compared at the expanded 1,800-question scale. The other four methods have pilot-only results.

## Main experimental findings

The target for AUROC and AUPRC is an incorrect, ambiguous, or failed baseline answer. AUPRC is non-interpolated average precision, and its reference level depends on error prevalence. Confidence intervals below use image-grouped bootstrap resampling. They describe variation over the evaluated image groups and are not between-method significance tests.

### Five-method comparison: 30-question pilot

The 30-question pilot contains six error cases (20% prevalence). The following point estimates and 95% image-grouped intervals are pilot-only:

| Risk score | AUROC (95% CI) | AUPRC (95% CI) |
| --- | ---: | ---: |
| Predictive entropy | 0.826 (0.552–1.000) | 0.591 (0.287–1.000) |
| Cosine embedding distance | 0.653 (0.319–0.938) | 0.486 (0.131–0.840) |
| Nearest-neighbor distance | 0.681 (0.358–0.942) | 0.497 (0.139–0.844) |
| Augmentation consistency | 0.479 (0.432–0.500) | 0.200 (0.067–0.367) |
| Sampled answer disagreement | 0.642 (0.460–0.868) | 0.328 (0.100–0.725) |

Entropy had the highest pilot point estimates, but the intervals are wide. All six pilot errors remained unchanged across the six augmentation views, and four of six errors received unanimous wrong labels in five sampled generations. These examples show that response stability does not imply correctness.

### Expanded clean evaluation: 1,800 questions

| Scope | Questions / images | Accuracy | Error prevalence | Entropy AUROC (95% CI) | Entropy AUPRC (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original pilot | 30 / 30 | 0.800 | 0.200 | 0.826 (0.552–1.000) | 0.591 (0.287–1.000) |
| Expanded local evaluation | 1,800 / 100 | 0.842 (0.819–0.866) | 0.158 | **0.798 (0.763–0.830)** | **0.410 (0.337–0.497)** |

The pilot row reports point accuracy; the expanded accuracy interval is image-grouped. The expanded entropy values are for predictive entropy on this specific 1,800-question sample, not universal performance claims. The pilot is nested in the expanded set, so differences between these rows are descriptive rather than independent replication. Full results are in the [robustness evaluation](docs/robustness_evaluation.md).

### Controlled shifts and abstention

- **Controlled image shifts:** 54 paired questions from three images were evaluated under 12 conditions (four perturbations at three severities). Accuracy fell from 0.815 on the paired clean subset to 0.685 under the strongest blur; entropy did not increase monotonically, and error-ranking estimates are uncertain with only three image groups. This does not show robust OOD or hallucination detection.
- **Selective prediction:** a threshold selected from 180 calibration questions on 10 images was evaluated on 720 questions from 40 disjoint images. At the 50% target calibration coverage, observed coverage was 53.8% and selective risk was 6.7% (unfiltered held-out risk: 17.1%). Coverage was reduced to reject more uncertain responses. This is a result from one image-disjoint split, not a formal risk guarantee.
- **Calibration metrics:** ECE and Brier score were proposed but not measured. The saved raw uncertainty scores are not calibrated error probabilities.

The [final report](docs/final_research_report.md) gives confidence intervals, category-level results, split details, and the complete limitations. Generated figures and raw experimental files are kept locally under `data/results/` and are intentionally excluded from Git; the [failure-analysis report](docs/failure_analysis.md) lists their filenames and reproduction command.

## Installation and reproducibility

The project requires Python 3.11 and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv python install 3.11
uv sync --extra inference
```

This installs the project, development tools, and optional PyTorch/Transformers inference dependencies. The first model run may download the pinned checkpoint to the local Hugging Face cache. Prepare the POPE annotations and the 100 referenced COCO validation images locally using the instructions in the [dataset plan](docs/dataset_plan.md). Dataset files, images, model weights, and experiment outputs are not checked into Git.

Run tests and lint checks with:

```bash
PYTHONPATH=src .venv/bin/pytest -q
.venv/bin/ruff check src scripts tests
```

With prepared local data and model weights available, reproduce the 30-question entropy pilot with:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/run_baseline.py \
  --sample-size 30 \
  --seed 2027 \
  --device mps \
  --max-new-tokens 32 \
  --record-uncertainty \
  --output data/results/pope_entropy_30_seed_2027.jsonl
PYTHONPATH=src .venv/bin/python scripts/evaluate_entropy.py
```

Offline mode requires that the model checkpoint is already cached. To reproduce the expanded clean run and analyze its saved outputs, use:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/run_full_pope.py \
  --device mps --seed 2027 --max-new-tokens 32

PYTHONPATH=src .venv/bin/python scripts/analyze_robustness.py \
  --seed 2027 --bootstrap-replicates 1000 \
  --target-coverages 0.5,0.7,0.8,0.9
```

The full inference run took about 94 minutes on the recorded Apple MPS setup. For the 54-question shift evaluation and local figure regeneration, see the exact commands in the [robustness report](docs/robustness_evaluation.md) and [failure analysis](docs/failure_analysis.md). Reproducing experiments reruns inference; the figure-generation script reads saved results only.

## Repository structure

```text
src/halluood/   Dataset, model, uncertainty, shift, and evaluation modules
scripts/        Data, inference, evaluation, and figure-generation commands
tests/          Automated tests, including mocked model paths
docs/           Research plans, dataset specification, evaluation reports
configs/        Reserved configuration directory
data/           Local datasets, images, model-derived outputs (Git-ignored)
```

## Evaluation documentation

- [Research plan and hypotheses](docs/research_plan.md)
- [Dataset and evaluation plan](docs/dataset_plan.md)
- [POPE baseline](docs/baseline_evaluation.md)
- [Predictive entropy](docs/entropy_evaluation.md)
- [Embedding-distance pilot](docs/embedding_evaluation.md)
- [Augmentation-consistency pilot](docs/augmentation_evaluation.md)
- [Sampled answer-disagreement pilot](docs/answer_disagreement_evaluation.md)
- [Entropy-based abstention](docs/abstention.md)
- [Earlier 30-question distribution-shift pilot](docs/distribution_shift_evaluation.md)
- [Expanded robustness evaluation](docs/robustness_evaluation.md)
- [Failure analysis and figure guide](docs/failure_analysis.md)
- [Final research report](docs/final_research_report.md)

## Limitations and future work

This is a single-model, single-dataset study with 100 clean evaluation images and only three images in the expanded shift subset. Questions sharing an image are correlated, and image-grouped intervals cannot compensate for limited image diversity. The five-method comparison has only six pilot errors. The main target is POPE answer correctness rather than independent claim-level hallucination labels; corruption may also make a question unanswerable while its original label is retained. Only entropy was evaluated at the expanded scale. The embedding reference bank overlaps half of the 100-image set, which prevents using it for a leakage-free expanded embedding comparison. POPE question-file licensing is unresolved, and image rights remain subject to the source-specific COCO/Flickr terms.

Future work should add other VLMs and datasets, expand the disjoint image sets, compare all methods at a matched larger scale, audit answerability under perturbation, measure false-positive object claims separately, and evaluate natural shifts. ECE and Brier score require a prespecified probability mapping trained on separate calibration images; threshold-based abstention alone does not establish calibrated probabilities or guaranteed risk.

## References and licensing

- Li, Y. et al. (2023). [Evaluating Object Hallucination in Large Vision-Language Models](https://aclanthology.org/2023.emnlp-main.20/). EMNLP 2023.
- Lin, T.-Y. et al. (2014). [Microsoft COCO: Common Objects in Context](https://doi.org/10.1007/978-3-319-10602-1_48). ECCV 2014.
- Marafioti, A. et al. (2025). [SmolVLM: Redefining small and efficient multimodal models](https://arxiv.org/abs/2504.05299). arXiv:2504.05299.
- [Official POPE repository](https://github.com/RUCAIBox/POPE) and [COCO Terms of Use](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm).
- The project source declares MIT in `pyproject.toml` and the root `LICENSE`. The SmolVLM checkpoint has its own terms, listed as Apache 2.0 on the [model card](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct). The POPE repository's MIT license does not clearly establish a separate license for its generated question/label files. COCO annotations are CC BY 4.0, but the consortium does not own the images; follow applicable Flickr terms. Review source-level conditions before redistributing any data or images.
