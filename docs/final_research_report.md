# HalluOOD: Training-Free OOD and Hallucination Calibration for Vision-Language Models

## Final research report

*Exploratory PhD-portfolio research project; not a peer-reviewed or published paper.*

## Abstract

This project studies whether inexpensive, training-free signals can help flag unreliable answers from a frozen vision-language model (VLM), particularly when images are degraded. The experiments use POPE, a yes/no object-presence benchmark, and one checkpoint: `HuggingFaceTB/SmolVLM-500M-Instruct`. The primary outcome is an incorrect, ambiguous, or failed POPE answer. This is an answer-correctness proxy; the project did not independently annotate free-form hallucination claims.

The evidence has two scales that must be kept separate. A 30-question, 30-image pilot compared five signals: binary predictive entropy, cosine embedding distance, nearest-neighbor embedding distance, augmentation consistency, and sampled answer disagreement. Predictive entropy had the highest pilot point estimates for error ranking (AUROC 0.826, AUPRC 0.591), but its image-grouped 95% intervals were wide. Only deterministic prediction and predictive entropy were expanded to all 1,800 locally prepared questions on 100 images. On this expanded local set, accuracy was 0.842; entropy ranked answer errors with AUROC 0.798 (95% image-grouped CI 0.763–0.830) and AUPRC 0.410 (0.337–0.497), against an error prevalence of 0.158. A separate 54-question, three-image perturbation subset showed lower accuracy for several stronger corruptions, but entropy did not increase monotonically with severity and its error-ranking estimates were imprecise. A disjoint-image abstention analysis showed lower observed held-out risk at lower coverage, without a formal risk guarantee.

The results support a bounded conclusion: binary predictive entropy was associated with answer errors in this one-model POPE evaluation, including the larger clean sample, but can be low for wrong or ambiguous answers. The current evidence does not establish general hallucination detection, robust performance across natural shifts, or a dependable ranking of all five signals.

## Introduction and research motivation

VLMs can produce confident-sounding statements that are not supported by an image. A practical system should ideally identify cases in which its answers may be unreliable, even when the model itself remains unchanged. HalluOOD therefore explores signals available from a frozen model, image representations, or repeated inference, and asks whether those signals rank erroneous answers usefully.

The project is motivated by two related but distinct problems. **Hallucination** concerns whether a claim is unsupported by the visual evidence. **Out-of-distribution (OOD) detection** concerns whether an input differs from a reference distribution. An unfamiliar image may still be answered correctly, and a familiar image may elicit an unsupported answer. These outcomes should not be conflated.

POPE provides a controlled starting point by asking whether a named object is present. Its Random, Popular, and Adversarial categories describe different ways of selecting absent objects; they are question-construction strata, not distribution shifts. POPE is narrower than general multimodal hallucination: it does not directly evaluate object attributes, relations, counts, or open-ended descriptions. The POPE benchmark was introduced for systematic object-hallucination evaluation in large VLMs ([Li et al., 2023](https://aclanthology.org/2023.emnlp-main.20/)).

## Research questions and hypotheses

The central question from the original research plan is:

> Which inexpensive, training-free uncertainty signals remain predictive of multimodal hallucination under different types and severities of distribution shift?

The hypotheses in the research plan were expectations to test, not established results:

- **H1 — Baseline signals:** predictive entropy and embedding distance may rank errors on clean data, but their usefulness may vary by shift.
- **H2 — Stability signals:** augmentation consistency and sampled answer disagreement may add useful information, but could fail when the model repeats a wrong answer or transformations preserve the same error.
- **H3 — Shift sensitivity:** signal quality and score magnitude may vary by shift type and severity.
- **H4 — Selective answering:** an entropy threshold chosen on separate calibration images may trade coverage for lower observed risk.

The recorded results provide partial, exploratory evidence for H1 and H4 on this POPE sample. The pilot comparison does not establish that embedding distance, augmentation consistency, or answer disagreement improves upon entropy. The small shift subset gives mixed evidence for H3. None of these experiments tests H1–H4 across a diverse set of models or natural domains.

## Related work and references

POPE frames object hallucination as binary questions about object presence, with negative objects sampled using Random, Popular, and Adversarial strategies ([POPE paper](https://aclanthology.org/2023.emnlp-main.20/)). The present work adopts that task format while explicitly using answer correctness as its measured error label. The underlying images come from MS COCO, a dataset of common objects in everyday scenes ([Lin et al., 2014](https://doi.org/10.1007/978-3-319-10602-1_48)).

The entropy and distance baselines are related to broader work on model confidence and OOD detection. Neural confidence can be miscalibrated, motivating separate evaluation of ranking, probability calibration, and selective decisions ([Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html)). Feature-space distance methods are a longstanding OOD approach; for example, Lee et al. use class-conditional Mahalanobis distance ([Lee et al., 2018](https://proceedings.neurips.cc/paper_files/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html)). HalluOOD's pilot uses two simpler image-to-reference distances and does not reproduce that method. Threshold-based selective prediction is also an established framework ([Geifman & El-Yaniv, 2017](https://papers.nips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html)). The selected frozen checkpoint belongs to the compact SmolVLM family described by Marafioti et al. ([2025](https://arxiv.org/abs/2504.05299)); this report evaluates the recorded checkpoint rather than making a general claim about that family.

## Dataset and experimental setup

### POPE data

The local expanded evaluation used the prepared Random, Popular, and Adversarial question files over 100 COCO 2014 validation images: 600 questions per category, six per image and category, for 1,800 question rows in total. The annotations are pinned to POPE repository revision `08d957b917e5a378a2f99d35b6293c536a66298b`; the local metadata records SHA-256 checksums for each source file. The 30-question pilot contains 10 questions per category on 30 distinct images and is nested inside the expanded set. Thus “expanded” means all rows in this local 100-image pilot, not the entire standard POPE release. The standard benchmark described in the dataset plan uses 500 COCO validation images; the present work covers only a subset. COCO's [official download page](https://cocodataset.org/#download) identifies the source image collection used by the local preparation.

The POPE source repository is the authors' official code-and-data repository ([RUCAIBox/POPE](https://github.com/RUCAIBox/POPE)). Its repository MIT license is verified for software, but the dataset plan correctly records the separate licensing status of the released question and label files as unresolved. COCO's official terms license its annotations under CC BY 4.0, while noting that the consortium does not own the photographs and that image use remains subject to applicable Flickr terms ([COCO Terms of Use](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm)). The images and derived data are kept local and excluded from Git; this report does not assert redistribution rights.

### Model and inference

All model experiments used the frozen `HuggingFaceTB/SmolVLM-500M-Instruct` checkpoint at revision `a7da5b986cb59b408707209984f360a5f4ad7e47`, from the [official Hugging Face model card](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct). Recorded environment details are Transformers 5.22.1, PyTorch 2.15.0, Apple MPS, and float16. Deterministic evaluations used greedy decoding with one beam, seed 2027, and a 32-token generation limit. The expanded clean run completed 1,800 predictions and entropy calculations without recorded inference failures in 5,648 seconds.

### Outcome and metric definitions

The positive error label is an incorrect, ambiguous, or failed yes/no answer. Ambiguous and failed outputs count as incorrect for accuracy and as positive cases for error detection; the expanded run recorded no inference failures. This target measures POPE answer correctness, not an independently verified hallucination claim. In particular, the main AUROC and AUPRC analyses combine false-positive object claims with other mistakes such as missed present objects. A separate hallucination-only analysis restricted to ground-truth `no` questions was not reported.

AUROC evaluates how well a score ranks the error class above the correct class. AUPRC is non-interpolated average precision; its no-skill reference is the error prevalence in that sample. Confidence intervals are percentile bootstrap intervals resampling image IDs as clusters, carrying all questions for each sampled image together. Expanded clean intervals use 1,000 replicates and seed 2027; final-figure pilot method intervals use 2,000 replicates and seed 2027, with 1,998 valid replicates where class-degenerate resamples were omitted. These intervals describe sampling variation over the recorded image groups and are not tests of between-method significance.

ECE and Brier score were proposed in the research plan but are **not reported**: the saved experiments did not fit and validate a prespecified score-to-error probability mapping suitable for those measures. Raw uncertainty scores are not probabilities. AUROC, AUPRC, and risk–coverage results should not be interpreted as calibration measurements.

## Methodology

All five scores were computed without updating model weights. “Training-free” here means no task-specific model fine-tuning; embedding extraction, multiple forward passes, calibration-set inference, and post-hoc threshold selection still have computational costs.

1. **Predictive entropy.** The implemented POPE score is normalized entropy over a two-answer candidate set. The model computes sequence log-likelihoods for the full candidate strings `Yes` and `No`, including each answer's actual tokenizer tokens; the two sequence likelihoods are normalized over this binary set, and entropy is divided by `log(2)`. A score near one indicates a more balanced candidate distribution; a score near zero indicates concentration on one candidate. This is **not** full-vocabulary next-token entropy and does not estimate the probability of hallucination. It is also a deviation from the research plan's initial full-vocabulary token-entropy proposal. The binary score depends on the candidate strings and sequence-likelihood construction, including potential effects of differing token counts.
2. **Cosine embedding distance.** A frozen image encoder produces a 960-dimensional mean-pooled projected image representation. The score is the minimum cosine distance, `1 - cosine similarity`, to a normalized reference embedding. Larger values indicate lower similarity to the closest reference image, not a more likely incorrect answer.
3. **Nearest-neighbor embedding distance.** The score is the minimum Euclidean distance from the raw pooled image vector to the raw vectors in the same reference bank. Its magnitude depends on the encoder and feature scale, so it should not be compared directly across encoders.
4. **Augmentation consistency.** The score is the fraction of six deterministic mild image views for which the normalized answer differs from the saved original answer. Views use brightness and contrast changes of ±10% and horizontal and vertical translations of up to 1% of an image dimension. Ambiguous output is a third answer state and counts as disagreement. This measures sensitivity to these particular views, not truth or confidence.
5. **Sampled answer disagreement.** Five sampled generations per question are normalized to `Yes`, `No`, or `ambiguous`. The score is categorical entropy over their empirical proportions, normalized by `log(3)`. The pilot used temperature 0.7, top-p 0.9, and recorded per-response seeds derived from seed 2027. Unanimity yields zero disagreement, even if all five answers are wrong.

The embedding reference bank contains 50 COCO validation images. For the pilot embedding comparison, these IDs were selected from images left after excluding all 30 pilot evaluation images, using PCG64 seed 2027; the recorded overlap was zero. The expanded clean entropy analysis does not use this bank. Because 50 reference IDs overlap with the full 100-image set, the bank cannot support a leakage-free embedding comparison over all 1,800 rows. For expanded abstention, those reference IDs were excluded before splitting the remaining 50 image groups into calibration and held-out sets.

## Experimental results

### Original 30-question pilot: five-score comparison

All five scores were joined to the same 30 image-question pairs and the same six baseline error labels (20% prevalence). The table reports point estimates and image-grouped 95% intervals. Pilot intervals are wide; overlapping marginal intervals are not a paired test of method differences.

| Score | AUROC (95% CI) | AUPRC / average precision (95% CI) |
| --- | ---: | ---: |
| Predictive entropy | 0.826 (0.552–1.000) | 0.591 (0.287–1.000) |
| Cosine distance | 0.653 (0.319–0.938) | 0.486 (0.131–0.840) |
| Nearest-neighbor distance | 0.681 (0.358–0.942) | 0.497 (0.139–0.844) |
| Augmentation consistency | 0.479 (0.432–0.500) | 0.200 (0.067–0.367) |
| Sampled answer disagreement | 0.642 (0.460–0.868) | 0.328 (0.100–0.725) |

Entropy had the highest pilot point estimates, but the uncertainty intervals make a reliable ranking impossible. For augmentation consistency, 29 of 30 examples had a score of zero, and all six baseline errors were unchanged across views. For answer disagreement, four of the six error examples received the same wrong normalized answer in all five samples. These are direct failure cases for the stability signals on this pilot.

The embedding study tested whether distance added information through an exploratory, label-independent rank fusion. Entropy-plus-cosine fusion had AUROC 0.806 and AUPRC 0.710; entropy-plus-nearest-neighbor fusion had AUROC 0.819 and AUPRC 0.698, compared with entropy alone at 0.826 and 0.591. The paired bootstrap intervals for the fusion-minus-entropy differences included zero for both metrics. These results do not establish that distance adds error-detection information beyond entropy.

### Expanded clean evaluation: 1,800 questions on 100 images

The expanded local evaluation produced 1,515 correct answers, accuracy 0.842, and 55 ambiguous responses (3.1%). The error class contained 285 questions (15.8% prevalence), including the ambiguous answers. Higher normalized entropy ranked errors above correct answers with AUROC 0.798 (95% CI 0.763–0.830) and AUPRC 0.410 (0.337–0.497). This AUPRC exceeds the corresponding prevalence baseline of 0.158. The mean normalized entropy was 0.163 (95% CI 0.147–0.180).

| POPE category | Accuracy (95% CI) | Ambiguous rate | Mean entropy (95% CI) | AUROC (95% CI) | AUPRC (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 0.878 (0.853–0.903) | 0.053 | 0.125 (0.108–0.143) | 0.769 (0.711–0.818) | 0.344 (0.251–0.453) |
| Popular | 0.842 (0.812–0.868) | 0.010 | 0.166 (0.146–0.186) | 0.819 (0.781–0.857) | 0.419 (0.337–0.534) |
| Adversarial | 0.805 (0.775–0.835) | 0.028 | 0.198 (0.177–0.221) | 0.787 (0.748–0.824) | 0.457 (0.369–0.548) |
| Overall | 0.842 (0.819–0.866) | 0.031 | 0.163 (0.147–0.180) | 0.798 (0.763–0.830) | 0.410 (0.337–0.497) |

Rates are fractions of all questions. Category-specific AUPRC baselines are their error prevalences (0.122, 0.158, and 0.195 for Random, Popular, and Adversarial, respectively); overall prevalence is 0.158. The results show useful ranking association in this sample, not a calibrated probability or a guarantee on new inputs.

The pilot is a nested subset, not an independent replication. Accuracy changed from 0.800 in the 30-question pilot to 0.842 in the expanded result; ambiguous-response rate changed from 0.033 to 0.031; mean entropy from 0.117 to 0.163; AUROC from 0.826 to 0.798; and AUPRC from 0.591 to 0.410. These are descriptive differences and should not be treated as independent-sample comparisons.

## Distribution-shift robustness

The expanded shift study reused 54 image-question pairs (18 questions per image across the three POPE categories) from three held-out images and applied all 12 combinations of Gaussian blur, brightness reduction, Gaussian noise, and contrast reduction at three severity levels. This produced 648 shifted inferences, alongside the 54 paired clean predictions. The selected IDs were `000000191964`, `000000199940`, and `000000304819`; the images were chosen from the held-out evaluation IDs with seed 2028. The transforms were applied in memory, preserving original files and labels.

The perturbations used blur radii 0.5, 1.5, and 3 px; brightness and contrast factors 0.8, 0.6, and 0.4; and Gaussian-noise standard deviations 5, 15, and 30. Against the paired-clean accuracy of 0.815 (95% CI 0.722–0.907), strongest blur had accuracy 0.685 (0.556–0.778), strongest brightness reduction 0.704 (0.685–0.722), strongest noise 0.704 (0.556–0.796), and strongest contrast reduction 0.759 (0.722–0.796). Across shifted conditions, mean entropy ranged from 0.157 to 0.238 and did not consistently rise with severity. For example, entropy declined from 0.206 at brightness severity 1 to 0.187 at severity 3 while accuracy also declined from 0.815 to 0.704; contrast entropy declined from 0.212 to 0.170 as severity increased.

Entropy's AUROC point estimates ranged from 0.532 to 0.827 across the shift conditions, while AUPRC ranged from 0.216 to 0.585. In the clean three-image subset, AUROC was 0.602 (95% CI 0.437–0.908) and AUPRC 0.264 (0.113–0.831). At blur severity 3, AUROC was 0.827 (0.512–0.964) and AUPRC 0.585 (0.502–0.847); at brightness severity 3, AUROC was 0.572 (0.354–0.757) and AUPRC 0.376 (0.273–0.614). Each condition's bootstrap sampled only three image clusters, so these intervals are exceptionally coarse; many AUROC intervals include chance. AUPRC must be read against each condition's error prevalence, which also changes under perturbation.

The evidence does not show a monotonic relationship between image degradation and entropy, nor does it establish reliable error detection under shifts. Accuracy fell for several severe corruptions, yet entropy sometimes fell or changed non-monotonically. A corrupted image may also hide the object needed to answer the original question. Since the expanded shift labels were inherited unchanged and there was no per-example independent answerability audit, some additional errors may reflect lost visual evidence rather than hallucination in an otherwise answerable case. An earlier 30-image pilot shift run is reported separately in `docs/distribution_shift_evaluation.md`; it is not pooled with this three-image expanded-subset study.

## Calibrated abstention and selective prediction

The decision rule answers when normalized entropy is at or below a threshold and abstains above it. For expanded evaluation, the 50-image embedding reference set was first excluded. The remaining 50 image groups were split using PCG64 seed 2027 into 10 calibration images (180 questions) and 40 held-out evaluation images (720 questions). The split manifest records zero image overlap among reference, calibration, and held-out sets. Thresholds were selected using calibration entropy only, without calibration error labels or held-out labels: the observed cutoff nearest to each requested calibration coverage was chosen, tied scores were kept together, and a lower-coverage cutoff was preferred for an equal-distance tie. The procedure matched target **coverage**; it did not target a risk level.

Before abstention, the held-out group had 123 errors among 720 questions (risk 0.171). The observed outcomes were:

| Target calibration coverage | Entropy threshold | Achieved calibration coverage | Held-out coverage (95% CI) | Selective risk (95% CI) | Answered / errors answered | Abstention rate (95% CI) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50% | 0.036220 | 0.506 | 0.538 (0.489–0.582) | 0.067 (0.035–0.105) | 387 / 26 | 0.463 (0.418–0.511) |
| 70% | 0.125281 | 0.700 | 0.728 (0.688–0.761) | 0.076 (0.050–0.106) | 524 / 40 | 0.272 (0.239–0.312) |
| 80% | 0.248957 | 0.806 | 0.825 (0.788–0.858) | 0.098 (0.069–0.132) | 594 / 58 | 0.175 (0.142–0.212) |
| 90% | 0.645006 | 0.894 | 0.894 (0.865–0.921) | 0.117 (0.087–0.151) | 644 / 75 | 0.106 (0.079–0.135) |

Intervals use 1,000 image-group bootstrap replicates with the selected threshold fixed. In this partition, observed selective risk was lower than the unselective held-out risk at each target, with fewer questions answered. This is a single-split observation, not a formal calibration or selective-risk guarantee. The earlier pilot used 20 calibration images and 30 separate evaluation images; it also showed lower observed risk at the more restrictive thresholds, but its estimates were especially small-sample and should not be combined with the expanded split.

## Failure analysis

The saved analysis identifies three examples that illustrate signal limitations:

- **Low-entropy wrong answer:** Popular question 1218 asks whether a chair is present in image `000000112664`. The label is `no`, but the model answered “Yes, there is a chair in the image.” Its normalized binary entropy was 0.0075, the lowest among parsed errors in the expanded run.
- **Ambiguous response with low entropy:** Random question 510 asks whether skis are present in image `000000148766`; the label is `no`. The model said it could not determine whether skis were present. The parser marked this response ambiguous, yet the binary entropy was 0.0043. The binary candidate score does not capture all uncertainty expressed in free-form text.
- **Repeated wrong answer:** Adversarial question 2448 asks whether a bottle is present in image `000000238029`. The label is `no`; the baseline answer and all five sampled answers were `yes`. Sampled answer disagreement was zero. Four of six pilot errors had five unanimous wrong yes/no responses; a separate correct answer also had high disagreement.

These examples show why score ranking can be informative in aggregate yet fail on individual examples. Low entropy or repeated outputs should not be presented as evidence of correctness.

## Discussion

The principal positive result is that binary predictive entropy ranked the defined POPE answer errors above correct answers in both the original pilot and the larger local clean evaluation. The expanded result is more stable than the 30-item estimate because it includes more questions and image groups. It nonetheless covers only 100 images from one source and one frozen model. The expanded result answers a narrower question than the original research question: it supports an association on clean POPE data, not a general conclusion about multimodal hallucination under arbitrary distribution shift.

The other four scores were evaluated only on the 30-question pilot. Their intervals are too broad for confident ranking. The embedding distances measure image similarity to a small reference bank; image familiarity is not answer correctness. Rank fusion did not show statistically distinguishable improvement from entropy in the paired bootstrap. Augmentation consistency was nearly constant and failed to flag the six errors. Sample disagreement was also often zero for errors because generations repeated the same wrong answer. These pilot failures are informative design constraints, not proof that the methods cannot work under other sampling or augmentation settings.

The controlled perturbations link input degradation to answer quality only tentatively. Severe blur and brightness reduction reduced accuracy on the three-image subset, but the uncertainty score did not reliably rise as accuracy fell. Distribution shift detection and hallucination detection remain separate evaluation targets: the embedding distances were evaluated as error-ranking scores on the pilot, not as a dedicated in-distribution versus out-of-distribution classifier, and were not tested on the expanded shift subset.

The abstention results show a useful empirical risk–coverage trade-off on one held-out split. Because threshold selection matches score coverage rather than controlling an error rate, the observed selective risks can change on new images or under shift. Practical use would also require an explicit policy for ambiguous responses and the cost of abstention.

## Limitations and threats to validity

1. **Limited image diversity and clustered questions.** The expanded clean result has 1,800 rows but only 100 distinct images, with six related questions per image in each POPE category. Image-grouped bootstrap intervals account for within-image correlation but cannot create independent image diversity. The pilot method comparison has only 30 images and six error cases.
2. **Very small shift subset.** The expanded controlled-shift experiment has only three images. Its 1,000-resample intervals remain coarse because the number of unique clusters is three. The perturbations are synthetic corruptions, not natural domain shifts, unfamiliar object distributions, or composition changes.
3. **No answerability adjudication under shift.** Original POPE labels were preserved for transformed images; there was no independent judgment for each transformed question about whether its object remained visible. Some shifted errors may be caused by unanswerability.
4. **Narrow benchmark target.** The error label combines incorrect, ambiguous, and failed responses. No independent human or claim-level annotation determines whether a textual claim was unsupported. The main error metrics are not the narrower false-positive hallucination rate on absent-object questions.
5. **Single model and checkpoint.** All model results use one SmolVLM checkpoint, runtime stack, device, prompt format, and decoding setup. The checkpoint's full training-data overlap with COCO cannot be established from these experiments.
6. **Only one score expanded.** Entropy is the only uncertainty score evaluated across all 1,800 rows. Embedding distance, augmentation consistency, and sampled disagreement remain pilot-only. The embedding reference bank also overlaps half of the 100-image expanded set, preventing a fair full-set embedding comparison without constructing a new disjoint design.
7. **Entropy construction.** The implemented score is binary candidate-sequence entropy rather than the initial full-vocabulary token-entropy proposal. It is sensitive to exact answer strings, tokenizer behavior, and sequence likelihoods. It can be low for ambiguous or incorrect free-form outputs.
8. **Metric and calibration limits.** AUPRC varies with error prevalence. Bootstrap intervals are not between-method significance tests. ECE and Brier score were not computed because no held-out calibrated probability model was evaluated. The abstention thresholds match coverage; they do not guarantee risk.
9. **Reproducibility boundaries.** Seeds, model revision, settings, and image-group splits are recorded, but exact generations may still depend on hardware and library kernels. Five sampled answers provide only a coarse disagreement estimate.
10. **Data licensing.** The POPE repository's software license does not settle the license for its generated question files. COCO image rights remain subject to source-level terms. This report does not resolve those questions or grant redistribution permission.

## Conclusion

HalluOOD implements and evaluates five training-free uncertainty signals for POPE object-presence answers, but the evidence is uneven in scale. Predictive entropy showed an error-ranking association on the 30-question pilot and on the expanded local evaluation of 1,800 questions from 100 images. Its expanded AUROC was 0.798 and AUPRC 0.410, with image-grouped intervals above the respective 0.5 AUROC and 0.158 prevalence baselines. Yet individual low-entropy errors occurred, and the three-image shift experiment showed non-monotonic uncertainty as accuracy declined under several corruptions. The other four signals have only small pilot evidence and no demonstrated advantage over entropy.

Separate-image threshold selection reduced observed held-out risk while reducing coverage in one expanded split, but it did not establish a guaranteed risk level. Overall, the project is a reproducible exploratory evaluation pipeline and a set of bounded observations. Stronger claims require more independent images, multiple models and datasets, a direct hallucination label, broader shifts, and calibration metrics measured on a rigorously separated evaluation design.

## Reproducibility instructions

All raw predictions, local image files, embeddings, split manifests, and figures are in the Git-ignored `data/` directory and are not part of this report. The numeric results above were cross-checked against the saved run summaries and the machine-readable final-figure audit file. The final figures referenced here are:

- `data/results/final_figures/uncertainty_method_comparison.svg`
- `data/results/final_figures/clean_shift_accuracy.svg`
- `data/results/final_figures/clean_shift_entropy.svg`
- `data/results/final_figures/risk_coverage_entropy_abstention.svg`

The following commands assume the prepared POPE annotations, local COCO images, pinned model cache, and Python environment are already available. `HF_HUB_OFFLINE=1` prevents model-weight downloads. The full clean and shift commands perform inference and may take substantial time; they are listed for reproducibility only and were not run while writing this report.

```bash
# Full local 1,800-question deterministic clean run
HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/run_full_pope.py \
  --device mps --seed 2027 --max-new-tokens 32

# Analyze full clean results and run the disjoint-image abstention analysis
PYTHONPATH=src .venv/bin/python scripts/analyze_robustness.py \
  --seed 2027 --bootstrap-replicates 1000 \
  --target-coverages 0.5,0.7,0.8,0.9

# Reproduce the 54-pair, three-image controlled-shift evaluation
HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/evaluate_expanded_shifts.py \
  --device mps --seed 2027 --subset-seed 2028 --image-count 3

# Recompute the final figures and audit summary from saved outputs only
PYTHONPATH=src .venv/bin/python scripts/create_final_figures.py \
  --results-dir data/results \
  --output-dir data/results/final_figures \
  --bootstrap-replicates 2000 --seed 2027
```

For pilot-method reproduction, the recorded reports provide the exact commands and settings for `scripts/evaluate_embedding_ood.py`, `scripts/evaluate_augmentation_pilot.py`, and `scripts/evaluate_answer_disagreement.py` in `docs/embedding_evaluation.md`, `docs/augmentation_evaluation.md`, and `docs/answer_disagreement_evaluation.md`. Those commands re-run feature extraction or inference for 30 questions; the final-figure script itself does not run a model. Run the test suite and linter with:

```bash
PYTHONPATH=src .venv/bin/pytest -q
PYTHONPATH=src .venv/bin/ruff check src scripts tests
```

To run the original 30-question deterministic pilot and its five-score analyses from scratch, the main commands are:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/run_baseline.py \
  --sample-size 30 --seed 2027 --device mps --max-new-tokens 32 \
  --record-uncertainty \
  --output data/results/pope_entropy_30_seed_2027.jsonl
PYTHONPATH=src .venv/bin/python scripts/evaluate_entropy.py

HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/evaluate_embedding_ood.py \
  --device mps --bootstrap-replicates 2000 --bootstrap-seed 2027
HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/evaluate_augmentation_pilot.py \
  --device mps --seed 2027
HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/evaluate_answer_disagreement.py \
  --temperature 0.7 --top-p 0.9 --seed 2027 --max-new-tokens 32 --device mps
```

## References

1. Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W. X., & Wen, J.-R. (2023). Evaluating Object Hallucination in Large Vision-Language Models. *Proceedings of EMNLP 2023*, 292–305. [ACL Anthology](https://aclanthology.org/2023.emnlp-main.20/). DOI: 10.18653/v1/2023.emnlp-main.20.
2. RUCAIBox. (n.d.). *POPE: Official code and data repository*. [Pinned annotation directory at revision `08d957b917e5a378a2f99d35b6293c536a66298b`](https://github.com/RUCAIBox/POPE/tree/08d957b917e5a378a2f99d35b6293c536a66298b/output/coco). The repository's software license is MIT; the license status of generated data files is unresolved in this project.
3. Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., & Zitnick, C. L. (2014). Microsoft COCO: Common Objects in Context. *Computer Vision – ECCV 2014*, 740–755. [Springer](https://doi.org/10.1007/978-3-319-10602-1_48). See also [COCO Terms of Use](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm).
4. Marafioti, A., Zohar, O., Farré, M., Noyan, M., Bakouch, E., Cuenca, P., Zakka, C., Ben Allal, L., Lozhkov, A., Tazi, N., Srivastav, V., Lochner, J., Larcher, H., Morlon, M., Tunstall, L., von Werra, L., & Wolf, T. (2025). *SmolVLM: Redefining small and efficient multimodal models*. arXiv:2504.05299. [arXiv](https://arxiv.org/abs/2504.05299). The experiment used the specific checkpoint revision recorded above.
5. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *Proceedings of the 34th International Conference on Machine Learning*, PMLR 70, 1321–1330. [PMLR](https://proceedings.mlr.press/v70/guo17a.html).
6. Lee, K., Lee, K., Lee, H., & Shin, J. (2018). A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks. *Advances in Neural Information Processing Systems 31*. [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html).
7. Geifman, Y., & El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks. *Advances in Neural Information Processing Systems 30*. [NeurIPS](https://papers.nips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html).
8. Hugging Face. (n.d.). *HuggingFaceTB/SmolVLM-500M-Instruct model card*. [Hugging Face Hub](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct).
