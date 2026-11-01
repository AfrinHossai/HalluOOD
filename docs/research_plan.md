# HalluOOD research plan

**Status:** proposed study; no experiments or results are reported here.

## 1. Goal and research question

Vision-language models (VLMs) can give fluent answers that are not supported by an image. They can also be evaluated on inputs that differ from the examples they usually see, such as unfamiliar scenes or degraded images. This study asks whether simple uncertainty signals can help identify those unsupported answers when the input distribution changes.

> **Central research question:** Which inexpensive, training-free uncertainty signals remain predictive of multimodal hallucination under different types and severities of distribution shift?

Here, **hallucination** means a factual claim in a model answer that is not supported by the image and question. **Distribution shift** means a meaningful change in the evaluation inputs compared with the reference inputs used to define in-distribution behavior. These are related but separate outcomes: an input can be out of distribution without causing a hallucination, and a model can hallucinate on an ordinary input.

The study will keep the selected VLMs frozen. It will compare the predictive value, reliability, and inference cost of each signal across hallucination labels and shift conditions. “Training-free” will mean that no VLM weights are updated and no task-specific model is trained. Any small post-hoc mapping from uncertainty scores to probabilities, or a threshold used for abstention, will be fit on a separate calibration split and reported explicitly as calibration rather than model training.

## 2. Hypotheses to test

These are **hypotheses, not established results**. The experiments may support, weaken, or fail to distinguish them.

- **H1 — Baseline signals:** Predictive entropy and embedding distance will each provide some information about hallucination risk, but their usefulness may vary by shift type and severity.
- **H2 — Stability signals:** Augmentation consistency and answer disagreement will add useful information in at least some conditions. They may be less useful when the model repeats the same unsupported answer or when an image transformation changes the evidence needed to answer.
- **H3 — Shift sensitivity:** A signal that works on one kind of shift may lose predictive value on another. Performance and calibration may also change as a controlled perturbation becomes stronger.
- **H4 — Selective answering:** A calibrated abstention rule will reduce hallucinations among answers the model chooses to provide, at the cost of answering fewer questions. The size of that trade-off is an empirical question.

## 3. Methods to compare

All signals will be computed from frozen model outputs or frozen embeddings. Signal direction, normalization, and answer-matching rules will be decided before the held-out test data are examined.

### Baselines

1. **Predictive entropy.** Use the model’s token probabilities while generating an answer to estimate uncertainty. The primary score will be the average token-level entropy over generated answer tokens, with a predeclared rule for excluding padding and special tokens. Higher entropy means the model was less certain among its available next-token choices. This score requires access to token probabilities; models that do not expose them will be recorded as ineligible for this baseline rather than assigned an invented score.
2. **Embedding-distance uncertainty.** Encode an evaluation image with a frozen image encoder and compare its embedding with a reference bank drawn only from the in-distribution reference split. Use a prespecified distance, such as cosine distance to the nearest reference embedding; larger distance indicates a less familiar image. The encoder, reference bank, and distance will be fixed before testing. This baseline measures input unfamiliarity, not answer correctness directly.

### Candidate methods

1. **Augmentation consistency.** Ask the same question about a small, fixed set of label-preserving transformed views of the image. Score how much the answers change across views. For short-answer tasks, use normalized answer matching; for free-form answers, use a fixed semantic-equivalence procedure specified before evaluation. Lower consistency (more change) indicates higher uncertainty. Transformations that alter the evidence or make an item unanswerable will be flagged and handled using the annotation protocol.
2. **Answer disagreement.** Generate a fixed number of answers for the unchanged image and question using a declared sampling configuration and fixed per-example seeds. Score disagreement among the answers using a prespecified normalized answer or semantic comparison method. This differs from augmentation consistency because the image stays unchanged. It costs additional generations, which will be measured.
3. **Calibrated abstention.** Convert an uncertainty score into a decision: answer when estimated hallucination risk is below a threshold, otherwise abstain. Select the probability mapping and threshold using only the calibration split, for a predeclared target risk or coverage. Do not tune them on the test split. Report both raw-signal behavior and calibrated behavior, because the calibration step uses labels even though it does not update VLM weights.

## 4. Data and evaluation conditions

Before running experiments, select and document image-question data that meet these criteria: a clear image and question, answerable ground truth, a defensible way to label unsupported claims, and permission to use the data for research. Record dataset versions, source, license or access terms, and annotation provenance. This plan does not select or download any dataset.

Use distinct partitions for (a) an in-distribution reference bank, (b) method development if needed, (c) calibration of probabilities and abstention thresholds, and (d) final held-out evaluation. Split by source image and, where possible, by scene or near-duplicate group so related examples cannot leak across partitions. Keep the final test labels and results unavailable during method and threshold selection.

Label hallucination separately from OOD status. The primary hallucination outcome will be answer-level: an answer is positive if it contains at least one material factual claim unsupported by the image or question. Preserve claim-level annotations where practical so a secondary analysis can measure the fraction of claims that are unsupported. Use trained human annotators or verified existing annotations, a written rubric, and adjudication for disagreements. Report the annotation process and agreement; do not treat an OOD label as a hallucination label.

Evaluate across multiple shift families:

- **Visual degradation:** controlled blur, brightness or contrast changes, compression, and partial occlusion. Include an unchanged level and several predeclared severity levels. Store the transformation parameters so every item can be reproduced.
- **Semantic or domain shift:** held-out sources or domains with different scenes, objects, or image styles from the reference data. Define the shift using source/domain boundaries, not by inspecting model errors.
- **Compositional shift:** held-out combinations of objects, attributes, or relations when the data support a clean split. This tests whether familiar elements in unfamiliar combinations affect signal quality.

Do not combine severity scales across families as if they meant the same thing. For synthetic visual degradation, define severity using transformation parameters and verify that the ground-truth answer remains valid. If a transformation removes necessary evidence or changes the answer, mark the item as unanswerable or relabel it under a documented protocol. For natural domain and compositional shifts, define ordered severity bins only when a defensible, predeclared measure exists; otherwise report the shift strata separately without implying a numeric severity scale.

Use at least one primary frozen VLM and, for claims intended to generalize across models, a second frozen VLM with a documented checkpoint and inference setup. Keep prompt templates, image resizing, decoding settings, and answer limits fixed across conditions. If a method needs stochastic generations, predeclare the number of generations (for example, five) and use fixed seeds such as `17`, `42`, `73`, `101`, and `2027`, consistently mapped to each item and replicate. Record when a model API does not support reproducible seeds.

## 5. Metrics and analysis

Compute metrics for hallucination prediction on the held-out data, separately for each model, shift family, and severity stratum. Also report OOD-detection results separately so that input unfamiliarity is not confused with hallucination prediction.

- **AUROC:** how well the score ranks hallucinated answers above supported answers across decision thresholds. Higher is better; it does not specify a useful operating threshold.
- **AUPRC:** precision-recall area for hallucination detection. Report hallucination prevalence alongside it, since AUPRC depends on the positive-class rate.
- **Expected Calibration Error (ECE):** how closely predicted probabilities match observed hallucination frequencies across confidence bins. Lower is better; report the binning rule because ECE depends on it.
- **Brier score:** mean squared error between predicted hallucination probabilities and binary labels. Lower is better and reflects both calibration and predictive accuracy.
- **Risk–coverage curves:** plot the hallucination rate among answered items (risk) against the fraction of items answered (coverage) as the system abstains on its most uncertain cases. Lower risk at the same coverage is preferable. Report the full curve and a prespecified summary such as area under the risk–coverage curve.

Uncertainty scores are not automatically probabilities. For ECE, Brier score, and calibrated abstention, use a simple prespecified score-to-probability mapping fit on the calibration split only; report the mapping and also show raw-score ranking results. Compare methods on the same test examples. Use paired confidence intervals, for example by bootstrapping at the image or scene-group level, so repeated or related examples are not treated as independent. Predeclare the primary metric and comparisons, then report all planned strata, including weak or negative results.

To assess “inexpensive,” report model calls or forward passes per question, number of generated tokens, and wall-clock inference time under the same hardware and batching conditions. State the additional cost of transformations and repeated generations alongside any predictive gain.

## 6. Reproducibility checklist

Before the held-out evaluation, freeze and record:

- dataset names, versions, licenses or access terms, split files, and duplicate-handling rules;
- model identifiers, checkpoint revisions, libraries, hardware, image preprocessing, prompts, decoding settings, and answer limits;
- random seeds, perturbation implementations and parameters, answer normalization or semantic matching, and handling of invalidated labels;
- the uncertainty-score definitions, calibration mapping, abstention target, metric implementations, and primary comparisons.

Save configuration files and version-control the analysis code and documentation. Keep test predictions, labels, and metric outputs together with enough metadata to reproduce each run. Report missing model scores, failed generations, exclusions, and deviations from this plan. Any change made after viewing held-out results should be labeled exploratory.

## 7. Limitations and possible failure conditions

- Hallucination can be subjective, especially for partially supported or vague answers. Answer-level labels can hide which claim failed, while claim-level annotation costs more and can still involve judgment.
- Synthetic image corruptions cannot represent every real distribution shift. A method may work on blur or brightness changes and fail on new domains, rare objects, or new relationships.
- Embedding distance can flag an unfamiliar but correctly answered image, and can miss an unsupported answer on a familiar-looking image.
- High entropy or answer disagreement can reflect ambiguity, long answers, or multiple valid phrasings rather than hallucination. Conversely, repeated generations can confidently repeat the same error.
- Augmentations may remove evidence, introduce artifacts, or leave the critical evidence unchanged. Consistency alone does not establish that an answer is true.
- Calibration and abstention thresholds can stop working when hallucination prevalence or the input mix changes. A calibration split drawn from one shift may not calibrate another.
- Log probabilities may not be available for all models, and repeated generations or transformed views add inference cost. This limits fair comparison and the practical meaning of “inexpensive.”
- Results from a small number of models, domains, or annotation sources will not establish that a signal is reliable for all VLMs or deployment settings.

The study should therefore make bounded claims: report where each signal works, where it fails, how its performance changes with shift conditions, and what additional inference or calibration data it requires.
