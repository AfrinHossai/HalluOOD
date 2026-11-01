# HalluOOD dataset and evaluation specification

**Status:** the official POPE annotation snapshot, deterministic 100-image pilot, and its matching COCO images are prepared locally. No model has been run, and no HalluOOD results are reported here. POPE facts below were checked against the paper and the authors’ public repository on 2026-10-31.

## 1. Fit with the research plan

The [HalluOOD research plan](research_plan.md) asks whether inexpensive, training-free uncertainty signals predict hallucination under distribution shift. POPE is a useful first benchmark slice because it turns object-presence hallucination into short yes/no questions with explicit labels. It does **not** by itself test general distribution shift: its standard COCO evaluation has one image source and no severity levels. Treat the three POPE negative-sampling strategies as different question-difficulty strata, not as OOD shifts. Use POPE for a clean object-existence pilot, then add separately identified controlled perturbations or other shifted sources before making claims about shift robustness.

POPE covers object existence. It does not cover all unsupported claims, such as incorrect attributes, relations, counts, or broader free-form answers. This limits the scope of conclusions from the pilot.

## 2. Verified POPE facts

### Official source and design

The primary source is the authors’ [POPE repository](https://github.com/RUCAIBox/POPE), which identifies itself as the official code and data page for [“Evaluating Object Hallucination in Large Vision-Language Models” (EMNLP 2023)](https://aclanthology.org/2023.emnlp-main.20/). The paper defines POPE as a binary object-presence task: a VLM sees an image and a question such as “Is there a chair in the image?” and should answer “Yes” or “No.” The paper balances present-object and nonexistent-object questions at 1:1 and describes three ways of selecting negative objects:

- **Random:** choose an object category that is not in the image.
- **Popular:** choose a common category that is not in the image.
- **Adversarial:** choose a category that often co-occurs with objects in the image, while still absent from it.

For its standard MSCOCO evaluation, the paper reports selecting 500 validation images with more than three annotated objects and asking six questions per image. The locally pinned random, popular, and adversarial files each validate to 3,000 rows and 500 image filenames. The official repository lists the three files under [`output/coco`](https://github.com/RUCAIBox/POPE/tree/main/output/coco); their revision and checksums are in the local pilot metadata.

### Data structure and image dependency

The repository’s README shows each question as a JSON object with these fields:

| Field | Meaning |
| --- | --- |
| `question_id` | Question identifier within a POPE file. |
| `image` | Image filename, for example a `COCO_val2014_...jpg` name. |
| `text` | The yes/no object-presence question. |
| `label` | Ground-truth string, `yes` or `no`. |

The files have a `.json` extension but the official generator writes one JSON object per line, so they should be read as JSONL-style files. The README also describes `coco_ground_truth_segmentation.json`, whose rows contain an image filename and an object list, and the generator’s outputs for object frequencies and co-occurrences. See the [official README](https://github.com/RUCAIBox/POPE#collect-ground-truth-objects) and [question-generation code](https://raw.githubusercontent.com/RUCAIBox/POPE/main/main.py).

The question rows refer to images by filename; they are not a replacement for the COCO image files. The required source is the **MS COCO 2014 validation image set**. The [official COCO download page](https://cocodataset.org/#download) lists the 2014 validation archive as `http://images.cocodataset.org/zips/val2014.zip` (about 6 GB) and documents the official Cloud Storage path `gs://images.cocodataset.org/val2014`. Each POPE filename follows `COCO_val2014_<12-digit-id>.jpg`, so the downloader requests only those exact files from `http://images.cocodataset.org/val2014/<filename>`. This direct object path was verified against a pilot image on the official host. The official page currently publishes HTTP links; the HTTPS alias failed hostname certificate verification in this environment, so the script does not disable TLS checks or route through a mirror. Because HTTP is unencrypted, the script restricts requests and redirects to the exact official host/path and decodes each JPEG, but this does not provide cryptographic transport authenticity. It does not download or extract the full archive, use an unofficial mirror, or fetch images outside the metadata’s 100-image pilot. Files are stored under `data/images/val2014/`; the whole `data/` directory is ignored by Git.

Run the local preparation command from the project root with `PYTHONPATH=src python scripts/download_images.py`. It reads the three prepared pilot annotations and `data/processed/pope_pilot_metadata.json`, requires their image sets to agree, retries failed requests, skips existing readable JPEGs, and validates each image by decoding it. Use `--retries`, `--timeout`, `--pilot-dir`, `--metadata`, and `--output-dir` to adjust local paths and network behavior.

### Licensing facts and unresolved data rights

The POPE repository has a [MIT license file](https://github.com/RUCAIBox/POPE/blob/main/LICENSE). Its text grants rights to “software and associated documentation files”; it does not clearly state a separate license for each generated question file, object annotation file, or image. Therefore, record the repository license as applying to the repository software, and treat the licensing of POPE data artifacts as **not fully verified** until the specific release is reviewed or clarified by its maintainers.

COCO’s [official terms](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm) say that COCO annotations and the website are under **CC BY 4.0**, while the COCO Consortium does not own the images’ copyrights; image use must follow the applicable Flickr terms, and users accept responsibility for use of the images. Do not describe the image archive as uniformly CC BY 4.0. The downloader retrieves images for local evaluation only; it does not determine image-level Flickr licensing or grant redistribution rights. Review image-level terms before reuse or publication, and do not redistribute image bytes unless the applicable rights allow it. The license for POPE’s derived question/label files remains an open item.

## 3. Proposed deterministic pilot subset

Use the authors’ released COCO POPE files rather than regenerating questions with the repository script. The current generator calls Python’s `random.sample` and `random.choice` without setting a seed, so regeneration is not guaranteed to reproduce the released question rows. Pinning the released files is the simpler reproducibility path.

The local annotation pilot follows this procedure:

1. Pin the POPE repository revision and record a SHA-256 checksum for each of the three source files in `data/processed/pope_pilot_metadata.json`.
2. Read the random, popular, and adversarial files. Form the eligible image set from the intersection of their image filenames; sort the filenames lexicographically and verify that the files agree.
3. Use NumPy `Generator(PCG64(2027))` to sample 100 distinct image filenames without replacement. The actual selected filenames and NumPy version are recorded in the local metadata. The prepared subset contains 100 images and 600 questions in each of the three strategies (1,800 rows total).
4. Keep **all** POPE question rows for each selected image together across all three strategies. A calibration/test split has not yet been assigned; when one is created, it must group by image and be recorded before model outputs are inspected.

This is a small pilot, not the full official POPE evaluation. With only 20 calibration image groups, fitted probability mappings and abstention thresholds will be unstable; treat those results as feasibility checks. Do not tune methods, parsers, or thresholds on the 80 held-out image groups. For the embedding-distance baseline, use a separate reference image bank that contains none of the calibration or test image IDs, and record its source and split.

### Optional paired shift extension

To connect POPE to the research plan’s shift question, an explicitly labeled extension can use the same selected images in clean and controlled-corruption conditions. One small proposed first step is clean images plus two predeclared Gaussian-blur levels (for example, Pillow blur radii 1 and 2); these are **HalluOOD experimental settings, not official POPE conditions**. Pin the image-processing library/version and parameters. Keep image IDs, questions, and official labels paired across conditions. Before treating a transformed row as evaluable, use a predeclared image-review rule to determine whether its object-presence answer is still visually answerable. Record `condition_id`, parameters, and any unanswerable/excluded status; do not quietly change the official POPE label. POPE’s random/popular/adversarial strata remain separate from blur severity.

## 4. Ground truth, model outputs, and uncertainty storage

Store immutable benchmark inputs separately from model-dependent predictions and scores. A practical table layout is `manifest`, `predictions`, and `uncertainty_scores`, keyed by a stable `row_id` derived from the pinned POPE revision, strategy, and `question_id`.

### Manifest and labels

For every question row, preserve at least:

- `row_id`, source repository/revision, source-file checksum, strategy (`random`, `popular`, or `adversarial`), `question_id`;
- `image_id` and original image filename, question text, queried object/category if available, and source image dataset/split;
- raw source label (`yes`/`no`) and normalized `gold_object_present` (1/0);
- label provenance (official POPE row and underlying annotation source), image rights/attribution notes, and any audit or answerability status;
- `condition_id` and transformation metadata for a clean or perturbed image.

Keep the official `label` unchanged. POPE’s “no” means that the object was absent from the annotations used to construct that question. Annotation omissions can make this differ from an exhaustive human judgment of what is visible; flag this as a benchmark limitation and preserve any independent audit label in a separate field with its annotator/provenance.

### Predictions

Store one row per `(row_id, model_revision, run_id, condition_id)`, including the exact model/checkpoint or API version, prompt template, decoding settings, generation seed, raw response text, normalized answer, parse status (`yes`, `no`, `abstain`, `ambiguous`, or `unparseable`), and any available answer-token probabilities. Preserve raw text so parsing can be rechecked. Do not default malformed or ambiguous answers to “yes.” Count abstentions or unparseable responses as errors for the primary all-row classification score and report parse coverage; report an official-parser compatibility score separately if needed.

The official [`evaluate.py`](https://raw.githubusercontent.com/RUCAIBox/POPE/main/evaluate.py) normalizes answers with simple “no/not” string checks and aligns predictions to labels by file order. For HalluOOD, join by stable question ID and verify there are no missing or duplicate IDs. This avoids accidental row-order mismatches and makes stricter parsing auditable.

### Uncertainty scores

Store raw, uncalibrated signal values in a separate table keyed to the same prediction or image row:

- predictive entropy and the tokens/normalization rule used to compute it;
- embedding-distance score and reference-bank identifier (this is image-level and repeats across that image’s questions);
- augmentation-consistency score and view/config identifiers;
- answer-disagreement score and generation seeds/replicate count;
- optional calibrated `p_hallucination`, calibrator version, and abstention decision/threshold.

Record score direction (`higher_means_more_risk`), missing-score reason, model calls, generated-token count, and timing metadata. Do not overwrite a raw score with a calibrated probability. Fit probability mappings and abstention thresholds on calibration image groups only, consistent with the research plan.

## 5. Safeguards against leakage

- Keep every row and strategy for an image in one split. If image IDs map to near-duplicate images or the same scene, group those together too.
- Do not use held-out image IDs in the calibration set, embedding reference bank, method selection, prompt selection, or threshold selection. Keep the exact split manifest fixed before viewing model outputs.
- Do not expose POPE labels to the VLM in the prompt. Use the same question wording and decoding settings across strategies unless a separate prompt experiment is preregistered.
- Do not join predictions by row order. Check uniqueness and exact ID coverage before computing scores; store raw outputs before parsing.
- Treat all three strategy rows for an image as clustered observations. Use image-level bootstrap resampling for intervals so six questions and repeated strategies do not masquerade as independent images.
- Record model training-data disclosures and known COCO overlap. Whether a selected VLM was trained on these COCO images cannot be verified until a model is chosen and its training provenance is reviewed; report unknown overlap as a limitation, not as proof of contamination or absence of contamination.
- Freeze question files and checksums. A changed repository `main` branch, regenerated dataset, or altered image file must receive a new dataset revision and cannot be silently substituted into the same run.

## 6. Evaluation protocol

### Standard POPE answer performance

Treat object presence (`yes`) as the positive class. On all questions in the held-out split, report the confusion matrix, accuracy, precision, recall, F1, and the fraction of answers normalized to “yes” (“Yes ratio”), as in the POPE paper/repository. Also report the false-positive rate on `gold=no` questions; this is the clearest direct measure of saying an absent object is present. Report results separately for random, popular, and adversarial strategies and avoid claiming their differences are distribution-shift effects.

### Hallucination-detection performance

For the primary POPE object-hallucination target, restrict to questions whose official label is `no`. Mark an answer as a POPE object hallucination when the model answers `yes`; a correct `no` is a non-hallucination. This evaluates whether an uncertainty score predicts unsupported positive object claims. Report the research-plan metrics—AUROC, AUPRC with prevalence, ECE, Brier score, and risk–coverage curves—on the held-out groups. Fit any score-to-probability mapping on calibration groups only, then apply it unchanged to test groups. Compute risk–coverage on the no-object questions as residual hallucination risk among answered questions.

As a separate secondary analysis, evaluate whether scores predict **any POPE answer error** on both `yes` and `no` questions. A `gold=yes`/predicted-`no` row is an object-detection miss, not a false-positive object hallucination; keeping the targets separate prevents general mistakes from being mislabeled as hallucinations. Report results by strategy and condition, with image-clustered paired confidence intervals. No benchmark or HalluOOD performance values are asserted in this plan.

## 7. Open questions before data acquisition

- Which exact POPE repository revision and checksums will be the frozen source snapshot?
- Does the POPE maintainer intend the MIT repository license to cover the generated question/label files, or do those files have a separate data license?
- For each selected COCO image, what source and image-level Flickr terms apply, and can the planned research use and any later release comply with them?
- Which frozen VLMs will be evaluated, and can their training data be checked for COCO image overlap?
- What strict answer parser and image-answerability audit rubric will be fixed before inference?
- Will the controlled-blur extension be included in the pilot, and what exact image-processing version and annotation audit will be preregistered?

Until these are resolved, the pilot subset is a reproducible sampling proposal, not a downloaded or frozen dataset artifact.

## Sources

- [POPE paper, ACL Anthology (EMNLP 2023)](https://aclanthology.org/2023.emnlp-main.20/)
- [POPE authors’ official repository and README](https://github.com/RUCAIBox/POPE)
- [Released COCO POPE question files](https://github.com/RUCAIBox/POPE/tree/main/output/coco)
- [POPE generator](https://raw.githubusercontent.com/RUCAIBox/POPE/main/main.py), [question builder](https://raw.githubusercontent.com/RUCAIBox/POPE/main/utils.py), and [evaluation script](https://raw.githubusercontent.com/RUCAIBox/POPE/main/evaluate.py)
- [POPE repository license](https://github.com/RUCAIBox/POPE/blob/main/LICENSE)
- [COCO download page](https://cocodataset.org/#download) and [COCO Terms of Use](https://cocodataset.org/#termsofuse)
