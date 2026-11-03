# Final HalluOOD figures and failure analysis

## Scope and data provenance

The figures use only experiment files already recorded under the local,
Git-ignored `data/results/` tree. The generator makes no model calls and does
not alter predictions, labels, images, or earlier reports. It verifies that the
five pilot score files contain the same 30 question/image pairs and the same
ground truth, normalized prediction, parse status, and error labels. It also
recomputes ranking metrics from per-question scores and checks them against the
saved comparison report.

The result called an **error** is any incorrect, ambiguous, or failed POPE
yes/no response. It is an answer-correctness proxy, not an independently
annotated hallucination claim. Predictive entropy is normalized entropy over
the candidate answer sequences `Yes` and `No`; it is neither full-vocabulary
entropy nor a probability that an answer is hallucinated.

The five displayed pilot scores are predictive entropy, cosine distance,
nearest-neighbor distance, augmentation consistency, and sampled answer
disagreement. Only predictive entropy has the expanded 1,800-question result.
Pilot estimates therefore must not be read as if all five methods were tested
at expanded scale. The AUPRC values are non-interpolated average precision;
their no-skill reference depends on error prevalence.

## Figure set

The following scalable SVG figures and a machine-readable source-data summary
are generated in `data/results/final_figures/`:

| File | Contents |
| --- | --- |
| `uncertainty_method_comparison.svg` | AUROC and AUPRC for the five pilot scores plus expanded predictive entropy; image-grouped 95% intervals and prevalence references. |
| `clean_shift_accuracy.svg` | Clean-versus-shift accuracy by severity, including the paired clean subset and full clean reference. |
| `clean_shift_entropy.svg` | Mean normalized entropy by severity with the same paired and full-clean references. |
| `risk_coverage_entropy_abstention.svg` | Pilot and expanded entropy risk–coverage curves with image-grouped pointwise intervals, plus pilot (`P`) and expanded held-out (`E`) calibrated-abstention points. |
| `final_figure_data_seed_2027.json` | Recomputed metrics, bootstrap intervals, split checks, and selected failure examples used by the plots. |

The shifted-condition plots show 54 matched question rows from **3 distinct
images** at each of the 12 perturbation conditions. Their 95% intervals are
read from the existing 1,000-replicate, image-group bootstrap. The dashed
clean reference uses the full evaluation of **1,800 questions from 100
images**. The paired clean points in the shift plots use the same 54 questions
and 3 images as the perturbations. Displaying both clean references makes the
sample-size difference explicit.

## Method comparison

The 30-question pilot has six error labels and 30 distinct images. The table
shows the observed metric followed by its 95% image-group percentile interval.
For this pilot, 2,000 bootstrap draws resampled image IDs with replacement;
all questions for a selected image would travel together. There is one pilot
question per image. Degenerate resamples lacking a class required by a metric
are omitted; the pilot ranking intervals each use 1,998 valid replicates. The
saved expanded-clean intervals use 1,000 image-group replicates over 100
images.

| Score and cohort | Questions / images | AUROC (95% interval) | AUPRC (95% interval) | Error prevalence |
| --- | ---: | ---: | ---: | ---: |
| Predictive entropy, pilot | 30 / 30 | 0.8264 (0.5517–1.0000) | 0.5913 (0.2871–1.0000) | 0.200 |
| Cosine distance, pilot | 30 / 30 | 0.6528 (0.3191–0.9376) | 0.4857 (0.1308–0.8400) | 0.200 |
| Nearest-neighbor distance, pilot | 30 / 30 | 0.6806 (0.3580–0.9424) | 0.4972 (0.1387–0.8438) | 0.200 |
| Augmentation consistency, pilot | 30 / 30 | 0.4792 (0.4318–0.5000) | 0.2000 (0.0667–0.3667) | 0.200 |
| Sampled answer disagreement, pilot | 30 / 30 | 0.6424 (0.4602–0.8675) | 0.3278 (0.1000–0.7250) | 0.200 |
| Predictive entropy, expanded clean | 1,800 / 100 | 0.7983 (0.7631–0.8298) | 0.4103 (0.3367–0.4966) | 0.158 |

Entropy had the highest pilot AUROC and AUPRC point estimates, while
augmentation consistency ranked below chance on AUROC and at the pilot error
prevalence on AUPRC. The image distances and answer disagreement were between
those extremes. These small-sample intervals are wide; overlap between them
does not establish that one method is superior. No paired significance test or
claim of statistical significance is made. The expanded result supports a
more precise estimate for entropy on this POPE sample, not an expanded-scale
comparison with the other methods.

The pilot AUPRC no-skill reference is 0.20 and the expanded reference is
0.158, matching the error prevalence in each cohort. AUPRC should not be
compared across the two cohorts without accounting for this prevalence change.

## Clean and controlled-shift measurements

All shifted conditions contain 54 questions from the same 3 images. The table
shows accuracy and mean normalized binary entropy, with the saved 95%
image-group intervals. The clean subset is paired with every condition. The
full 100-image clean evaluation is a separate reference, not a paired control
for the three-image subset. In that full clean run, 285 answers were marked as
errors, including 55 ambiguous outputs; no inference failures were recorded.

| Condition | Accuracy (95% interval) | Mean normalized entropy (95% interval) |
| --- | ---: | ---: |
| Full clean, 1,800 questions / 100 images | 0.8417 (0.8189–0.8656) | 0.1629 (0.1470–0.1797) |
| Paired clean, 54 questions / 3 images | 0.8148 (0.7222–0.9074) | 0.1801 (0.1294–0.2286) |
| Gaussian blur, severity 1 | 0.8148 (0.7222–0.9074) | 0.1574 (0.1029–0.1954) |
| Gaussian blur, severity 2 | 0.7778 (0.5556–0.9259) | 0.2335 (0.2004–0.2949) |
| Gaussian blur, severity 3 | 0.6852 (0.5556–0.7778) | 0.1905 (0.1597–0.2306) |
| Brightness reduction, severity 1 | 0.8148 (0.7222–0.9074) | 0.2057 (0.1100–0.2597) |
| Brightness reduction, severity 2 | 0.7593 (0.7222–0.7963) | 0.1995 (0.1297–0.2443) |
| Brightness reduction, severity 3 | 0.7037 (0.6852–0.7222) | 0.1874 (0.1521–0.2342) |
| Gaussian noise, severity 1 | 0.7593 (0.5556–0.9074) | 0.1918 (0.1517–0.2210) |
| Gaussian noise, severity 2 | 0.7037 (0.6852–0.7222) | 0.1629 (0.1026–0.2127) |
| Gaussian noise, severity 3 | 0.7037 (0.5556–0.7963) | 0.2382 (0.1336–0.3159) |
| Contrast reduction, severity 1 | 0.8148 (0.7222–0.9074) | 0.2121 (0.0880–0.2976) |
| Contrast reduction, severity 2 | 0.7593 (0.7222–0.7963) | 0.1911 (0.0953–0.2434) |
| Contrast reduction, severity 3 | 0.7593 (0.7222–0.7963) | 0.1700 (0.0955–0.2091) |

In this three-image subset, accuracy declined with stronger blur and brightness
reduction, while noise and contrast showed smaller, non-monotonic changes.
Entropy also failed to rise monotonically with severity: its mean fell from
severity 1 to 3 for brightness and contrast, and changed non-monotonically for
blur and noise. The saved shift summary reports entropy AUROC values from
0.5318 to 0.8267 across the 12 conditions, but the image-group intervals are
wide and resample only three distinct images. These numbers are descriptive
of this subset, not reliable estimates of broad robustness.

All original POPE ground-truth labels were retained after transformation.
No independent item-level adjudication established that every question
remained visually answerable at every severity. Earlier qualitative review of
pilot contact sheets is not a substitute for such validation on the expanded
shift subset.

## Risk–coverage and calibrated abstention

The figure shows empirical entropy risk–coverage curves for the original
30-question pilot and the 1,800-question expanded evaluation. Pilot abstention
points (`P`) use the saved evaluation of 30 images and 20 separate calibration
images. Expanded held-out abstention points (`E`) use 720 questions from 40
images, with threshold selection on 180 questions from 10 calibration images.
For the expanded split, calibration, evaluation, and the 50-image embedding
reference set have zero image overlap. For the earlier pilot, calibration and
evaluation images also have zero overlap. The thresholds were selected using
calibration entropy only, without held-out correctness labels. Error bars show
95% image-group bootstrap intervals with the selected thresholds or empirical
score cutoffs held fixed.

The pilot abstention intervals below were calculated from its saved per-question
decisions by resampling the 30 evaluation images 2,000 times. The expanded
intervals were saved by the original evaluation and use 1,000 replicates over
40 held-out image groups.

| Target calibration coverage | Pilot evaluation coverage (95% interval) | Pilot selective risk (95% interval) |
| ---: | ---: | ---: |
| 50% | 0.7000 (0.5333–0.8667) | 0.0476 (0.0000–0.1539) |
| 70% | 0.7333 (0.5667–0.8667) | 0.0455 (0.0000–0.1500) |
| 80% | 0.9000 (0.7992–1.0000) | 0.1481 (0.0357–0.2963) |
| 90% | 0.9000 (0.7992–1.0000) | 0.1481 (0.0357–0.2963) |

The pilot used 20 calibration questions from 20 images; the requested coverage
levels do not always produce distinct evaluation coverages because tied
entropy values remain together.

On the held-out set, 123 of 720 answers were errors before abstention (17.1%).
The recorded threshold results were:

| Target calibration coverage | Held-out coverage (95% interval) | Held-out selective risk (95% interval) |
| ---: | ---: | ---: |
| 50% | 0.5375 (0.4889–0.5819) | 0.0672 (0.0351–0.1051) |
| 70% | 0.7278 (0.6875–0.7611) | 0.0763 (0.0505–0.1057) |
| 80% | 0.8250 (0.7875–0.8583) | 0.0976 (0.0685–0.1318) |
| 90% | 0.8944 (0.8652–0.9209) | 0.1165 (0.0868–0.1506) |

Observed held-out risk was lower at the more restrictive thresholds in this
sample, with coverage reduced accordingly. This does not guarantee risk control
on another sample or shift; the calibration set contains only 10 images, and
coverage matching is not a formal selective-risk guarantee. For the pilot and
expanded entropy curves, the figure adds 95% image-group bootstrap whiskers at
the observed cutoffs nearest 20%, 40%, 60%, 80%, and 100% nominal coverage. Each
bootstrap applies the original score cutoff unchanged, so the whiskers reflect
both coverage and selective-risk variation under resampled image groups. They
are pointwise intervals, not a simultaneous confidence band for every curve
position.

## Representative failure cases

The examples below are selected mechanically from saved predictions: the
lowest-entropy parsed error, the lowest-entropy ambiguous output, and the
lowest-entropy pilot error with five unanimous wrong sampled answers.

### Low-entropy incorrect answer

In the expanded clean run, Popular question **1218** on image
`000000112664` asks, “Is there a chair in the image?” The POPE label is `no`,
but the model returned “Yes, there is a chair in the image.” Its normalized
binary entropy is **0.0075**, the lowest among parsed incorrect predictions in
the 1,800-question result. The low score is not a calibrated confidence
probability; it shows that the model strongly favored one of the two candidate
answer sequences while favoring the wrong one.

### Ambiguous text with low binary entropy

Random question **510** on image `000000148766` asks, “Is there a skis in the
image?” The ground-truth label is `no`. The raw answer says, “There is no
mention of skis in the provided facts, so we cannot determine if there are any
skis present in the image.” It is marked ambiguous by the response parser, so
it counts as an error even though its normalized binary entropy is only
**0.0043**. This illustrates a mismatch between the two-candidate entropy
signal and uncertainty expressed in free-form language.

### Five sampled answers repeat the same error

In the 30-question answer-disagreement pilot, Adversarial question **2448** on
image `000000238029` asks whether a bottle is present. Its POPE label is `no`;
the deterministic answer is `yes`, and all five sampled generations are also
normalized to `yes` (one says “glass bottle”). The disagreement score is
**0.0000**. There were **four** incorrect pilot examples with five unanimous
sampled yes/no answers, and every one received zero disagreement. Repetition
can therefore preserve an error instead of revealing it.

## Limitations

- The five-method comparison has only 30 questions and six errors. Its
  image-group bootstrap intervals are wide, especially for AUPRC; they do not
  support confident method ranking or claims of statistical significance.
- The pilot methods share the same 30 images and correctness labels. Their
  estimates are paired, and overlapping marginal intervals are not a test of
  between-method differences.
- The expanded clean set has 1,800 questions but only 100 images, with
  multiple questions from each image. Its confidence intervals resample images
  as groups; row-wise independence would understate uncertainty.
- The shift subset has only three images. Even image-grouped bootstrap
  intervals are coarse at that cluster count. Its perturbations are synthetic
  image degradations, not a substitute for natural domain or compositional
  shifts.
- Blur, noise, brightness, or contrast changes may hide evidence or make an
  originally answerable question visually ambiguous. The inherited ground
  truth remains fixed, and there is no new independent answerability audit for
  each transformed example.
- Only predictive entropy has an expanded-scale score comparison. The two
  embedding distances, augmentation consistency, and answer disagreement are
  pilot-only. All results use one model revision and the POPE answer-error
  proxy, not independent human hallucination annotations.
- Consistent answers do not guarantee correctness, and low binary entropy does
  not imply a low hallucination probability. The score measures a restricted
  answer distribution, not the truth of the response.

## Reproduction

From the repository root, with the local result files present:

```bash
PYTHONPATH=src .venv/bin/python scripts/create_final_figures.py \
  --results-dir data/results \
  --output-dir data/results/final_figures \
  --bootstrap-replicates 2000 \
  --seed 2027
```

The script validates sample identities, predictions, labels, saved metric
tables, the 1,800-row clean result, all 648 shifted rows, and image-level split
separation before writing the four SVG figures and JSON audit summary. It reads
only local results and does not perform model inference.
