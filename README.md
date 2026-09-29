# Copernicus-Nav 

Completed Image Segmentation for the UGV...

## Terrain Segmentation Model

`segmodel_v1_2026-09-19` — a class-weighted checkpoint (epoch 8) selected for deployment over an unweighted alternative. The unweighted model scored slightly higher on aggregate mIoU, but never detected `person` or `log` at all (IoU 0.0 for both), meaning those obstacles were invisible to any downstream costmap. The weighted model trades a small amount of accuracy on common classes (grass, concrete) for detecting rare, safety-relevant ones.

| Metric | Unweighted | Weighted (deployed) |
|---|---|---|
| Overall mIoU | 0.3464 | 0.3310 |
| Hazardous mIoU | 0.3193 | 0.3079 |
| person IoU | 0.0000 | 0.4102 |
| log IoU | 0.0000 | 0.4108 |
| puddle recall | 0.4216 | 0.8481 |

Model export: ONNX, opset 17, SHA256 `23910cc4953e093c7d1c1b449f288b00a07934fdb07f4cf599da192de6fb7e0f`. Released as GitHub tag [`v1-segmodel-2026-09-19`](https://github.com/AryaShekhar13/Copernicus-Nav/releases/tag/v1-segmodel-2026-09-19).

## Example Predictions

Each panel shows: input image, ground truth, model prediction, entropy (uncertainty) heatmap.

### Best case

![Best case](docs/images/best_case.png)

A clear frame with 91.7% pixel accuracy. Ground truth and prediction line up closely for large, well-defined regions (sky, grass, path). Entropy is low almost everywhere, correctly concentrated at object boundaries and around the person silhouette, exactly where uncertainty should be higher.

### Worst case

![Worst case](docs/images/worst_case.png)

70.0% pixel accuracy. The clearest failure here is the puddle: in the input image it's one continuous water-filled rut, and ground truth marks it as a single clean region. The prediction instead breaks it into a fragmented patchwork of puddle and mud, confusing the two classes at the boundary. This is a visible instance of a pattern the numbers below confirm: mud is the hazard class the model struggles with most.

### Most hazard pixels in sample

![Hazard-heavy case](docs/images/hazard_heavy_case.png)

A frame with a large concentration of hazard-class ground truth (water, puddle, mud, rubble combined). Useful for checking whether high-entropy regions in the uncertainty map line up with where the prediction disagrees with ground truth on hazard classes specifically, rather than just overall accuracy.

## Hazard Detection and Uncertainty Analysis

The model outputs a per-pixel entropy score alongside its segmentation, intended to flag pixels it is unsure about even when the class prediction itself is wrong. This was evaluated on hazardous classes (water, puddle, mud, rubble) using held-out test data (test.lst, 1672 frames).

Question: when the model misclassifies a hazard pixel, does its entropy score flag it as suspicious?

| Hazard | Missed pixels | Catch rate at 0.2 nats | at 0.4 nats | at 0.6 nats |
|---|---|---|---|---|
| water | 93,803 | 0.997 | 0.960 | 0.874 |
| puddle | 1,656,529 | 0.889 | 0.712 | 0.529 |
| mud | 815,235 | 0.914 | 0.777 | 0.617 |
| rubble | 170,786 | 0.957 | 0.870 | 0.786 |
| pooled | | 0.904 | 0.749 | 0.583 |

![Hazard catch rate](docs/images/hazard_catch_rate.png)

- Water has the worst raw detection recall (0.090) of any hazard, but the best uncertainty catch rate. When the model misses water, it is reliably uncertain about it rather than confidently wrong, the good-case outcome for a safety-oriented costmap.
- Mud is the weak point: the highest missed-pixel count and the lowest catch rate at every threshold. At a 0.6-nat threshold, 38% of missed mud pixels carry no uncertainty warning at all.
- A 0.2-nat threshold catches 90.4% of all missed hazard pixels pooled, at the cost of flagging a larger share of all pixels overall as uncertain (about 40%), a downstream planning/costmap cost, not a perception-accuracy one.

![Entropy vs error rate](docs/images/entropy_vs_error.png)

Caveat: these are pixel-level catch rates, not spatial. Whether the missed fraction (for example 13% of water at 0.6 nats) scatters randomly or clusters into one solid unflagged patch changes whether a threshold is actually safe in practice. This is the next check planned before treating any threshold as a final costmap parameter.

### Calibration

ECE (15-bin, non-void pixels): 0.0139. The pooled number is flattered by the fact that 71% of pixels fall in the top confidence bin (0.98 confidence, 0.979 accuracy), where calibration is near-perfect. Every mid-confidence bin (0.47 to 0.93) is overconfident by 3 to 6 percentage points. Mild, and not yet acted on; if corrected later, it should be via temperature scaling fit on a validation split, not test.

![Calibration diagram](docs/images/calibration_diagram.png)

<!-- comparison-2026-09-29:start -->
## Pretrained-model comparison (2026-09-29)

**Decision: keep TerrainSegModel (baseline). SegFormer is not adopted.**

Setup: same training recipe for every model, RELLIS-3D, gates evaluated on the **test** split. Validation has no water pixels, so it cannot judge water.

### Results (test split)
| model | test mIoU | person rec / prec | puddle rec / prec | mud rec / prec | water rec / prec | CPU ms (info only) |
|---|---|---|---|---|---|---|
| baseline_mud_s0 | 0.347 | 0.98 / 0.56 | 0.75 / 0.80 | 0.89 / 0.18 | 0.20 / 0.04 | 164 |
| baseline_s0 | 0.382 | 0.97 / 0.67 | 0.92 / 0.76 | 0.52 / 0.31 | 0.74 / 0.17 | 159 |
| baseline_s1 | 0.381 | 0.99 / 0.55 | 0.88 / 0.82 | 0.64 / 0.41 | 0.27 / 0.29 | 177 |
| proxy_seed123_ep8 | 0.398 | 0.99 / 0.54 | 0.79 / 0.85 | 0.63 / 0.39 | 0.43 / 0.45 | 154 |
| segformer_b0_s0 | 0.398 | 0.98 / 0.60 | 0.92 / 0.73 | 0.81 / 0.38 | 0.01 / 0.11 | 192 |
| segformer_b1_s0 | 0.395 | 0.98 / 0.56 | 0.88 / 0.77 | 0.85 / 0.31 | 0.23 / 0.74 | 333 |
| segformer_b1_s1 | 0.404 | 0.99 / 0.65 | 0.91 / 0.75 | 0.70 / 0.48 | 0.02 / 0.58 | 367 |

Baseline seeds span mIoU 0.381 to 0.398, so differences of about 1 to 2 points are seed noise. Latency was measured on Kaggle CPU and is informational only, since the current target is simulation.

### Graphs
![Recall and precision on the hazard classes, with recall gates](perception_training/docs/comparison_results_2026-09-29/fig_focus_classes.png)
*Recall and precision on the hazard classes, with recall gates*

![mIoU, accuracy vs CPU latency, validation-loss curves](perception_training/docs/comparison_results_2026-09-29/fig_summary.png)
*mIoU, accuracy vs CPU latency, validation-loss curves*

![RELLIS-3D test predictions](perception_training/docs/comparison_results_2026-09-29/fig_rellis_predictions.png)
*RELLIS-3D test predictions*

![Sample frames: image, ground truth, predictions, entropy](perception_training/docs/comparison_results_2026-09-29/fig_samples.png)
*Sample frames: image, ground truth, predictions, entropy*

### Pros and cons
**TerrainSegModel (baseline, 3 seeds)**
- Pros: person recall 0.975 to 0.988 on every seed; log recall about 0.58 to 0.60; mIoU 0.381 to 0.398, on par with SegFormer; already wired into the ONNX and entropy-map pipeline; well calibrated.
- Cons: mud recall about 0.52 to 0.64, below the 0.70 gate; water recall unstable across seeds (0.275 to 0.735); dirt and building classes are near zero.

**SegFormer-B1 (2 seeds)**
- Pros: mud recall 0.845 and 0.703, about 0.17 above the baseline mean; strong rubble (0.90), pole (0.50) and fence (0.77) on seed 0; person recall 0.983 and 0.988; test ECE 0.010 to 0.013; puddle recall 0.88 to 0.91.
- Cons: water recall 0.234 and 0.022, failing the 0.27 gate on both seeds; log recall 0.001 on seed 0; mIoU gain of about 1.2 points is inside seed noise; overfits early (best epoch 3 to 5); seed 1 barely clears the mud gate.

**SegFormer-B0 (1 seed)**
- Pros: mud recall 0.815; person recall 0.985; mIoU 0.398.
- Cons: water recall 0.006; log recall 0.007; the worst calibration (ECE 0.033).

**Mud-weighted baseline (mud weight x2, rejected)**
- Pros: mud recall 0.893.
- Cons: mud precision 0.18, water precision 0.04, log recall 0.23, mIoU 0.347. It reaches its recall by over-predicting mud.

### Why the baseline stays
1. No model beats it outside seed noise on mIoU.
2. Every SegFormer variant fails the water gate, and B0 and B1 (seed 0) also fail log recall. Both are physical hazards.
3. The one real SegFormer gain (mud recall, about +0.17 to +0.25) does not offset the water and log failures.

### Open issues and caveats
- **Water and mud are data and evaluation problems, not architecture problems.** Test has about 109k water pixels and val has none. The baseline itself misses the water and mud gates on average, so treat both classes conservatively in the costmap (high cost, entropy-based inflation).
- Val shares sequences with train (ADR 0003), so val flatters every model. Weigh test more.
- Only 1 to 3 seeds per model, and RELLIS-3D only. Gazebo evaluation has not been run yet.
- Reopen model work only if simulation shows a specific failure. The first fix would then be adding a small set of Gazebo-rendered frames to training.

Raw results, per-seed JSONs, training histories and figures: `perception_training/docs/comparison_results_2026-09-29/`.
Reproduce with `src/train.py --model <name> --seed <n>`, then `src/compare_eval.py run` and `table`.
<!-- comparison-2026-09-29:end -->
