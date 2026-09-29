# Pretrained comparison, 2026-09-29

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
![Recall and precision on the hazard classes, with recall gates](fig_focus_classes.png)
*Recall and precision on the hazard classes, with recall gates*

![mIoU, accuracy vs CPU latency, validation-loss curves](fig_summary.png)
*mIoU, accuracy vs CPU latency, validation-loss curves*

![RELLIS-3D test predictions](fig_rellis_predictions.png)
*RELLIS-3D test predictions*

![Sample frames: image, ground truth, predictions, entropy](fig_samples.png)
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

Raw results, per-seed JSONs, training histories and figures: this folder.
Reproduce with `src/train.py --model <name> --seed <n>`, then `src/compare_eval.py run` and `table`.
