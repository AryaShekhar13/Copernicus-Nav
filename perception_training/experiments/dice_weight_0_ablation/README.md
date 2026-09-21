# Dice-weight-0 ablation (2026-09-21)

**Status: preliminary, n=1. Not a validated fix — see caveats below.**

## What this is
Single-variable retrain vs v1's original recipe: `--dice-weight 0.0`
(mathematically identical to plain CE, since total = ce + 0*dice),
original v1 class weights (NOT the 4x-boosted hazard-weighted-retrain weights),
no hflip (hazard-weighted-retrain's dataset.py hardcodes it on; this run used
main's dataset.py instead). Purpose was narrow: isolate whether Dice caused
v2's ECE blowup (0.0399 -> 0.1541) and log-class collapse. It was explicitly
NOT scoped as a rubble/mud fix attempt.

20-epoch budget, early-stopped at epoch 11 (patience=5, best val loss at epoch 3).
Gate-checked at 3 epochs via `evaluate.py --max-batches 60` before committing
full budget (sky/grass/tree IoU all >0.5, no repeat of the §6.1 collapse).

## Results vs v1 baseline

### Val (full evaluate.py, no --max-batches)
| Metric | v1 baseline | This run |
|---|---|---|
| Overall mIoU (n=15) | 0.4192 | 0.4326 |
| ECE | 0.0399 | 0.0281 |
| mud IoU/recall | 0.4735 / 0.8651 | 0.5202 / 0.7269 |
| rubble IoU/recall | 0.4103 / 0.6306 | 0.4909 / 0.7559 |

### Test (--split test, added via `evaluate_with_max_batches_and_split.py`)
| Metric | v1 baseline | This run |
|---|---|---|
| mud recall / precision | 0.76 / 0.25 | 0.49 / **0.39** |
| rubble recall / precision | 0.21 / 0.64 | 0.35 / 0.69 |
| Test ECE | not computed | 0.0084 |

(Precision derived from printed IoU/recall/GT-px: TP=recall*GT, union=TP/IoU, FP=union-GT.)

## Interpretation

- **Confirmed:** removing Dice does not reproduce v2's calibration collapse.
  v2's problems came from its specific combination of changes (4x weights +
  Dice 0.5 + hflip), not from retraining being inherently unstable.
- **Promising, plausible mechanism:** mud precision improvement (0.25->0.39)
  is consistent with the prior per-scene diagnostic (see main HANDOVER.md
  §14.2 follow-up), which found mud's test failure is a genuine model
  confusion (over-predicting on grass, even on scenes seen in training),
  not a data-coverage gap -- so a cleaner retrain has a real mechanism to
  help it. This is one run; needs a second seed to confirm it's not variance.
- **Likely noise, do not over-claim:** rubble recall improvement (0.21->0.35).
  Test's only rubble scene has ZERO rubble exposure in train (confirmed via
  scene-diversity check). A single retrain can't have learned rubble
  features from a scene it never saw; this is far more likely this run's
  random init generalizing slightly better by chance than a real effect.

## What this is NOT
- Not a decision to adopt this checkpoint as the new shipped model.
- Not confirmation mud precision reliably improves (n=1).
- Not evidence rubble's fundamental data-scarcity problem is solved.

## Next step if pursued
Repeat this exact recipe (same configs, same code in this folder) with 1-2
more random seeds. If mud precision consistently lands >0.35 on test across
seeds, that's a real, actionable finding for a future retrain proposal.
