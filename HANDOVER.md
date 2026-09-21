# HANDOVER

(Base document not found in this session — appending today's updates below. Merge with the full version manually.)


---
## Session update — 2026-09-21
### New critical items

9. **§14.2 scene-diversity check: RUN, conclusive.** Rubble's test weakness is a
   pure unseen-environment problem (test's only rubble scene, 00001, has ZERO
   rubble exposure in train) — not retrain-fixable. Mud's weakness is a genuine
   model confusion (verified via per-scene precision breakdown: the model
   over-predicts mud even on scenes it trained on, e.g. scene 00002 at 0.022
   precision) — retrain-fixable in principle. Both are now real §7 findings.

10. **§14.1 dice-weight-0 ablation: RUN, promising but n=1.** Confirms Dice
    wasn't the cause of v2's ECE blowup (this run: ECE 0.0084 test / 0.0281 val,
    healthy). Test mud precision improved 0.25→0.39. Test rubble recall
    improved 0.21→0.35 but is likely retrain variance, not a real effect,
    given the unseen-scene finding above — do not report this as fixed.
    Branch: `dice-weight-0-ablation-2026-09-21`, not yet merged anywhere,
    needs a second seed before being treated as confirmed.


### 14.3 Dice-weight-0 ablation — results (2026-09-21)

Ran to completion. Single-variable vs v1: `--dice-weight 0.0` (= plain CE,
since total = ce + 0*dice), original v1 class_weights, no hflip. 20-epoch
budget, early-stopped epoch 11 (patience=5, best at epoch 3). Gated correctly
at 3 epochs via `evaluate.py --max-batches 60` (sky/grass/tree IoU all >0.5)
before spending full budget — no repeat of §6.1's collapse.

**Val** (comparable n=15 to v1's table): mIoU 0.4326 vs v1's 0.4192, ECE 0.0281
vs v1's 0.0399. Every hazard class's IoU improved.

**Test** (v1's evaluate.py had no `--split` flag — added one, see committed
`evaluate_with_max_batches_and_split.py`):

| Metric | v1 (test) | This run (test) |
|---|---|---|
| mud recall / precision | 0.76 / 0.25 | 0.49 / 0.39 |
| rubble recall / precision | 0.21 / 0.64 | 0.35 / 0.69 |
| ECE | not computed | 0.0084 |

(Precision derived: TP=recall×GT_px, union=TP/IoU, FP=union−GT_px,
precision=TP/(TP+FP) — `evaluate.py` only prints IoU/recall.)

**Interpretation:** mud's precision gain is plausible and mechanistically
consistent with the per-scene breakdown below — worth a second seed to
confirm. Rubble's recall gain is most likely noise (see §14.2's finding that
train has zero rubble exposure in test's only rubble scene — a single retrain
can't have learned features from data it never saw). **Do not report rubble
as improved without a second seed contradicting the scene-scarcity finding.**

Committed to branch `dice-weight-0-ablation-2026-09-21` (not `main`, not
`hazard-weighted-retrain` — it's its own experiment folder,
`perception_training/experiments/dice_weight_0_ablation/`, with a README
covering the same caveats as here). Not merged; treat as a lead, not a
decision.

### 14.4 Rubble/mud per-scene diagnostic (2026-09-21)

§14.2's scene-diversity check completed:

| Class | Train scenes | Test scenes | Test-only (unseen) |
|---|---|---|---|
| rubble | 00000, 00003, 00004 | 00001 | 00001 (100% of test rubble) |
| mud | 00000, 00002, 00003, 00004 | 00000, 00001, 00002 | 00001 only |

Followed with a per-scene precision/recall breakdown (v1 ONNX inference,
`v1-segmodel-2026-09-19` release, SHA256 verified) on mud's 3 test scenes:

| Scene | In train? | Precision | Recall | FP share |
|---|---|---|---|---|
| 00000 | yes | 0.222 | 0.968 | 38.7% |
| 00001 | no | 0.354 | 0.690 | 40.7% |
| 00002 | yes | 0.022 | 0.677 | 20.5% |

**Conclusion:** rubble's test failure is a pure data-coverage gap (not
retrain-fixable — needs rubble-labeled frames from more/different scenes).
Mud's test failure is a genuine model confusion — the unseen scene actually
has better precision than two scenes the model trained on, ruling out
scene-scarcity as mud's main problem. Scene 00002 in particular is a striking
outlier (near-zero real mud, 98% false-positive rate) worth a quick visual
check (not yet done) before further retrain investment.

**This resolves item 7 in the critical-items list — ready to go to WS2.**

### Kaggle environment notes addendum

- **Git push via `~/.git-credentials` + `credential.helper store` failed
  with "Invalid username or token" on 2026-09-21 despite a valid, unexpired
  token with correct scope.** Root cause not fully isolated — possibly a
  Kaggle sandboxing quirk with the credential store. Workaround that worked:
  embed the token directly in the push URL instead
  (`git push https://{token}@github.com/...`), never printed/logged. Try
  this first in future sessions rather than re-debugging the credential
  store.
