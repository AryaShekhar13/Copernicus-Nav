# Pretrained comparison, 2026-09-29 (test split, shipped-v1 recipe)

Decision: keep TerrainSegModel for deployment. Do not adopt SegFormer.

Test results (recall/precision where given):
- segformer_b1_s0: mIoU 0.395, mud recall 0.845, water recall 0.234, person 0.983, CPU ONNX 333 ms
- segformer_b1_s1: mIoU 0.404 (hazard 0.427), ECE 0.013, mud 0.703, water 0.022, puddle 0.909, person 0.988, CPU ONNX ~333 ms
- baseline seeds: mIoU 0.381-0.398, mud recall 0.52-0.64, water recall 0.275-0.735, CPU ONNX 154-177 ms

Why SegFormer is rejected:
- B1 mean mIoU is about 1.2 points above the baseline mean, inside baseline seed spread (about 1.7 points). Not a real gain.
- B1 fails the water recall gate (>= 0.27) on both seeds (0.234, 0.022) and log recall on seed 0 (0.001).
- B1 is about 2x slower on CPU ONNX (333 vs 154-177 ms); fails the 200 ms / 5 Hz example gate.
- B0 fails water (recall 0.006).
- Mud-weighted baseline (mud weight x2) rejected: mud precision 0.18, water precision 0.04, mIoU 0.347.

Open issues (data/evaluation, not architecture):
- Water: val has no water pixels, test has about 109k. Baseline itself misses the water gate across seeds.
- Baseline mud recall (about 0.60) misses the 0.70 gate. Treat water and mud conservatively in the costmap.
- Val shares sequences with train (ADR 0003), so val flatters every model; weigh test more.
- CPU timings are relative (Kaggle). Re-time the ONNX files on the target laptop.

Next: Gazebo frame set with ground-truth labels, scored with the same recall/precision gates
(person recall >= 0.95). Reopen model work only if Gazebo shows a specific failure.
