# Pretrained comparison, 2026-09-29 (test split, recipe = shipped v1, no augmentation)

Decision: keep TerrainSegModel for deployment. Do not adopt SegFormer.

- No model beats the baseline on mIoU outside seed noise (baseline seeds: 0.381-0.398).
- SegFormer B0/B1 raise mud recall (0.82/0.85 vs 0.52-0.64 for baseline seeds) but B1 is ~2x slower on CPU ONNX (333 vs 154-177 ms) and B0 fails water (recall 0.006, ECE 0.033).
- Water is unresolved for every model: baseline recall 0.735/0.275/0.425 over three runs; val has no water pixels, test has ~109k.
- Mud-weighted baseline (mud weight x2) rejected: mud precision 0.18, water precision 0.04, mIoU 0.347.
- Baseline misses the water and mud recall gates; treat both classes conservatively downstream.
- CPU timings are relative (Kaggle); re-time ONNX files on the target laptop.
