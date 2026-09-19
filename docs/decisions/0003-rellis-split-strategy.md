# ADR 0003: Use Official RELLIS-3D Frame-Level Split (Deviation from Sequence-Level Requirement)

## Status
Accepted

## Context
`team_division.md` requires a sequence-level train/val/test split for RELLIS-3D ("no frame-random shuffling") to avoid leakage between temporally adjacent frames.

Inspection of the official RELLIS-3D release's `Rellis_3D_image_split` (train.lst / val.lst / test.lst) shows it is frame-level, not sequence-level:
- train sequences: 00000, 00002, 00003, 00004
- val sequences: 00000, 00001
- test sequences: 00000, 00001, 00002
- Overlaps: train/val share 00000; train/test share 00000 and 00002; val/test share 00000 and 00001.

With only 5 total sequences (00000-00004), building a strict sequence-level split ourselves would leave very small, coarse val/test sets and would not be directly comparable to published RELLIS-3D benchmark numbers.

## Decision
Use the official RELLIS-3D split (train.lst / val.lst / test.lst) as-is for training and evaluation, for direct comparability with published results and the broader literature.

## Consequences
- Val/test metrics (mIoU, per-class IoU, ECE/calibration) are expected to be optimistically biased relative to true generalization to unseen terrain/runs, because of sequence overlap across splits.
- This is a known, documented limitation of this project's perception evaluation numbers, not an oversight. It should be stated alongside any reported metrics.
- Workstream 5's simulation-based evaluation (closed-loop episodes in Gazebo, not RELLIS-3D frames) is unaffected by this decision and remains the primary measure of real generalization, since it never reuses RELLIS-3D frames at all.
- If time permits later, an additional sequence-level split can be produced and reported side-by-side as a stricter lower-bound estimate; not required for initial implementation.
