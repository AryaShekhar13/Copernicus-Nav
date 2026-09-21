# Architecture — Uncertainty-Aware Off-Road Traversability & Navigation

Status: **Perception (Workstream 1) implemented and released (`v1-segmodel-2026-09-19`).
Workstreams 2–5 remain draft / not yet implemented.**

Scope: simulation stack choice, repo/package layout, module boundaries, interface
contracts, config management, data/model management, launch structure, testing,
logging/eval, local-vs-Kaggle split, team/branch workflow, and open decisions —
for the 5 workstreams defined in `team_division.md`.

**Project scope note: this is a simulation-only project. No physical robot is being
built.** Every reference to a robot platform, sensor mount, or actuator model below
describes a *simulated* entity used inside Gazebo — chosen for realistic dynamics and
sensor behavior, not for matching a real chassis that will ever be fabricated. Nothing
in this document should be read as implying hardware procurement, wiring, or a bring-up
checklist. Where the original proposal referenced "matching a real datasheet," that
language has been removed or reframed below as "simulation fidelity," since there is no
physical unit to match against.

---

## 1. Simulation Stack

**Choice: ROS 2 Humble + Gazebo Fortress (via `ros_gz`), Ubuntu 22.04.**

- Humble is the current LTS with the longest support window and the most mature Nav2
  integration, documentation, and community fixes — this matters most for the costmap
  plugin work (Workstream 2) and `ros2_control`/`gz_ros2_control` (Workstream 4).
- Fortress is the Gazebo release officially paired with Humble through `ros_gz`.
  Jazzy + Harmonic is newer but requires Ubuntu 24.04 and has thinner third-party
  package coverage (ORB-SLAM3 wrappers, off-road robot description packages), which
  matters for Workstream 3 and 4.
- No discrete GPU required for this pairing: Gazebo physics is CPU-bound, and sensor
  rendering (camera, LiDAR) can run headless via software rasterization
  (`LIBGL_ALWAYS_SOFTWARE=1` / `MESA_GL_VERSION_OVERRIDE`) at reduced but acceptable
  frame rates for development. GPU work (segmentation training) is fully offloaded to
  Kaggle — see Section 9.
- This is a recommendation, not a hard requirement of the interface contracts — the
  topic/message contracts below are ROS 2-generic and unaffected by the Humble/Fortress
  choice.

**Simulated robot reference:** a differential-drive UGV modeled on the Clearpath
Jackal's public URDF/Xacro and `ros2_control` examples — used purely as a well-documented,
simulation-ready starting point (mass, footprint, sensor mount points, differential-drive
controller stack), not as a spec to be physically reproduced. **Still open (WS4 to
finalize)**, but should be locked Day 0 since it drives simulated inertial parameters,
simulated actuator limits, and sensor-noise models used inside Gazebo.

---

## 2. Top-Level Repository Layout

Monorepo, single Git history, package-per-workstream ownership so PR review and
CODEOWNERS map cleanly to people.

```
offroad-nav/
├── README.md, CONTRIBUTING.md
├── docs/                        (architecture.md, interfaces.md, decisions/, runbooks/)
├── docker/                      (Dockerfile.dev, Dockerfile.ci, devcontainer.json)
├── .github/workflows/           (ci.yml, lint.yml)
├── scripts/                     (bootstrap.sh, check_interfaces.py, download_checkpoint.sh)
├── ros2_ws/src/
│   ├── terrain_interfaces
│   ├── terrain_description      (simulated robot model only — no hardware BOM)
│   ├── terrain_gazebo_worlds
│   ├── terrain_perception
│   ├── terrain_costmap
│   ├── terrain_localization
│   ├── terrain_nav2_config
│   ├── terrain_eval
│   └── terrain_bringup
├── perception_training/         (notebooks/, src/, configs/, splits/, exported_models/)
├── results/                     (gitignored)
└── tests/
```

Rationale for keeping `perception_training/` outside `ros2_ws/`: it has a completely
different dependency stack (PyTorch, CUDA, `timm`) that shouldn't collide with the
ROS 2 workspace's Python environment, and it runs on Kaggle, not locally. The only
thing that crosses the boundary is an exported model artifact + a `classes.yaml` —
the interface contract defined in Section 5.

**Implementation status:** only `perception_training/` exists in the repo so far
(`configs/`, `src/`, `exported_models/` partially — see Section 8). `ros2_ws/`,
`results/`, `tests/`, `docker/`, `.github/workflows/`, and `scripts/` have not been
scaffolded yet.

---

## 3. Package Responsibilities & Dependency Graph

Data flow: `terrain_description` + `terrain_gazebo_worlds` publish sensor topics →
`terrain_perception` (segmentation + uncertainty) and `terrain_localization` (SLAM)
consume camera/imu → `terrain_costmap` consumes perception output + tf → Nav2 core
(`terrain_nav2_config`) consumes costmap + produces `/cmd_vel` → `terrain_eval`
orchestrates episodes and logs results across the whole stack.

No package other than `terrain_bringup` and `terrain_eval` may depend on more than one
other workstream's package — this is the main lever for parallel development and
independent testability.

**Implementation status:** `terrain_perception` (the ROS node itself) has not been
built. What exists is its Kaggle-side dependency: a trained, exported, released model
(Section 8/9) plus two offline Python entry points (`perception_training/src/inference.py`,
`perception_training/src/pipeline.py`) that a future `terrain_perception` node should
wrap, not duplicate. No other package in this graph has any implementation yet.

---

## 4. Deviations From the Original Proposal (with justification)

1. `perception_training/` is split out of the ROS workspace entirely. Training never
   runs inside colcon/ROS — it's a separate Python project whose only deliverable
   crossing into ROS-land is a frozen checkpoint + `classes.yaml`.
2. `terrain_nav2_config` is split out from `terrain_costmap`, so the plugin stays a
   narrow, independently-buildable, independently-testable library while Nav2-wide
   tuning lives separately.
3. **(ADR 0003)** The RELLIS-3D split is the *official* frame-level train/val/test
   split, not a custom sequence-level split as originally intended. The official split
   has sequence overlap across train/val/test. Adopted for comparability with published
   RELLIS-3D benchmarks; documented as a known source of optimistic bias in reported
   val/test metrics — every quantitative claim made anywhere downstream of this split
   inherits that bias and should be read as an upper bound, not a ground-truth number.
4. Class-weighted loss was added after the unweighted baseline revealed several classes
   (`person`, `log`, `rubble`) were essentially undetected (IoU 0). Training-time change,
   not a contract deviation, but it changes which checkpoint is "the" model —
   see Section 8.
5. **(New)** All hardware-matching language from the original proposal has been removed.
   The simulated robot model exists to give Gazebo realistic dynamics and sensor
   placement — it is not, and will never be, checked against a physical build.

Everything else (topics, message types, costmap plugin API, results schema, the
"`/gt_gps` never subscribed to by planner/costmap/SLAM" rule) is kept as specified.

---

## 5. Interface Contracts

**Decision: segmentation + uncertainty are published as two standard
`sensor_msgs/Image` topics, not a custom message.** (ADR 0002, signed off by WS1 and WS2.)

- `/perception/segmentation` — `sensor_msgs/Image`, `mono8` (or `mono16` if >255 classes),
  per-pixel class index.
- `/perception/uncertainty` — `sensor_msgs/Image`, `32FC1`, same resolution, same
  header stamp.
- `classes.yaml` — static file (not a topic), class-index → name → default base-cost
  mapping, versioned alongside the model checkpoint it corresponds to.
- Consumers synchronize the two topics with `message_filters::ApproximateTimeSynchronizer`
  on header stamp.

This avoids a `terrain_interfaces` build-time dependency across all five workstreams,
and standard tools (rviz2, ros2 bag, image_view) work on plain Image topics for free.

Everything else in the original contract table is adopted unchanged: `/camera/image_raw`,
`/imu`, `/scan`/`/points`, `/gt_gps` (eval-only, hard-firewalled), `/cmd_vel`, tf +
`nav_msgs/Odometry`, the `CostmapLayer` plugin API, and the per-episode results schema.

**Hard rule, to be enforced by tooling, not just convention:** `/gt_gps` must never be
subscribed to by any node under `terrain_costmap`, `terrain_nav2_config`, or
`terrain_localization`'s live (non-eval) code path. `scripts/check_interfaces.py` should
grep all package source for `/gt_gps` subscriptions and fail CI if found outside
`terrain_eval`. **Not yet implemented — `scripts/` doesn't exist yet. Currently this
rule is convention only.**

**Known contract gap — top priority, blocks Workstream 2:** `/perception/uncertainty`
is a required output of `terrain_perception`, but the uncertainty method actually
validated and released (MC-Dropout — Section 9) does not run on the deployed ONNX
model. ONNX export freezes the model to a single deterministic forward pass;
MC-Dropout requires multiple stochastic passes with dropout active, which the exported
graph does not support. **There is currently no implementation that can populate
`/perception/uncertainty` from the live/deployed model.**

**Resolution path (decision still to be formally locked as `0005-deployed-uncertainty-method.md`):**
five options exist (PyTorch-live, multi-mask ONNX export, stochastic-Dropout ONNX,
single-pass entropy proxy, deep ensemble) — see Section 13, risk #13, for the full
comparison. Recommendation: if `terrain_perception` may depend on PyTorch directly,
use it live and reuse the validated MC-Dropout numbers exactly (cheapest, zero new
validation work). If the deployed path must be ONNX-only, fall back to a single-pass
softmax-entropy proxy — trivial to add to an ONNX equivalent of `pipeline.py`, but
requires its own calibration/ECE re-validation before the existing numbers can be
trusted for the new signal. **Do not begin Workstream 2's risk-aversion costmap layer
until this is locked** — building against an undefined signal compounds the gap
instead of resolving it.

---

## 6. Configuration & Parameter Management

- Topic names defined once in `docs/interfaces.md`, mirrored as ROS 2 launch arguments
  — never hardcoded inside node source. `terrain_bringup` is the only place that wires
  concrete remappings together.
- Node parameters: one YAML per node under each package's `config/`, loaded via
  `launch_ros`'s `parameters=[...]`.
- Episode/terrain configs: PyYAML files under `terrain_eval/config/episodes/` and
  `terrain_gazebo_worlds/config/terrain/`.
- Results schema: defined once as a JSON Schema in `terrain_eval/schema/results_schema.json`;
  the metrics logger validates each row before writing.
- Risk-aversion parameter: exposed as a normal Nav2-style dynamic parameter on the
  costmap plugin, default + valid range to be documented in `docs/interfaces.md` once
  Section 5's uncertainty-method decision is locked (the parameter's meaning depends
  on which uncertainty signal it's weighting).

**Implementation status:** `perception_training/configs/` exists and is in active use
(`classes.yaml`, `dataset.yaml`, `training.yaml`). None of the ROS-side config
(`docs/interfaces.md`, per-node YAMLs, episode configs, results schema) exists yet.

---

## 7. Launch File Organization

`terrain_bringup/launch/` will contain: `sim_bringup.launch.py` (Gazebo + simulated
robot + sensors), `perception.launch.py` (real or `--mock` perception node),
`localization.launch.py` (SLAM node), `nav2_bringup.launch.py` (Nav2 core + costmap
plugin + nav2 config), `full_stack.launch.py` (composes all, `mode:=mock|integration|full`),
`eval_harness.launch.py` (full_stack + episode runner).

Each individual launch file must be runnable standalone against stub/mock inputs from
every other workstream. Each package ships a lightweight mock/stub node so downstream
packages never block on upstream being finished. **Not yet implemented — no `ros2_ws/`
packages exist yet.** This is the single highest-leverage next step for build maturity
(see the companion note on sequencing at the end of this document).

---

## 8. Data / Model / Checkpoint Organization

- RELLIS-3D dataset itself is never committed to Git; downloaded via script or attached
  as a Kaggle Dataset. Only the frozen train/val/test sequence-split file is committed.
  **(ADR 0003)** the committed split is the official RELLIS-3D frame-level split, not a
  custom sequence-level split — see Section 4, deviation 3.
- Model checkpoints are not committed raw to Git. Exported models (ONNX preferred,
  TorchScript fallback) are attached to versioned GitHub Releases, referenced from
  `perception_training/exported_models/README.md` with filename convention
  `segmodel_v{n}_{date}.onnx`, a SHA256 checksum, and the paired `classes.yaml`.
- Every exported model records its exact training config snapshot, split file version,
  and run ID in the release notes for reproducibility.

**Implementation status:**
- **Released:** `v1-segmodel-2026-09-19` — `segmodel_v1_2026-09-19.onnx` (opset 17) +
  `classes.yaml`, SHA256 `23910cc4953e093c7d1c1b449f288b00a07934fdb07f4cf599da192de6fb7e0f`,
  validated against the source PyTorch model (max abs diff 4.18e-05).
- **Model selected: the class-weighted checkpoint (epoch 8)**, not the unweighted
  baseline — mIoU 0.331 vs. 0.346, ECE 0.041 vs. 0.036, traded for fixing zero recall
  on `person` and `log`. **This decision should be formalized as an ADR** — it is
  currently only in release notes, not `docs/decisions/`.
- **Known process risk:** the shipped v1 checkpoint currently exists only as an ONNX
  export — it is unrecoverable in PyTorch form. Any future retrain is from scratch, not
  fine-tuning. If the ONNX file is ever lost or found to have an undiscovered export
  bug, there is no way back to the original trained weights. A committed, reproducible
  training recipe is a prerequisite for calling anything here shipping-ready.
- **Temporary deviation from the "checkpoints not committed raw" rule:** `best.pt` was
  pushed directly to a `best-checkpoint-results` branch as a safety net against Kaggle
  kernel/session resets. This branch is **not** the release artifact, is currently
  stale relative to `main`, and lacks a README distinguishing it from the shipped
  model. **Disposition still open** — should be brought up to date, deleted now that
  the ONNX release supersedes its purpose, or explicitly documented as a raw-checkpoint
  archive only. See Section 15 for the recommended resolution.
- `perception_training/exported_models/README.md` referenced above does not exist yet.
- Two source modules exist beyond original scope, both in `perception_training/src/`:
  `inference.py` (frozen inference on pre-batched tensors, offline/Kaggle-side) and
  `pipeline.py` (full raw-image-in pipeline including MC-Dropout, offline/Kaggle-side
  only — **not** the live ONNX/CPU deployment path; see Section 9's uncertainty gap).

---

## 9. Local vs. Kaggle Split

**Runs locally (CPU only):** Gazebo simulation (headless/software render),
Nav2/costmap/planner/controller, ORB-SLAM3/RTAB-Map, `terrain_eval` episode generator
and logging, ONNX-runtime inference of the already-trained exported model.

**Runs on Kaggle (GPU):** RELLIS-3D fine-tuning of the segmentation backbone,
MC-Dropout/ensemble uncertainty training, batch offline inference for
mIoU/precision-recall/ECE evaluation, large-batch calibration/reliability-diagram
computation.

**Key risk:** live CPU inference during a real-time Gazebo run may not keep up with the
costmap's required update rate. **Still not benchmarked end-to-end as of this document**
— the one internal latency figure available (2.9–4.7 FPS, from earlier perception-only
profiling) is not a validated real-time number; it was measured in isolation, not inside
a live Gazebo/Nav2 loop, and a prior benchmarking attempt was contaminated by orphaned
background processes. Mitigation: `terrain_perception` should support a `--mock`/`--replay`
mode that either replays precomputed Kaggle-batch-generated frames time-aligned to a
recorded bag, or runs the real ONNX model live at reduced resolution/rate, benchmarked
ahead of time.

**Clarification:** this split was always correct — MC-Dropout was always scoped to
Kaggle, never to live CPU inference. The document deliberately treats "MC-Dropout/ensemble"
as interchangeable Kaggle-side training/evaluation options; it does not mandate
MC-Dropout specifically for the deployed path, which leaves room for the single-pass
entropy fallback in Section 5 without contradicting this section.

---

## 10. Testing Strategy

1. **Unit tests per package:** `terrain_costmap` cost-function tests (gtest)
   independent of Nav2/Gazebo; `terrain_perception` node tests using a dummy ONNX model
   so no real checkpoint or GPU is needed in CI; `terrain_eval` split-script and
   schema-validation tests with synthetic data.
2. **Contract/interface tests** (`launch_testing`): bring up one package plus mocks of
   its direct dependencies, assert the right topics/types/rates appear. Runs on every PR.
3. **Integration tests:** `full_stack.launch.py mode:=integration` with the real
   Gazebo world and Nav2, run periodically at phase boundaries, not on every commit.

CI runs layers 1 and 2, headless, no GPU needed, since nothing GPU-dependent is
exercised outside Kaggle.

**Implementation status:** none of this exists yet — no `terrain_perception` node, no
CI config, no `tests/`. The real `v1-segmodel-2026-09-19.onnx` release is available now,
so the "dummy ONNX model" in item 1 could reasonably be swapped for a tiny slice of the
real one once `terrain_perception` exists, if preferred over a synthetic stub.

---

## 11. Logging / Evaluation / Results Storage

Every episode run writes one row to `results/runs/{run_id}/episodes.csv`, matching the
results schema exactly. Each run directory stores a config snapshot and optionally a
ros2 bag (gitignored). `terrain_eval`'s aggregation notebook reads all runs, tags by
system variant, and produces final comparison tables/plots. Training-side logging stays
inside `perception_training/`; only final exported-model metrics get copied into
release notes.

**Implementation status:** training-side metrics (mIoU, per-class IoU/precision/recall,
ECE, MC-Dropout entropy) have been tracked ad hoc in the Kaggle notebook and copied into
Release notes, not through a proper experiment tracker. Fine for a single-person
perception workstream so far; worth adopting before more training runs happen — a
kernel reset has already cost work once (`inference.py`/`pipeline.py` needed rewriting).

---

## 12. Team Workflow / Maintainability

Package-to-person mapping is 1:1 with the five workstreams. CODEOWNERS enforces review
by package owner; interface doc changes need sign-off from every affected workstream.
Short-lived feature branches, PRs gated by CI. ADRs capture every contract-level
decision. Docker/devcontainer pins exact versions. `ament_lint` / black+flake8 in CI.

**Implementation status:** only `docs/decisions/0003-rellis-split-strategy.md` exists
as a real ADR so far. The weighted-vs-unweighted model selection (Section 8) and the
uncertainty/ONNX gap (Section 5/13) are exactly the kind of contract-level decisions
this section says should get an ADR, and don't have one yet:

- `0004-checkpoint-resume-strategy.md` (the best.pt/latest.pt early-stopping and
  resume-safety fix — informally referred to as "ADR 0004" until now)
- `0005-deployed-uncertainty-method.md` (once Section 5's open decision is locked)
- `0006-weighted-model-selection.md` (formalizing the class-weighted checkpoint choice)

See Section 15 for how branch/PR workflow should tie into ADR creation going forward.

---

## 13. Risks & Integration Problems

1. ORB-SLAM3 build friction — mitigate via pinned Docker image, RTAB-Map as documented
   fallback.
2. Live CPU-only perception inference may not meet real-time costmap rate — needs early
   latency benchmark. **Still not benchmarked.**
3. Gazebo/ROS 2 version mismatch — pin in Day 0 ADR and Docker image.
4. Segmentation wire format — **resolved** (Section 5, ADR 0002).
5. RELLIS-3D split reproducibility — **resolved via ADR 0003**, with the documented
   tradeoff of using the official (leaky) split instead of a custom sequence-level one.
6. Kaggle session/quota limits — training must be checkpoint-resumable from the start.
   **Partially bit us in practice:** a kernel restart lost an in-progress, uncommitted
   `inference.py` before it reached Git — commit working files more frequently, not
   just checkpoints.
7. C++ vs Python for costmap plugin — explicit team decision needed, affects WS2 schedule.
8. `/gt_gps` leakage — CI grep check plus manual review at integration boundaries.
   **Not yet implemented.**
9. Async sensor fusion staleness — acceptable staleness bound must be explicit and tested.
10. Nav2 BT/planner/controller tuning ownership — must be explicitly assigned.
11. Results schema drift — mitigated by JSON-Schema validation, only if every run uses
    the shared logger.
12. Repo bloat from checkpoints/bags — enforced via `.gitignore` and release-artifact
    pattern from commit #1. **Partially violated:** `best.pt` sits on
    `best-checkpoint-results`, not cleaned up now that the ONNX release supersedes it.
13. **Deployed model cannot currently produce `/perception/uncertainty`** — MC-Dropout,
    the validated uncertainty method, does not survive ONNX export. Options:
    - **(a)** Run the PyTorch model live for uncertainty — simplest, zero new
      validation, but works against risk #2's real-time CPU concern.
    - **(b)** Export several frozen ONNX graphs with different fixed dropout masks and
      average — stays ONNX-only, multiplies file size and inference cost.
    - **(c)** Keep ONNX's `Dropout` op stochastic via `training_mode` — closest to
      free, but fragile and provider-dependent; real risk of silently degrading to
      deterministic behavior.
    - **(d)** Single-pass uncertainty proxy (max-softmax-probability or entropy) for
      the deployed model — free, trivial to add, but methodologically different from
      what was validated; needs its own calibration check.
    - **(e)** Deep ensemble (3–5 independently-trained models, exported deterministically,
      averaged) — generally stronger than MC-Dropout, not a deviation from the original
      "MC-Dropout/ensemble" framing, but requires retraining multiple models from scratch.
    - **Recommendation:** (a) if a PyTorch dependency is allowed in `terrain_perception`
      at all — cheapest, reuses validated numbers exactly. If not, (d) is the pragmatic
      fallback, with an explicit re-validation step. (b)/(c)/(e) are higher-cost,
      probably not worth it unless (a)/(d) prove insufficient.
14. **(New)** No branch protection or enforced review currently exists — see Section 15.
    Direct pushes to `main` and to `best-checkpoint-results` are current practice; this
    is a real risk once more than one contributor is active, and worth tightening now,
    cheaply, before it is.

---

## 14. Decisions to Lock Before More Code Is Written (Day 0 checklist)

| Decision | Status |
|---|---|
| ROS 2 distro + Gazebo version pin (Humble + Fortress) | Proposed, not locked |
| Simulated robot reference (Jackal-derived) | Proposed, not locked |
| Segmentation/uncertainty wire format | **Locked** (Section 5, ADR 0002) |
| Model export format + inference runtime (ONNX) | **Locked**, validated and released |
| Deployed uncertainty method | **Open — top priority, see Section 5/13#13** |
| Costmap plugin language: C++ vs Python | Open, WS2 |
| Risk-aversion parameter (name/type/default/range) | Open — blocked on uncertainty-method decision |
| Acceptable staleness bound for perception frames | Open |
| Repo layout and package names | `perception_training/` locked and in use; `ros2_ws/` names still proposed |
| Results schema (JSON Schema) | Open, WS5/eval |
| RELLIS-3D split file | **Locked via ADR 0003** |
| Checkpoint storage/versioning mechanism | **Locked and working** — GitHub Release + SHA256 + `classes.yaml` |
| Docker/devcontainer base image | Open |
| Branch/PR policy and CODEOWNERS mapping | **Open — see Section 15** |
| `best-checkpoint-results` branch disposition | Open |

---

## 15. Main Branch Structure & Git Workflow

Current practice (direct pushes to `main` and to `best-checkpoint-results`, no PR or
review step) was workable for a single-person perception workstream but does not scale
to five workstreams landing code concurrently, and already has one loose end
(`best-checkpoint-results`) that this section closes out.

### 15.1 `main` branch contract

`main` should always be in a state that:
- Builds clean (`colcon build` for `ros2_ws/`, and `perception_training/` imports
  without error) — enforced by CI once `.github/workflows/ci.yml` exists.
- Never contains a raw model checkpoint (`.pt`) — only references to GitHub Releases,
  per Section 8's existing rule.
- Never contains dataset files — only the frozen split file, per Section 8.
- Has an accurate, up-to-date `HANDOVER.md` at the repo root at all times — this has
  historically drifted (documented as done, never actually uploaded); treat a stale or
  missing `HANDOVER.md` on `main` as a CI-checkable condition once tooling exists
  (e.g., a required-file check in `check_interfaces.py`), not just a manual habit.

### 15.2 Branch types

- **`main`** — protected, always buildable, the single source of truth. No direct
  pushes once more than one contributor is active (see 15.4).
- **`ws{n}/<short-description>`** — one short-lived feature branch per unit of work,
  scoped to a single workstream's package(s) wherever possible (e.g.
  `ws1/dice-weight-ablation`, `ws2/costmap-plugin-skeleton`). Matches the
  package-to-person 1:1 mapping from Section 12, so branch ownership and code
  ownership stay aligned.
- **`experiment/<short-description>`** — for exploratory work not yet intended to
  merge (e.g. this session's `dice-weight-0-ablation-2026-09-21` pattern). Allowed to
  be self-contained and messy internally, but must still include a README stating its
  purpose and status, and must never be the only copy of a result worth keeping — if an
  experiment branch produces a finding worth acting on, that finding gets written into
  an ADR and merged into `main`'s documentation even if the experiment code itself
  doesn't merge.
- **No long-lived parallel branches that duplicate `main`'s purpose.**
  `best-checkpoint-results` is exactly this anti-pattern — a checkpoint safety net that
  outlived its purpose once the ONNX Release mechanism existed, but was never cleaned
  up or clearly labeled as archival-only. **Resolution:** either (a) delete it now that
  the Release is the real backstop, or (b) if kept, rename to something
  self-documenting (e.g. `archive/pre-onnx-checkpoints`) with a README stating it is
  not a deployable branch and will not be updated further. Do not leave it in its
  current ambiguous, silently-stale state.

### 15.3 Commit and checkpoint safety

Given the Kaggle session-loss history (Section 9, Section 11, Section 13 risk #6 — work
has already been lost twice to kernel resets before reaching Git): commit working
Python source (`.py` files under `perception_training/src/`) to a feature branch
*frequently*, independent of whether a training run has finished — don't wait for a
"clean" checkpoint moment to push. Checkpoints themselves stay out of `main` per
Section 8's rule, but the *code* that produced them should never exist only inside a
live Kaggle session.

### 15.4 PR and review policy (to adopt once a second contributor joins any workstream)

- `main` becomes protected: no direct pushes, PR required.
- Each PR requires review from the relevant package's CODEOWNER (Section 12); PRs
  touching `docs/interfaces.md` require sign-off from every workstream whose package
  consumes the changed interface.
- CI (once it exists — Section 10, layers 1–2) must pass before merge.
- Every PR that makes a contract-level decision (changes a topic contract, a model
  selection, a checkpoint/versioning rule) must add or update an ADR under
  `docs/decisions/` in the same PR — the decision and its justification should never
  live only in a PR description or release notes, per the gap already identified in
  Section 8 and Section 12.

### 15.5 Merge cadence for `HANDOVER.md`

Given the repeated pattern this session of a merged handover being drafted but paused
mid-session (correctly, when told to hold off) and never actually landing on `main`:
treat `HANDOVER.md` updates as their own short-lived branch
(`docs/handover-<date>`), opened and merged in the same session whenever practical, so
the gap between "session's findings exist" and "findings are on `main` and
discoverable by the next person" stays as short as possible. If a session must pause
mid-update, say so explicitly in the branch/PR description so the next person (human or
otherwise) doesn't assume `main`'s `HANDOVER.md` is current.

---

## 16. Summary of Open Decisions, Ranked by Blocking Impact

1. **Deployed uncertainty method** (Section 5/13#13) — blocks all of Workstream 2's
   risk-aversion costmap logic. Highest priority.
2. **Simulated robot reference lock** (Jackal-derived, Section 1) — blocks
   `terrain_description` and downstream simulation fidelity.
3. **Costmap plugin language** (C++ vs Python, Section 13#7) — blocks Workstream 2
   scaffolding.
4. **`best-checkpoint-results` branch disposition** (Section 15.2) — cheap to resolve,
   currently a silent trap for anyone who lands there cold.
5. **Real-time latency benchmark** (Section 9, Section 13#2) — depends on #1 being
   resolved first (uncertainty compute cost must be fixed before an honest end-to-end
   number is possible).
6. Everything else in Section 14's table — cheap, parallelizable ADR-writing, no
   sequencing dependency on one another.
