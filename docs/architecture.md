# Architecture Proposal — Uncertainty-Aware Off-Road Traversability & Navigation

Status: **Perception (Workstream 1) implemented and released (`v1-segmodel-2026-09-19`). All other workstreams remain draft/not yet implemented.**
Scope: covers simulation stack choice, repo/package layout, module boundaries, contracts, config, data/model management, launch structure, testing, logging/eval, local-vs-Kaggle split, and team workflow for the 5 workstreams in `team_division.md`.

---

## 1. Simulation Stack

**Choice: ROS 2 Humble + Gazebo Fortress (via `ros_gz`), Ubuntu 22.04.**

- Humble is the current LTS with the longest support window and by far the most mature Nav2 integration, documentation, and community fixes — important because Nav2 costmap-plugin work (Workstream 2) and `ros2_control`/`gz_ros2_control` (Workstream 4) are exactly the areas where being on well-trodden versions saves the most debugging time.
- Fortress is the Gazebo release officially paired with Humble through `ros_gz`. Jazzy + Harmonic is newer but requires Ubuntu 24.04 and has thinner third-party package coverage (ORB-SLAM3 wrappers, Warthog/Jackal descriptions, etc.), which matters for Workstream 3 and 4.
- No local discrete GPU is fine for this pairing: Gazebo physics is CPU-bound, and sensor rendering (camera, LiDAR) can run headless via software rasterization (`LIBGL_ALWAYS_SOFTWARE=1` / `MESA_GL_VERSION_OVERRIDE`) at reduced but acceptable frame rates for development. GPU-only work (segmentation training) is fully offloaded to Kaggle — see Section 9.
- This is a recommendation, not a hard requirement of the document's contracts — the topic/message contracts in `team_division.md` are ROS 2-generic and unaffected by the Humble/Fortress choice.

**Robot platform reference:** Clearpath Jackal (not Warthog) — smaller, well-documented URDF/Xacro and `ros2_control` examples, sensor mounts easy to match to a real datasheet, and it keeps the differential-drive controller stack simple. This is Workstream 4's call to finalize but should be locked Day 0 since it drives URDF dimensions, actuator limits, and noise models. **Not yet locked as of this update** — still open for WS4.

---

## 2. Top-Level Repository Layout

Monorepo, single Git history, package-per-workstream ownership so PR review and CODEOWNERS map cleanly to people.

offroad-nav/

README.md, CONTRIBUTING.md
docs/ (architecture.md, interfaces.md, decisions/, runbooks/)
docker/ (Dockerfile.dev, Dockerfile.ci, devcontainer.json)
.github/workflows/ (ci.yml, lint.yml)
scripts/ (bootstrap.sh, check_interfaces.py, download_checkpoint.sh)
ros2_ws/src/ (terrain_interfaces, terrain_description, terrain_gazebo_worlds, terrain_perception, terrain_costmap, terrain_localization, terrain_nav2_config, terrain_eval, terrain_bringup)
perception_training/ (notebooks/, src/, configs/, splits/, exported_models/)
results/ (gitignored)
tests/

Rationale for keeping `perception_training/` outside `ros2_ws/`: it has a completely different dependency stack (PyTorch, CUDA, `timm`, W&B) that nobody wants colliding with the ROS 2 workspace's Python environment, and it runs on Kaggle, not locally. The only thing that crosses the boundary is an exported model artifact + a `classes.yaml`, which is exactly the interface contract already defined in the document.

**Implementation note:** only `perception_training/` exists in the repo so far (`configs/`, `src/`, `exported_models/` partially — see Section 8). `ros2_ws/`, `results/`, `tests/`, `docker/`, `.github/workflows/`, and `scripts/` have not been scaffolded yet — that's the next step once another workstream starts.

---

## 3. Package Responsibilities & Dependency Graph

Data flow: terrain_description + terrain_gazebo_worlds publish sensor topics -> terrain_perception (segmentation+uncertainty) and terrain_localization (SLAM) consume camera/imu -> terrain_costmap consumes perception output + tf -> Nav2 core (terrain_nav2_config) consumes costmap + produces /cmd_vel -> terrain_eval orchestrates episodes and logs results across the whole stack.

No package other than `terrain_bringup` and `terrain_eval` is allowed to depend on more than one other workstream's package — this is the main lever for parallel development and independent testability.

**Implementation note:** `terrain_perception` (the ROS node itself) has not been built yet. What exists today is its Kaggle-side dependency: a trained, exported, released model (Section 8/9) plus two offline Python entry points (`perception_training/src/inference.py`, `perception_training/src/pipeline.py`) that a future `terrain_perception` node implementation should wrap, not duplicate.

---

## 4. Deviations From the Document (with justification)

1. `perception_training/` is split out of the ROS workspace entirely. Training never runs inside colcon/ROS at all — it's a separate Python project whose only deliverable crossing into ROS-land is a frozen checkpoint + `classes.yaml`.
2. `terrain_nav2_config` is split out from `terrain_costmap`, so the plugin stays a narrow, independently-buildable, independently-testable library while Nav2-wide tuning lives separately.
3. **(New, ADR 0003)** The RELLIS-3D split is the *official* frame-level train/val/test split, not a custom sequence-level split as originally specified in this section's data-management intent. The official split has sequence overlap across train/val/test. Adopted for comparability with published RELLIS-3D benchmarks; documented as a known source of optimistic bias in reported val/test metrics. See `docs/decisions/0003-rellis-split-strategy.md`.
4. **(New)** Class-weighted loss was added after the unweighted baseline revealed several classes (`person`, `log`, `rubble`) were essentially undetected (IoU 0). This is a training-time change, not a contract deviation, but it changes which checkpoint is "the" model — see Section 8.

Everything else (topics, message types, costmap plugin API, results schema, the "GPS never subscribed to by planner/costmap/SLAM" rule) is kept as specified.

---

## 5. Interface Contracts — Refinements

**Decision: segmentation + uncertainty are published as two standard `sensor_msgs/Image` topics, not a custom message.**

- `/perception/segmentation` — `sensor_msgs/Image`, `mono8` (or `mono16` if >255 classes), per-pixel class index.
- `/perception/uncertainty` — `sensor_msgs/Image`, `32FC1`, same resolution, same header stamp.
- `classes.yaml` — static file (not a topic), class-index to name to default base-cost mapping, versioned alongside the model checkpoint it corresponds to.
- Consumers synchronize the two topics with `message_filters::ApproximateTimeSynchronizer` on header stamp.

This avoids a `terrain_interfaces` build-time dependency across all five workstreams, and standard tools (rviz2, ros2 bag, image_view) work on plain Image topics for free. This is captured as ADR 0002, signed off by WS1 and WS2.

Everything else in the document's contract table is adopted unchanged: `/camera/image_raw`, `/imu`, `/scan`/`/points`, `/gt_gps` (eval-only, hard-firewalled), `/cmd_vel`, tf + `nav_msgs/Odometry`, the `CostmapLayer` plugin API, and the per-episode results schema.

**Hard rule, enforced by tooling not just convention:** `/gt_gps` must never be subscribed to by any node under `terrain_costmap`, `terrain_nav2_config`, or `terrain_localization`'s live (non-eval) code path. `scripts/check_interfaces.py` greps all package source for `/gt_gps` subscriptions and fails CI if found outside `terrain_eval`. **Not yet implemented** — `scripts/` doesn't exist yet.

**Known contract gap, unresolved (new):** `/perception/uncertainty` is a required output of `terrain_perception`, but the uncertainty method actually validated and released (MC-Dropout — Section 9) does not run on the deployed ONNX model. ONNX export freezes the model to a single deterministic forward pass; MC-Dropout requires multiple stochastic passes with dropout active, which the exported graph does not support. **This means there is currently no implementation that can populate `/perception/uncertainty` from the live/deployed model.** See Section 9 and Section 13 (risk #13) for options and the decision still needed.

---

## 6. Configuration & Parameter Management

- Topic names defined once in `docs/interfaces.md`, mirrored as ROS 2 launch arguments — never hardcoded inside node source. `terrain_bringup` is the only place that wires concrete remappings together.
- Node parameters: one YAML per node under each package's `config/`, loaded via `launch_ros`'s `parameters=[...]`.
- Episode/terrain configs: PyYAML files under `terrain_eval/config/episodes/` and `terrain_gazebo_worlds/config/terrain/`.
- Results schema: defined once as a JSON Schema in `terrain_eval/schema/results_schema.json`; the metrics logger validates each row before writing.
- Risk-aversion parameter: exposed as a normal Nav2-style dynamic parameter on the costmap plugin, default + valid range documented in the interfaces doc.

**Implementation note:** `perception_training/configs/` exists and is in active use — `classes.yaml` (20-class mapping, hazard flags, color hex), `dataset.yaml` (RELLIS-3D paths/split files), `training.yaml` (hyperparameters, input size, checkpoint paths). None of the ROS-side config (`docs/interfaces.md`, per-node YAMLs, episode configs, results schema) exists yet.

---

## 7. Launch File Organization

`terrain_bringup/launch/` contains: `sim_bringup.launch.py` (Gazebo + robot + sensors), `perception.launch.py` (real or `--mock` perception node), `localization.launch.py` (SLAM node), `nav2_bringup.launch.py` (Nav2 core + costmap plugin + nav2 config), `full_stack.launch.py` (composes all, `mode:=mock|integration|full`), `eval_harness.launch.py` (full_stack + episode runner).

Each individual launch file must be runnable standalone against stub/mock inputs from every other workstream. Each package ships a lightweight mock/stub node so downstream packages never block on upstream being finished. **Not yet implemented** — no `ros2_ws/` packages exist yet.

---

## 8. Data / Model / Checkpoint Organization

- RELLIS-3D dataset itself is never committed to Git; downloaded via script or attached as a Kaggle Dataset. Only the frozen train/val/test sequence-split file is committed. **Refinement (ADR 0003):** the committed split is the official RELLIS-3D frame-level split (via `dataset.yaml`'s `split_files`), not a custom sequence-level split — see Section 4, deviation 3.
- Model checkpoints are not committed raw to Git. Exported models (ONNX preferred, TorchScript fallback) are attached to versioned GitHub Releases, referenced from `perception_training/exported_models/README.md` with filename convention `segmodel_v{n}_{date}.onnx`, a SHA256 checksum, and the paired `classes.yaml`.
- Every exported model records its exact training config snapshot, split file version, and run ID in the release notes for reproducibility.

**Implementation status (new):**
- **Released:** [`v1-segmodel-2026-09-19`](https://github.com/AryaShekhar13/Copernicus-Nav/releases/tag/v1-segmodel-2026-09-19) — `segmodel_v1_2026-09-19.onnx` (opset 17, legacy TorchScript-based exporter) + `classes.yaml`, SHA256 `23910cc4953e093c7d1c1b449f288b00a07934fdb07f4cf599da192de6fb7e0f`, validated against the source PyTorch model (max abs diff 4.18e-05). Release notes include the full metrics tradeoff and deployment rationale (below).
- **Model selected: the class-weighted checkpoint (epoch 8), not the unweighted baseline**, despite lower aggregate mIoU (0.331 vs 0.346) and worse ECE (0.041 vs 0.036) — the unweighted model had zero recall on `person` and `log` (safety-relevant for path planning), which weighting fixed at the cost of a small amount of accuracy on dominant classes like grass. Full numbers in the release notes and in `docs/decisions/` (recommend formalizing this as an ADR — not yet done, see Section 14).
- **Deviation from the "checkpoints not committed raw" rule, temporarily:** `best.pt` was pushed directly to a `best-checkpoint-results` Git branch, as a working safety net against Kaggle kernel/session resets during training. This branch is **not** the release artifact and is **currently stale** relative to `main` (missing `calibration.py`, `export.py`, `class_weights.npy`, `reliability_diagram.png` that exist on `main`). It should either be brought up to date, deleted now that the ONNX release exists as the real backstop, or explicitly documented as a raw-checkpoint archive only.
- **`perception_training/exported_models/README.md` referenced by this section does not exist yet** — should be added, pointing at the Release rather than duplicating the model file in-repo.
- Two new source modules exist beyond what this section originally scoped, both in `perception_training/src/`:
  - `inference.py` — frozen inference on pre-batched tensors (`load_model_for_inference`, `predict`, `run_inference_and_export`), PyTorch/checkpoint-based, offline/Kaggle-side only.
  - `pipeline.py` — full raw-image-in pipeline (`run_segmentation_pipeline`): reads a raw image file, applies the exact training-time preprocessing (`preprocessing.py`), runs MC-Dropout inference, and saves a segmentation+uncertainty pair via `export.py`. Also PyTorch/checkpoint-based, offline/Kaggle-side only — **not** the live ONNX/CPU deployment path (see Section 9's uncertainty gap).

---

## 9. Local vs. Kaggle Split

Runs locally (CPU only): Gazebo simulation (headless/software render), Nav2/costmap/planner/controller, ORB-SLAM3/RTAB-Map, terrain_eval episode generator and logging, ONNX-runtime inference of the already-trained exported model.

Runs on Kaggle (GPU): RELLIS-3D fine-tuning of the segmentation backbone, MC-Dropout/ensemble uncertainty training, batch offline inference for mIoU/precision-recall/ECE evaluation, large-batch calibration/reliability-diagram computation.

Key risk: live CPU inference during a real-time Gazebo run may not keep up with the costmap's required update rate. Mitigation: `terrain_perception` supports a `--mock`/`--replay` mode that either replays precomputed Kaggle-batch-generated frames time-aligned to a recorded bag, or runs the real ONNX model live at reduced resolution/rate, benchmarked ahead of time.

**Clarification driven by implementation (new):** this split was already correct in the original document — MC-Dropout was always scoped to Kaggle, never to live CPU inference — but it was easy to lose sight of while building the pipeline, so making it explicit here: **the live/deployed ONNX model is expected to be segmentation-only unless a live-compatible uncertainty method is added.** The document deliberately writes "MC-Dropout/ensemble," treating the two as interchangeable options for the Kaggle-side training/evaluation uncertainty method — it does not mandate MC-Dropout specifically, which leaves room for a different, ONNX-compatible method to be chosen for the deployed path without contradicting this section. See Section 13, risk #13 for the options under consideration.

---

## 10. Testing Strategy

1. Unit tests per package: `terrain_costmap` cost-function tests (gtest) independent of Nav2/Gazebo; `terrain_perception` node tests using a dummy ONNX model so no real checkpoint or GPU is needed in CI; `terrain_eval` split-script and schema-validation tests with synthetic data.
2. Contract/interface tests (`launch_testing`): bring up one package plus mocks of its direct dependencies, assert the right topics/types/rates appear. This is what CI runs on every PR.
3. Integration tests: `full_stack.launch.py mode:=integration` with the real Gazebo world and Nav2, run periodically at Phase boundaries, not on every commit.

CI runs layers 1 and 2, headless, no GPU needed since nothing GPU-dependent is exercised outside Kaggle.

**Implementation note:** none of this exists yet — no `terrain_perception` node, no CI config, no `tests/`. The real `v1-segmodel-2026-09-19.onnx` release is available now, though, so the "dummy ONNX model" in item 1 could reasonably be swapped for a tiny slice of the real one once `terrain_perception` exists, if that's preferred over a synthetic stub.

---

## 11. Logging / Evaluation / Results Storage

Every episode run writes one row to `results/runs/{run_id}/episodes.csv`, matching the document's schema exactly. Each run directory stores a config snapshot and optionally a ros2 bag (gitignored). `terrain_eval`'s aggregation notebook reads all runs, tags by system variant, and produces final comparison tables/plots. Training-side logging (W&B/TensorBoard) stays inside `perception_training/`; only final exported-model metrics get copied into release notes.

**Implementation note:** training-side metrics (mIoU, per-class IoU/precision/recall, ECE, MC-Dropout entropy) were tracked ad hoc in the Kaggle notebook and copied into the Release notes, not through W&B/TensorBoard as this section anticipates. Fine for a single-person perception workstream so far; worth adopting proper experiment tracking before more training runs happen, to avoid losing intermediate results to kernel resets (already happened once with `inference.py`/`pipeline.py` needing to be rewritten after a restart).

---

## 12. Team Workflow / Maintainability

Package-to-person mapping is 1:1 with the five workstreams. CODEOWNERS enforces review by package owner; interface doc changes need sign-off from every affected workstream. Short-lived feature branches, PRs gated by CI. ADRs capture every contract-level decision. Docker/devcontainer pins exact versions. ament_lint / black+flake8 in CI.

**Implementation note:** only `docs/decisions/0003-rellis-split-strategy.md` exists as a real ADR so far. The weighted-vs-unweighted model selection (Section 8) and the uncertainty/ONNX gap (Section 9/13) are exactly the kind of contract-level decisions this section says should get an ADR, and don't have one yet — recommend adding `0004-checkpoint-resume-strategy.md` (referenced informally above as "ADR 0004" for the best.pt/latest.pt fix) and `0005-deployed-uncertainty-method.md` once the uncertainty question is settled.

---

## 13. Risks & Integration Problems

1. ORB-SLAM3 build friction — mitigate via pinned Docker image, RTAB-Map as documented fallback.
2. Live CPU-only perception inference may not meet real-time costmap rate — needs early latency benchmark. **Still not benchmarked as of this update.**
3. Gazebo/ROS 2 version mismatch — pin in Day 0 ADR and Docker image.
4. Segmentation wire format was undecided in the source doc — resolved in Section 5, needs WS1/WS2 sign-off.
5. RELLIS-3D split reproducibility — split file generated once, frozen, versioned on change. **Resolved via ADR 0003, with the documented tradeoff of using the official (leaky) split instead of a custom sequence-level one.**
6. Kaggle session/quota limits — training must be checkpoint-resumable from the start. **Partially bit us in practice:** a kernel restart lost an in-progress, uncommitted `inference.py` before it reached Git — recommend committing working files more frequently, not just checkpoints.
7. C++ vs Python for costmap plugin — explicit team decision needed, affects WS2 schedule.
8. `/gt_gps` leakage — CI grep check plus manual review at integration boundaries.
9. Async sensor fusion staleness — acceptable staleness bound must be explicit and tested.
10. Nav2 BT/planner/controller tuning ownership — must be explicitly assigned.
11. Results schema drift — mitigated by JSON-Schema validation, only if every run uses the shared logger.
12. Repo bloat from checkpoints/bags — enforced via .gitignore and release-artifact pattern from commit #1. **Partially violated:** `best.pt` is currently on the `best-checkpoint-results` branch as a kernel-reset safety net; not committed to `main`, but also not yet cleaned up now that the ONNX release supersedes its purpose.
13. **(New) Deployed model cannot currently produce `/perception/uncertainty`.** MC-Dropout, the validated uncertainty method (ECE 0.0407, per-class entropy characterized), does not survive ONNX export — the exported graph is a single deterministic pass. Options, not yet decided:
    - **(a)** Run the PyTorch model (not ONNX) live for uncertainty, accepting the dependency/performance cost — simplest, zero new validation work, but works against risk #2's real-time CPU concern and against the ONNX-only local/Kaggle split in Section 9.
    - **(b)** Export several frozen ONNX graphs with different fixed dropout masks and average them — stays fully within ONNX Runtime, but multiplies both file size and inference cost by however many copies are exported.
    - **(c)** Keep ONNX's `Dropout` op stochastic at inference via its `training_mode` input — closest to "free," but fragile: not how `torch.onnx.export` behaves by default, and ONNX Runtime support for stochastic `training_mode` Dropout varies by version/provider; real risk of silently degrading to deterministic behavior.
    - **(d)** Replace MC-Dropout with a single-pass uncertainty proxy (e.g. max-softmax-probability or entropy of the one-pass softmax output) for the deployed model specifically. Free, trivial to add to `pipeline.py`'s ONNX equivalent, but a methodologically different signal from what was validated — the existing ECE/entropy numbers would not automatically carry over and it would need its own calibration check.
    - **(e)** Deep ensemble (3-5 independently-trained models, each exported deterministically, averaged) — generally a stronger estimator than MC-Dropout, and the architecture doc already treats "MC-Dropout/ensemble" as interchangeable (Section 9), so this isn't a deviation — but requires retraining multiple models from scratch, real time cost.
    Recommendation captured here for whoever picks this up: (a) if `terrain_perception` is allowed a PyTorch dependency at all — cheapest, reuses validated numbers exactly. If not, (d) is the pragmatic fallback, with an explicit re-validation step before trusting it. (b)/(c)/(e) are higher-cost and probably not worth it unless (a)/(d) prove insufficient.

---

## 14. Decisions to Lock Before Any Code Is Written (Day 0 checklist)

- ROS 2 distro + Gazebo version pin (proposed: Humble + Fortress) — **still proposed, not locked.**
- Robot platform reference for URDF (proposed: Jackal) — **still proposed, not locked.**
- Segmentation/uncertainty wire format: two Image topics vs custom msg (proposed: two Image topics) — **locked (Section 5).**
- Model export format + inference runtime: ONNX/onnxruntime vs TorchScript/libtorch — **locked: ONNX, validated and released.**
- Costmap plugin language: C++ vs Python prototype-only — **still open, WS2.**
- Risk-aversion parameter: name, type, default, valid range — **still open, WS2.**
- Acceptable staleness bound for perception frames used by the costmap — **still open.**
- Repo layout and package names — **`perception_training/` locked and in use; `ros2_ws/` package names still just proposed, not scaffolded.**
- Results schema finalized as JSON Schema — **still open, WS5/eval.**
- RELLIS-3D split file generated once and frozen — **locked via ADR 0003 (official split adopted, not a new custom one).**
- Checkpoint storage/versioning mechanism — **locked and working: GitHub Release + SHA256 + `classes.yaml`, demonstrated with `v1-segmodel-2026-09-19`.**
- Docker/devcontainer base image finalized — **still open.**
- Branch/PR policy and CODEOWNERS mapping confirmed — **still open; current practice is direct pushes to `main`/`best-checkpoint-results` with no PR/review step, worth tightening once more than one contributor is active.**
- **(New) Deployed uncertainty method for the live/ONNX model** — **open, see Section 13 risk #13.**
- **(New) `best-checkpoint-results` branch disposition** (update, delete, or formally document as checkpoint-only archive) — **open.**

---

This document has been updated in place to reflect Workstream 1's actual implementation and the decisions made along the way (class-weighted model selection, ONNX export/release process, the MC-Dropout/ONNX uncertainty gap). Workstreams 2-5 remain at the original proposal stage — next step for them is unchanged: scaffold the empty package structure with stub nodes/launch files once the still-open Day 0 items above are settled.
