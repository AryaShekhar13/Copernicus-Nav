# Architecture Proposal — Uncertainty-Aware Off-Road Traversability & Navigation

Status: **Draft for review — no implementation yet.**
Scope: covers simulation stack choice, repo/package layout, module boundaries, contracts, config, data/model management, launch structure, testing, logging/eval, local-vs-Kaggle split, and team workflow for the 5 workstreams in `team_division.md`.

---

## 1. Simulation Stack

**Choice: ROS 2 Humble + Gazebo Fortress (via `ros_gz`), Ubuntu 22.04.**

- Humble is the current LTS with the longest support window and by far the most mature Nav2 integration, documentation, and community fixes — important because Nav2 costmap-plugin work (Workstream 2) and `ros2_control`/`gz_ros2_control` (Workstream 4) are exactly the areas where being on well-trodden versions saves the most debugging time.
- Fortress is the Gazebo release officially paired with Humble through `ros_gz`. Jazzy + Harmonic is newer but requires Ubuntu 24.04 and has thinner third-party package coverage (ORB-SLAM3 wrappers, Warthog/Jackal descriptions, etc.), which matters for Workstream 3 and 4.
- No local discrete GPU is fine for this pairing: Gazebo physics is CPU-bound, and sensor rendering (camera, LiDAR) can run headless via software rasterization (`LIBGL_ALWAYS_SOFTWARE=1` / `MESA_GL_VERSION_OVERRIDE`) at reduced but acceptable frame rates for development. GPU-only work (segmentation training) is fully offloaded to Kaggle — see Section 9.
- This is a recommendation, not a hard requirement of the document's contracts — the topic/message contracts in `team_division.md` are ROS 2-generic and unaffected by the Humble/Fortress choice.

**Robot platform reference:** Clearpath Jackal (not Warthog) — smaller, well-documented URDF/Xacro and `ros2_control` examples, sensor mounts easy to match to a real datasheet, and it keeps the differential-drive controller stack simple. This is Workstream 4's call to finalize but should be locked Day 0 since it drives URDF dimensions, actuator limits, and noise models.

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

---

## 3. Package Responsibilities & Dependency Graph

Data flow: terrain_description + terrain_gazebo_worlds publish sensor topics -> terrain_perception (segmentation+uncertainty) and terrain_localization (SLAM) consume camera/imu -> terrain_costmap consumes perception output + tf -> Nav2 core (terrain_nav2_config) consumes costmap + produces /cmd_vel -> terrain_eval orchestrates episodes and logs results across the whole stack.

No package other than `terrain_bringup` and `terrain_eval` is allowed to depend on more than one other workstream's package — this is the main lever for parallel development and independent testability.

---

## 4. Deviations From the Document (with justification)

1. `perception_training/` is split out of the ROS workspace entirely. Training never runs inside colcon/ROS at all — it's a separate Python project whose only deliverable crossing into ROS-land is a frozen checkpoint + `classes.yaml`.
2. `terrain_nav2_config` is split out from `terrain_costmap`, so the plugin stays a narrow, independently-buildable, independently-testable library while Nav2-wide tuning lives separately.

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

**Hard rule, enforced by tooling not just convention:** `/gt_gps` must never be subscribed to by any node under `terrain_costmap`, `terrain_nav2_config`, or `terrain_localization`'s live (non-eval) code path. `scripts/check_interfaces.py` greps all package source for `/gt_gps` subscriptions and fails CI if found outside `terrain_eval`.

---

## 6. Configuration & Parameter Management

- Topic names defined once in `docs/interfaces.md`, mirrored as ROS 2 launch arguments — never hardcoded inside node source. `terrain_bringup` is the only place that wires concrete remappings together.
- Node parameters: one YAML per node under each package's `config/`, loaded via `launch_ros`'s `parameters=[...]`.
- Episode/terrain configs: PyYAML files under `terrain_eval/config/episodes/` and `terrain_gazebo_worlds/config/terrain/`.
- Results schema: defined once as a JSON Schema in `terrain_eval/schema/results_schema.json`; the metrics logger validates each row before writing.
- Risk-aversion parameter: exposed as a normal Nav2-style dynamic parameter on the costmap plugin, default + valid range documented in the interfaces doc.

---

## 7. Launch File Organization

`terrain_bringup/launch/` contains: `sim_bringup.launch.py` (Gazebo + robot + sensors), `perception.launch.py` (real or `--mock` perception node), `localization.launch.py` (SLAM node), `nav2_bringup.launch.py` (Nav2 core + costmap plugin + nav2 config), `full_stack.launch.py` (composes all, `mode:=mock|integration|full`), `eval_harness.launch.py` (full_stack + episode runner).

Each individual launch file must be runnable standalone against stub/mock inputs from every other workstream. Each package ships a lightweight mock/stub node so downstream packages never block on upstream being finished.

---

## 8. Data / Model / Checkpoint Organization

- RELLIS-3D dataset itself is never committed to Git; downloaded via script or attached as a Kaggle Dataset. Only the frozen train/val/test sequence-split file (`perception_training/splits/split_v1.yaml`, sequence IDs not frame indices) is committed.
- Model checkpoints are not committed raw to Git. Exported models (ONNX preferred, TorchScript fallback) are attached to versioned GitHub Releases, referenced from `perception_training/exported_models/README.md` with filename convention `segmodel_v{n}_{date}.onnx`, a SHA256 checksum, and the paired `classes.yaml`.
- Every exported model records its exact training config snapshot, split file version, and run ID in the release notes for reproducibility.

---

## 9. Local vs. Kaggle Split

Runs locally (CPU only): Gazebo simulation (headless/software render), Nav2/costmap/planner/controller, ORB-SLAM3/RTAB-Map, terrain_eval episode generator and logging, ONNX-runtime inference of the already-trained exported model.

Runs on Kaggle (GPU): RELLIS-3D fine-tuning of the segmentation backbone, MC-Dropout/ensemble uncertainty training, batch offline inference for mIoU/precision-recall/ECE evaluation, large-batch calibration/reliability-diagram computation.

Key risk: live CPU inference during a real-time Gazebo run may not keep up with the costmap's required update rate. Mitigation: `terrain_perception` supports a `--mock`/`--replay` mode that either replays precomputed Kaggle-batch-generated frames time-aligned to a recorded bag, or runs the real ONNX model live at reduced resolution/rate, benchmarked ahead of time.

---

## 10. Testing Strategy

1. Unit tests per package: `terrain_costmap` cost-function tests (gtest) independent of Nav2/Gazebo; `terrain_perception` node tests using a dummy ONNX model so no real checkpoint or GPU is needed in CI; `terrain_eval` split-script and schema-validation tests with synthetic data.
2. Contract/interface tests (`launch_testing`): bring up one package plus mocks of its direct dependencies, assert the right topics/types/rates appear. This is what CI runs on every PR.
3. Integration tests: `full_stack.launch.py mode:=integration` with the real Gazebo world and Nav2, run periodically at Phase boundaries, not on every commit.

CI runs layers 1 and 2, headless, no GPU needed since nothing GPU-dependent is exercised outside Kaggle.

---

## 11. Logging / Evaluation / Results Storage

Every episode run writes one row to `results/runs/{run_id}/episodes.csv`, matching the document's schema exactly. Each run directory stores a config snapshot and optionally a ros2 bag (gitignored). `terrain_eval`'s aggregation notebook reads all runs, tags by system variant, and produces final comparison tables/plots. Training-side logging (W&B/TensorBoard) stays inside `perception_training/`; only final exported-model metrics get copied into release notes.

---

## 12. Team Workflow / Maintainability

Package-to-person mapping is 1:1 with the five workstreams. CODEOWNERS enforces review by package owner; interface doc changes need sign-off from every affected workstream. Short-lived feature branches, PRs gated by CI. ADRs capture every contract-level decision. Docker/devcontainer pins exact versions. ament_lint / black+flake8 in CI.

---

## 13. Risks & Integration Problems

1. ORB-SLAM3 build friction — mitigate via pinned Docker image, RTAB-Map as documented fallback.
2. Live CPU-only perception inference may not meet real-time costmap rate — needs early latency benchmark.
3. Gazebo/ROS 2 version mismatch — pin in Day 0 ADR and Docker image.
4. Segmentation wire format was undecided in the source doc — resolved in Section 5, needs WS1/WS2 sign-off.
5. RELLIS-3D split reproducibility — split file generated once, frozen, versioned on change.
6. Kaggle session/quota limits — training must be checkpoint-resumable from the start.
7. C++ vs Python for costmap plugin — explicit team decision needed, affects WS2 schedule.
8. `/gt_gps` leakage — CI grep check plus manual review at integration boundaries.
9. Async sensor fusion staleness — acceptable staleness bound must be explicit and tested.
10. Nav2 BT/planner/controller tuning ownership — must be explicitly assigned.
11. Results schema drift — mitigated by JSON-Schema validation, only if every run uses the shared logger.
12. Repo bloat from checkpoints/bags — enforced via .gitignore and release-artifact pattern from commit #1.

---

## 14. Decisions to Lock Before Any Code Is Written (Day 0 checklist)

- ROS 2 distro + Gazebo version pin (proposed: Humble + Fortress)
- Robot platform reference for URDF (proposed: Jackal)
- Segmentation/uncertainty wire format: two Image topics vs custom msg (proposed: two Image topics)
- Model export format + inference runtime: ONNX/onnxruntime vs TorchScript/libtorch
- Costmap plugin language: C++ vs Python prototype-only
- Risk-aversion parameter: name, type, default, valid range
- Acceptable staleness bound for perception frames used by the costmap
- Repo layout and package names
- Results schema finalized as JSON Schema
- RELLIS-3D split file generated once and frozen
- Checkpoint storage/versioning mechanism
- Docker/devcontainer base image finalized
- Branch/PR policy and CODEOWNERS mapping confirmed

---

This is the proposal for review. Nothing here has been implemented yet — next step, once the above is agreed, is scaffolding the empty package structure with stub nodes/launch files so all five workstreams can start independent development immediately.
