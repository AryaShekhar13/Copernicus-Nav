# Team Division & Task Breakdown

## Interface Contracts (Lock These on Day 0 — Before Any Independent Work Starts)

| Contract | Definition |
|---|---|
| Camera topic | `/camera/image_raw` — `sensor_msgs/Image` |
| IMU topic | `/imu` — `sensor_msgs/Imu` |
| LiDAR topic (if used) | `/scan` or `/points` — `sensor_msgs/LaserScan` or `PointCloud2` |
| Ground-truth GPS/pose (eval only) | `/gt_gps` — `sensor_msgs/NavSatFix`, or raw Gazebo ground-truth pose — **never subscribed to by planner/costmap/SLAM** |
| Command output | `/cmd_vel` — `geometry_msgs/Twist` |
| Localization output | standard `tf` + `nav_msgs/Odometry` — Nav2-native, no custom message |
| Segmentation output | fixed image size, per-pixel class index array (`.png` or `numpy` array) + a documented class-index-to-name mapping file (`classes.yaml`) |
| Uncertainty output | per-pixel float map, same resolution as segmentation, saved alongside it |
| Costmap plugin interface | conforms to Nav2 `costmap_2d` plugin API (`CostmapLayer` base class) |
| Per-episode results schema | one row per episode: `episode_id, terrain_config, success(bool), collisions(int), path_efficiency(float), completion_time(s), replans(int), min_clearance(m), loc_error(m), latency(ms)` — CSV or JSON, agreed column names |

Everyone builds against these contracts using stubs/mocks until integration day. This is what lets 5 people work independently without blocking on each other.

---

## Workstream 1 — Perception (Segmentation + Uncertainty)

**Task:** Fine-tune a lightweight segmentation model on RELLIS-3D terrain classes; add an uncertainty head; evaluate per-class IoU/precision-recall and calibration.

**Brief breakdown:**
- Download and preprocess RELLIS-3D (sequence-level train/val/test split — no frame-random shuffling)
- Fine-tune a pretrained backbone on the 20-class RELLIS-3D ontology
- Add MC-Dropout (or small ensemble) for per-pixel uncertainty
- Evaluate: mIoU, per-class IoU (hazardous classes reported separately), precision/recall, reliability diagram / ECE for calibration
- Publish a frozen inference script early (even with a dummy/random output stub) so Workstream 2 can integrate against the interface before the model is finished

**Exact tools:**
- **PyTorch** (model training/inference)
- **`segmentation_models.pytorch`** or **MMSegmentation** (segmentation backbone + head implementations — PIDNet/BiSeNet-style architectures)
- **MobileNetV3 / EfficientNet-lite** (pretrained encoder, via `timm` or `torchvision.models`)
- **RELLIS-3D devkit** (official dataset loader/annotation tools from the dataset's GitHub repo)
- **Weights & Biases** or **TensorBoard** (training/eval logging)
- **NumPy / OpenCV** (mask processing, resizing, class-index mapping)
- **scikit-learn** (`calibration_curve` for ECE/reliability diagrams)
- Compute: Google Colab / Kaggle notebooks (single GPU sufficient for a lightweight backbone)

---

## Workstream 2 — Traversability & Costmap (Core Contribution)

**Task:** Build the class+uncertainty → cost mapping function and the Nav2 costmap plugin; run ablations.

**Brief breakdown:**
- Define the traversability lookup table (per-class base cost) and the uncertainty-weighted risk adjustment function
- Implement a custom Nav2 `costmap_2d` plugin that ingests (segmentation mask, uncertainty map) and outputs a cost grid
- Add a tunable risk-aversion parameter
- Run ablation: with vs. without uncertainty term in cost function
- Develop/test offline against Workstream 1's stub output before the real model is ready

**Exact tools:**
- **ROS 2 (Humble/Jazzy)** + **Nav2** (`nav2_costmap_2d` package, plugin base classes)
- **C++** (Nav2 costmap plugins are typically C++ for real-time performance; Python via `rclpy` acceptable for prototyping)
- **colcon** (ROS 2 build system)
- **NumPy** (cost function prototyping before porting to the ROS 2 plugin)
- **Matplotlib** (visualizing cost grids during development)
- **rqt / rviz2** (visual debugging of the published costmap)

---

## Workstream 3 — Localization (VO/SLAM + Evaluation)

**Task:** Integrate a visual odometry/SLAM system, publish pose estimates, evaluate against ground truth on both RELLIS-3D (offline) and simulation (closed-loop).

**Brief breakdown:**
- Integrate ORB-SLAM3 (or equivalent) as a ROS 2 node
- Run it offline against RELLIS-3D image sequences, compare to the dataset's SLAM-derived ground-truth pose
- Compute ATE and RPE
- Later: run the same node against simulated camera/IMU topics in Gazebo, compare to `/gt_gps`/ground-truth pose
- Publish pose via standard `tf`/`nav_msgs/Odometry` so Nav2 consumes it with no custom glue code

**Exact tools:**
- **ORB-SLAM3** (or **RTAB-Map**, or **VINS-Mono** as alternatives) — open-source, ROS 2-wrapped
- **evo** (Python package specifically for computing ATE/RPE and plotting trajectory comparisons)
- **ROS 2 `tf2`** (pose/transform publishing)
- **RELLIS-3D devkit** (ground-truth pose format/loader)
- **rviz2** (trajectory visualization)

---

## Workstream 4 — Simulation Environment & Hardware Layer

**Task:** Build the Gazebo world (terrain, obstacles, materials), robot model, sensor configuration, and hardware abstraction layer.

**Brief breakdown:**
- Build/import heightmap terrain (DEM or procedural Perlin/Simplex)
- Assign per-material friction (`mu`/`mu2`) values to terrain patches (mud, trail, rock)
- Add static obstacles and 1-2 scripted dynamic actors
- Build the robot URDF/Xacro (chassis, sensor mounts matching a real platform like Warthog/Jackal)
- Configure sensor plugins (camera, IMU, LiDAR, `navsat`/GPS) with noise matched to real hardware datasheets
- Set up `ros2_control` + `gz_ros2_control` hardware interface (actuator limits, latency)
- Ensure GPS topic is isolated to `/gt_gps` only, never wired to the nav stack
- Publish this world/robot early, even with placeholder terrain, so it can accept `/cmd_vel` and publish sensor topics for others to integrate against

**Exact tools:**
- **Gazebo (Fortress or Harmonic, `gz sim`)**
- **SDF / URDF / Xacro** (robot and world description formats)
- **`gz_ros2_control`** (simulated hardware interface plugin)
- **`ros2_control`** + **`diff_drive_controller`**
- **Gazebo sensor plugins**: camera, IMU, `navsat` (GPS), LiDAR — all built-in, configured via SDF tags
- **Blender** (optional — custom obstacle/terrain mesh creation)
- **QGIS or a DEM source** (optional — real elevation data for heightmap import)

---

## Workstream 5 — Evaluation Harness & Baselines

**Task:** Build the data-split scripts, episode generator, metrics logger, and run the 3-tier baseline comparison.

**Brief breakdown:**
- Write the RELLIS-3D sequence-level train/val/test split script (enforce no frame leakage)
- Build an episode generator: randomized start/goal, terrain config selection, obstacle placement, per Workstream 4's world
- Implement metrics logging matching the agreed results schema
- Implement/run Baseline 1 (generic obstacle detection + standard planner) and Baseline 2 (segmentation + traversability, no uncertainty) alongside the full system
- Aggregate results into final comparison tables/plots

**Exact tools:**
- **Python** (`pandas` for results aggregation, `argparse`/config-driven episode generation)
- **PyYAML** (episode/terrain configuration files)
- **ROS 2 `launch`** (parameterized episode launch files — varying start/goal/terrain per run)
- **Matplotlib / Seaborn** (final result plots — success rate, collision rate, path efficiency comparisons across baselines)
- **Jupyter Notebook** (final analysis and report-figure generation)

---

## Sequencing (Independent → Convergence)

1. **Day 0:** Freeze all interface contracts above (topics, message types, file formats, results schema).
2. **Phase 1 (independent, parallel):** Each workstream builds against stubs/mocks of others' outputs.
3. **Phase 2 (pairwise integration):** Merge Workstream 1 → Workstream 2 (real segmentation into costmap); merge Workstream 3 + Workstream 4 (localization running against the live sim).
4. **Phase 3 (full convergence):** Wire everything into Nav2 end-to-end; Workstream 5 runs the full harness and produces final numbers.
