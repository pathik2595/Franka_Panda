# Franka Panda Isaac Sim Bin Picking

Perception-driven bin picking and pick-and-place experiments for a Franka Panda robot in NVIDIA Isaac Sim. The current working demo spawns multiple USD objects in a tray, asks the operator which object to pick, segments the target from RGB-D using SAM 3, estimates a top-down grasp from the masked point cloud, and places the object into a second tray using perception-based free-space search.

## Current Demo

Run the stable persistent clutter pipeline:

```powershell
& "C:\isaacsim\python.bat" "C:\Claude\Franka_Panda\Scripts\persistent_clutter_grasp_filter.py"
```

The no-flag command currently defaults to:

- Franka Panda in `assets/scene.usd`
- persistent clutter mode
- pickable USD objects: `peach`, `rubiks_cube`, `mango`
- RGB-D capture from `/World/RGBD_Camera`
- SAM 3 prompt-based target segmentation
- mask/depth based object pose, yaw, height, and gripper-width estimation
- top-down perception grasping with Lula IK
- perception-based free-space placement in the destination tray

## What Works

- Randomized clutter spawning inside the pick tray with object yaw variation.
- Interactive object selection: for example `peach`, `cube`, or `mango`.
- RGB-D capture from the Isaac Sim camera.
- SAM 3 segmentation using object-specific prompts such as `red and yellow peach`, `rubiks cube`, and `yellow mango`.
- 3D target point-cloud extraction from the SAM mask and depth image.
- Object center, yaw, height, and close-width estimation from perception rather than ground-truth object pose.
- Top-down pick execution through Lula IK and the Panda gripper.
- Occupancy-aware placement: the destination tray is scanned with RGB-D, occupied points are filtered, and an empty placement cell is sampled.
- Debug artifacts for every trial, including RGB, SAM overlays, masks, geometry JSON, grasp candidates, and placement diagnostics.

Recent successful trials picked and placed cluttered objects including Rubik's cube, mango, and peach. Peach can roll after release because of dynamics, but the perception and placement stages still select valid targets and free placement areas.

## Project Structure

```text
Scripts/
  persistent_clutter_grasp_filter.py  active persistent clutter pick/place pipeline
  perception_pose.py                  RGB-D capture, SAM 3 masks, pose/yaw/width estimation
  perception_placement.py             destination-tray occupancy and free-cell placement
  grasp_debug.py                      grasp candidate debug writer
  measure_prim_world_bbox.py          helper for tray/prim world bounds

assets/
  scene.usd                           main Isaac Sim scene
  panda.usd                           Panda scene asset
  panda_description.yaml              Lula robot description config
  environment/                        table and tray USD assets
  objects/                            USD object assets used by the scene and trials

Trials/
  trial_001_rubiks_cube/              RGB-D, SAM, geometry, and placement results
  trial_002_mango/                    RGB-D, SAM, geometry, and placement results
  trial_003_peach/                    RGB-D, SAM, geometry, and placement results

```

## Perception Pipeline

1. Capture RGB and depth from `/World/RGBD_Camera`.
2. Run SAM 3 with a prompt selected from the target object catalog.
3. Reject masks that do not overlap the pick tray enough.
4. Project valid target depth pixels into world coordinates.
5. Estimate target center, yaw, object height, and gripper close width from the masked point cloud.
6. Build obstacle points from non-target depth above the tray floor.
7. Execute a top-down grasp using the estimated target geometry.
8. Capture the destination tray and sample a free placement location from RGB-D occupancy.

The robot chooses pick targets from the camera, SAM mask, and depth data. Isaac Sim ground-truth object poses are used only for setup and debugging.

## Debug Outputs

Each run writes trial artifacts under:

```text
Trials/trial_XXX_object_name/
```

Useful files:

- `rgb.png`: camera frame used for perception
- `sam_overlay.png`: target mask overlay
- `sam_mask.png`: binary SAM mask
- `sam_bbox.png`: mask bounding box visualization
- `perception_geometry.json`: estimated pose, yaw, width, and dimensions
- `target_points_world.npy`: target point cloud
- `obstacle_points_world.npy`: clutter/obstacle point cloud
- `grasp_candidates.json`: grasp yaw checks and selected candidate
- `perception_place.json`: placement search candidates and selected free cell

These outputs are excellent for screenshots, reports, and LinkedIn media. For this repo, the current successful trial folders are included under `Trials/` as project evidence.

## Dependencies

Expected local runtime:

- NVIDIA Isaac Sim with Python at `C:\isaacsim\python.bat`
- Isaac Sim robotics APIs, Replicator RGB-D annotators, USD/PXR bindings, and Lula IK
- Python packages used by the scripts: `numpy`, `Pillow`, `ultralytics`, `clip`, `timm`
- SAM 3 setup through Ultralytics: https://docs.ultralytics.com/models/sam-3

The SAM model/checkpoint should be installed or downloaded using the Ultralytics SAM 3 instructions.

## Next Improvements

- Add Contact-GraspNet as the next grasp-planning layer so the system can move beyond a single top-down grasp and generate ranked 6-DoF grasp candidates directly from the RGB-D/point-cloud observation.
