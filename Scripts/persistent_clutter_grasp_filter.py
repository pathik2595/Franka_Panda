import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp

from grasp_debug import save_grasp_candidates
from perception_placement import sample_perception_place_position as sample_perception_place_position_impl


WORKSPACE = Path(__file__).resolve().parents[1]
EXTRA_SITE_PACKAGES = WORKSPACE / ".isaac_python_packages"
if EXTRA_SITE_PACKAGES.exists():
    sys.path.append(str(EXTRA_SITE_PACKAGES))


def parse_args():
    workspace = Path(__file__).resolve().parents[1]
    asset_root = workspace / "assets"
    parser = argparse.ArgumentParser(description="Simple Lula IK pick-and-place demo for Franka Panda.")
    parser.add_argument("--usd-path", default=str(asset_root / "scene.usd"))
    parser.add_argument("--robot-prim-path", default="/World/panda/franka")
    parser.add_argument("--urdf-path", default=str(workspace / "isaac_bin_picking/src/panda_description/urdf/panda.urdf"))
    parser.add_argument("--robot-description-path", default=str(asset_root / "panda_description.yaml"))
    parser.add_argument("--ee-frame", default="grasp_center")
    parser.add_argument("--object-mode", choices=["usd", "cuboid"], default="usd")
    parser.add_argument("--object-usd-path", default=str(workspace / "assets/objects/tomatosoupcan/tomato_soup_can.usd"))
    parser.add_argument("--object-prim-path", default="/World/tomato_soup_can")
    parser.add_argument("--trial-object-source", choices=["fixed", "random-usd-list", "sequence-usd-list"], default="random-usd-list")
    parser.add_argument(
        "--trial-objects",
        nargs="+",
        default=["peach", "rubiks_cube", "mango"],
    )
    parser.add_argument(
        "--scene-objects",
        nargs="+",
        default=["peach", "rubiks_cube", "mango"],
    )
    parser.add_argument(
        "--target-object",
        choices=[
            "peach",
            "rubiks_cube",
            "mango",
            "can",
            "mug",
            "marker",
        ],
        default=None,
    )
    parser.add_argument("--persistent-clutter", action="store_true", default=True)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--pick-tray-center", nargs=3, type=float, default=[-0.15, -0.28, 0.25])
    parser.add_argument("--pick-tray-interior-size", nargs=2, type=float, default=[0.50, 0.35])
    parser.add_argument("--spawn-tray-margin", type=float, default=0.110)
    parser.add_argument("--bin-random-x-range", nargs=2, type=float, default=None)
    parser.add_argument("--bin-random-y-range", nargs=2, type=float, default=None)
    parser.add_argument("--clutter-spawn-x-range", nargs=2, type=float, default=None)
    parser.add_argument("--clutter-spawn-y-range", nargs=2, type=float, default=None)
    parser.add_argument("--clutter-min-separation", type=float, default=0.085)
    parser.add_argument("--clutter-footprint-padding", type=float, default=0.012)
    parser.add_argument("--clutter-spawn-candidates", type=int, default=80)
    parser.add_argument("--clutter-center-exclusion-radius", type=float, default=0.0)
    parser.add_argument("--drop-spawn-clutter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--drop-spawn-height", type=float, default=0.080)
    parser.add_argument("--drop-spawn-settle-steps", type=int, default=160)
    parser.add_argument("--drop-spawn-max-retries", type=int, default=20)
    parser.add_argument("--drop-spawn-tray-margin", type=float, default=0.010)
    parser.add_argument("--use-clutter-spawn-slots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clutter-zone-jitter", type=float, default=0.030)
    parser.add_argument("--close-clutter-spawn", action="store_true", default=False)
    parser.add_argument("--reject-unreachable-perception", action="store_true", default=True)
    parser.add_argument("--bin-floor-z", type=float, default=0.25)
    parser.add_argument("--spawn-clearance", type=float, default=0.002)
    parser.add_argument("--place-tray-center", nargs=3, type=float, default=[-0.20, 0.40, 0.25])
    parser.add_argument("--place-tray-interior-size", nargs=2, type=float, default=[0.50, 0.35])
    parser.add_argument("--place-tray-margin", type=float, default=0.015)
    parser.add_argument("--place-random-x-range", nargs=2, type=float, default=[-0.38, -0.08])
    parser.add_argument("--place-random-y-range", nargs=2, type=float, default=[0.27, 0.53])
    parser.add_argument("--use-place-slots", action="store_true", default=False)
    parser.add_argument("--place-slot-jitter", type=float, default=0.010)
    parser.add_argument("--place-min-separation", type=float, default=0.075)
    parser.add_argument("--use-perception-place", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--place-grid-step", type=float, default=0.015)
    parser.add_argument("--place-clearance-radius", type=float, default=0.035)
    parser.add_argument("--place-occupied-z-margin", type=float, default=0.018)
    parser.add_argument("--randomize-yaw", action="store_true", default=True)
    parser.add_argument("--object-roll", type=float, default=0.0)
    parser.add_argument("--object-pitch", type=float, default=0.0)
    parser.add_argument("--cube-position", nargs=3, type=float, default=[-0.15, -0.45, 0.305])
    parser.add_argument("--place-position", nargs=3, type=float, default=[-0.20, 0.40, 0.33])
    parser.add_argument("--object-size", nargs=3, type=float, default=[0.07, 0.035, 0.05])
    parser.add_argument("--cube-size", type=float, default=None)
    parser.add_argument("--cube-mass", type=float, default=0.02)
    parser.add_argument("--friction", type=float, default=4.0)
    parser.add_argument("--disable-finger-pad-colliders", action="store_true")
    parser.add_argument("--keep-finger-mesh-colliders", action="store_true")
    parser.add_argument("--finger-pad-size", nargs=3, type=float, default=[0.016, 0.004, 0.032])
    parser.add_argument("--finger-pad-inward-protrusion", type=float, default=0.0015)
    parser.add_argument("--finger-pad-z-offset", type=float, default=0.027)
    parser.add_argument("--grasp-source", choices=["scripted-top-down", "clutter-aware-top-down"], default="scripted-top-down")
    parser.add_argument("--pose-source", choices=["scripted", "sam3-rgbd"], default="sam3-rgbd")
    parser.add_argument("--camera-prim-path", default="/World/RGBD_Camera")
    parser.add_argument("--sam3-model-path", default=str(workspace / "sam3.1_multiplex.pt"))
    parser.add_argument("--sam-prompt", default=None)
    parser.add_argument("--perception-output-dir", default=str(workspace / "Trials"))
    parser.add_argument("--perception-width", type=int, default=640)
    parser.add_argument("--perception-height", type=int, default=480)
    parser.add_argument("--perception-warmup-frames", type=int, default=12)
    parser.add_argument("--perception-capture-retries", type=int, default=8)
    parser.add_argument("--perception-rt-subframes", type=int, default=4)
    parser.add_argument("--sam-conf", type=float, default=0.25)
    parser.add_argument("--reject-mask-outside-pick-tray", action="store_true", default=True)
    parser.add_argument("--pick-tray-mask-margin", type=float, default=0.030)
    parser.add_argument("--min-pick-tray-mask-fraction", type=float, default=0.05)
    parser.add_argument("--grasp-yaw-offset", type=float, default=0.0)
    parser.add_argument("--grasp-z-offset", type=float, default=0.0)
    parser.add_argument("--grasp-squeeze-margin", type=float, default=0.0015)
    parser.add_argument("--grasp-close-percentiles", nargs=2, type=float, default=[30.0, 70.0])
    parser.add_argument("--min-computed-gripper-width", type=float, default=0.001)
    parser.add_argument("--max-computed-gripper-width", type=float, default=0.038)
    parser.add_argument("--max-top-down-grasp-width", type=float, default=0.078)
    parser.add_argument("--max-top-down-aspect-ratio", type=float, default=3.5)
    parser.add_argument("--grasp-yaw-samples", type=int, default=18)
    parser.add_argument("--gripper-finger-length", type=float, default=0.090)
    parser.add_argument("--gripper-finger-thickness", type=float, default=0.018)
    parser.add_argument("--gripper-obstacle-z-margin", type=float, default=0.012)
    parser.add_argument("--max-gripper-collision-points", type=int, default=4)
    parser.add_argument("--min-obstacle-clearance", type=float, default=0.003)
    parser.add_argument("--max-target-finger-points", type=int, default=8)
    parser.add_argument("--allow-unsafe-grasp-fallback", action="store_true", default=False)
    parser.add_argument("--pose-sequence", choices=["center", "scripted"], default="center")
    parser.add_argument("--closed-gripper-width", type=float, default=0.012)
    parser.add_argument("--lift-success-threshold", type=float, default=0.06)
    parser.add_argument("--place-success-xy-threshold", type=float, default=0.12)
    parser.add_argument("--place-success-z-min", type=float, default=0.26)
    parser.add_argument("--approach-height", type=float, default=0.18)
    parser.add_argument("--place-approach-height", type=float, default=0.06)
    parser.add_argument("--transfer-height-margin", type=float, default=0.02)
    parser.add_argument("--tool-pitch", type=float, default=np.pi)
    parser.add_argument("--move-steps", type=int, default=100)
    parser.add_argument("--long-move-steps", type=int, default=180)
    parser.add_argument("--cartesian-segment-steps", type=int, default=8)
    parser.add_argument("--gripper-steps", type=int, default=100)
    parser.add_argument("--settle-steps", type=int, default=80)
    parser.add_argument("--home-steps", type=int, default=100)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


args = parse_args()


def tray_range(center, size, axis_index, margin=0.0):
    half_size = float(size[axis_index]) * 0.5
    center_value = float(center[axis_index])
    return [center_value - half_size + margin, center_value + half_size - margin]


def apply_tray_default_ranges():
    if args.bin_random_x_range is None:
        args.bin_random_x_range = tray_range(args.pick_tray_center, args.pick_tray_interior_size, 0)
    if args.bin_random_y_range is None:
        args.bin_random_y_range = tray_range(args.pick_tray_center, args.pick_tray_interior_size, 1)
    if args.clutter_spawn_x_range is None:
        args.clutter_spawn_x_range = tray_range(
            args.pick_tray_center,
            args.pick_tray_interior_size,
            0,
            margin=args.spawn_tray_margin,
        )
    if args.clutter_spawn_y_range is None:
        args.clutter_spawn_y_range = tray_range(
            args.pick_tray_center,
            args.pick_tray_interior_size,
            1,
            margin=args.spawn_tray_margin,
        )
    if args.place_random_x_range is None:
        args.place_random_x_range = tray_range(
            args.place_tray_center,
            args.place_tray_interior_size,
            0,
            margin=args.place_tray_margin,
        )
    if args.place_random_y_range is None:
        args.place_random_y_range = tray_range(
            args.place_tray_center,
            args.place_tray_interior_size,
            1,
            margin=args.place_tray_margin,
        )


apply_tray_default_ranges()
simulation_app = SimulationApp({"headless": args.headless})

import carb
import omni.timeline
import omni.replicator.core as rep
import omni.usd
import perception_pose
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
from PIL import Image, ImageDraw
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.world import World
try:
    from isaacsim.core.prims import SingleArticulation as Articulation
except ImportError:
    from isaacsim.core.prims import Articulation
try:
    from isaacsim.core.prims import GeometryPrim
except ImportError:
    GeometryPrim = None
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.core.utils.prims import delete_prim, is_prim_path_valid
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage, open_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver


ROBOT_PRIM_PATH = args.robot_prim_path
OBJECT_PRIM_PATH = args.object_prim_path
HIGH_FRICTION_MATERIAL_PATH = "/World/PhysicsMaterials/high_friction_grasp"
FINGER_PAD_COLLIDER_NAME = "high_friction_pad_collider"
EE_FRAME = args.ee_frame.rstrip("/").split("/")[-1]
HOME_JOINT_POSITIONS = np.array([0.012, -0.5686, 0.0, -2.8106, 0.0, 3.0367, 0.741])
ARM_JOINT_INDICES = np.arange(7)
FINGER_JOINT_INDICES = np.array([7, 8])
USD_OBJECT_CATALOG = {
    "can": {
        "usd_path": WORKSPACE / "assets/objects/tomatosoupcan/tomato_soup_can.usd",
        "prim_path": "/World/random_pick_object",
        "spawn_z": 0.305,
        "closed_width": 0.012,
        "perception_squeeze_margin": 0.010,
        "max_top_down_grasp_width": 0.090,
        "prefer_catalog_grasp": True,
        "roll": 0.0,
        "pitch": 0.0,
        "sam_prompt": "soup can",
    },
    "peach": {
        "usd_path": WORKSPACE / "assets/objects/peach/peach.usd",
        "prim_path": "/World/random_pick_object",
        "spawn_z": 0.280,
        "closed_width": 0.006,
        "perception_squeeze_margin": 0.003,
        "max_top_down_grasp_width": 0.090,
        "prefer_catalog_grasp": True,
        "roll": 0.0,
        "pitch": 0.0,
        "sam_prompt": "red and yellow peach",
        "sam_prompt_candidates": ["red and yellow peach", "peach", "small peach", "orange peach"],
    },
    "mug": {
        "usd_path": WORKSPACE / "assets/objects/mug/mug.usd",
        "prim_path": "/World/random_pick_object",
        "spawn_z": 0.315,
        "closed_width": 0.012,
        "roll": 0.0,
        "pitch": 0.0,
        "sam_prompt": "mug",
    },
    "rubiks_cube": {
        "usd_path": WORKSPACE / "assets/objects/rubiks_cube.usd",
        "prim_path": "/World/random_pick_object",
        "spawn_z": 0.285,
        "closed_width": 0.010,
        "perception_squeeze_margin": 0.022,
        "max_top_down_grasp_width": 0.105,
        "prefer_catalog_grasp": True,
        "roll": 0.0,
        "pitch": 0.0,
        "sam_prompt": "rubiks cube",
    },
    "mango": {
        "usd_path": WORKSPACE / "assets/objects/mango_059/model_mango_059_69323.usd",
        "prim_path": "/World/random_pick_object",
        "spawn_z": 0.280,
        "closed_width": 0.006,
        "perception_squeeze_margin": 0.003,
        "max_top_down_grasp_width": 0.090,
        "prefer_catalog_grasp": True,
        "roll": 0.0,
        "pitch": 0.0,
        "sam_prompt": "yellow mango",
        "sam_prompt_candidates": ["yellow mango", "mango"],
    },
    "marker": {
        "usd_path": WORKSPACE / "assets/objects/marker.usd",
        "prim_path": "/World/random_pick_object",
        "spawn_z": 0.270,
        "closed_width": 0.0048,
        "roll": 0.0,
        "pitch": 0.0,
        "grasp_yaw_offset": np.pi / 2.0,
        "skip_reason": "not reliable with the current top-down pinch grasp",
    },
}


def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)


class ReferencedUsdObject:
    def __init__(self, prim_path):
        self.prim_path = prim_path
        self.tracked_prim_path = find_tracked_object_prim_path(prim_path)

    def set_world_pose(self, position, orientation):
        stage = get_current_stage()
        prim = stage.GetPrimAtPath(self.prim_path)
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp()
        orient_op = xform.AddOrientOp()
        translate_op.Set(Gf.Vec3d(*np.asarray(position, dtype=np.float64)))
        quat = np.asarray(orientation, dtype=np.float64)
        orient_op.Set(Gf.Quatf(float(quat[0]), Gf.Vec3f(float(quat[1]), float(quat[2]), float(quat[3]))))

    def get_world_pose(self):
        stage = get_current_stage()
        prim = stage.GetPrimAtPath(self.tracked_prim_path)
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = matrix.ExtractTranslation()
        try:
            rotation = matrix.ExtractRotationQuat()
            imag = rotation.GetImaginary()
            orientation = np.array([rotation.GetReal(), imag[0], imag[1], imag[2]], dtype=np.float64)
        except Exception:
            orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return np.array([translation[0], translation[1], translation[2]], dtype=np.float64), orientation

    def apply_physics_material(self, material):
        stage = get_current_stage()
        material_prim_path = getattr(material, "prim_path", HIGH_FRICTION_MATERIAL_PATH)
        material_prim = stage.GetPrimAtPath(material_prim_path)
        if not material_prim:
            carb.log_warn(f"Physics material prim not found for USD object: {material_prim_path}")
            return

        usd_material = UsdShade.Material(material_prim)
        root_prim = stage.GetPrimAtPath(self.tracked_prim_path)
        for prim in Usd.PrimRange(root_prim):
            if prim.IsA(UsdGeom.Gprim):
                try:
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(usd_material)
                except Exception as exc:
                    carb.log_warn(f"Could not bind physics material to {prim.GetPath()}: {exc}")


def step_world(world, steps=1, render=True):
    for _ in range(steps):
        world.step(render=render)


def ensure_timeline_playing():
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()


def reset_world_and_play(world):
    world.reset()
    ensure_timeline_playing()


def get_articulation_base_pose(robot):
    if hasattr(robot, "get_world_poses"):
        positions, orientations = robot.get_world_poses()
        return positions[0], orientations[0]
    return robot.get_world_pose()


def move_to_pose(
    world,
    robot,
    ik_solver,
    articulation_ik,
    target_position,
    target_orientation,
    steps=160,
    gripper_width=None,
):
    converged = False
    gripper_action = None
    if gripper_width is not None:
        gripper_action = ArticulationAction(
            joint_positions=np.array([gripper_width, gripper_width]),
            joint_indices=FINGER_JOINT_INDICES,
        )

    for _ in range(steps):
        base_position, base_orientation = get_articulation_base_pose(robot)
        ik_solver.set_robot_base_pose(base_position, base_orientation)

        action, success = articulation_ik.compute_inverse_kinematics(
            target_position=np.asarray(target_position, dtype=np.float64),
            target_orientation=np.asarray(target_orientation, dtype=np.float64),
        )
        if success:
            converged = True
            robot.apply_action(action)
        else:
            carb.log_warn(f"IK did not converge for target {target_position}.")
        if gripper_action is not None:
            robot.apply_action(gripper_action)
        step_world(world)
    return converged


def move_cartesian_line(
    world,
    robot,
    ik_solver,
    articulation_ik,
    start_position,
    end_position,
    target_orientation,
    segments=20,
    steps_per_segment=None,
    gripper_width=None,
):
    steps_per_segment = args.cartesian_segment_steps if steps_per_segment is None else steps_per_segment
    start_position = np.asarray(start_position, dtype=np.float64)
    end_position = np.asarray(end_position, dtype=np.float64)
    converged = True
    for index in range(1, segments + 1):
        alpha = index / segments
        target_position = start_position + alpha * (end_position - start_position)
        converged = (
            move_to_pose(
                world,
                robot,
                ik_solver,
                articulation_ik,
                target_position,
                target_orientation,
                steps=steps_per_segment,
                gripper_width=gripper_width,
            )
            and converged
        )
    return converged


def command_gripper(world, robot, width, steps=80):
    action = ArticulationAction(joint_positions=np.array([width, width]), joint_indices=FINGER_JOINT_INDICES)
    for _ in range(steps):
        robot.apply_action(action)
        step_world(world)


def move_to_home(world, robot, steps=None):
    if steps is None:
        steps = args.home_steps
    arm_action = ArticulationAction(joint_positions=HOME_JOINT_POSITIONS, joint_indices=ARM_JOINT_INDICES)
    gripper_action = ArticulationAction(joint_positions=np.array([0.04, 0.04]), joint_indices=FINGER_JOINT_INDICES)
    for _ in range(steps):
        robot.apply_action(arm_action)
        robot.apply_action(gripper_action)
        step_world(world)


def reset_cube_pose(world, cube, position, orientation):
    cube.set_world_pose(position=position, orientation=orientation)
    if hasattr(cube, "set_linear_velocity"):
        cube.set_linear_velocity(np.zeros(3))
    if hasattr(cube, "set_angular_velocity"):
        cube.set_angular_velocity(np.zeros(3))
    step_world(world, 40)


def get_cube_position(cube):
    if hasattr(cube, "get_world_pose"):
        position, _ = cube.get_world_pose()
        return np.asarray(position, dtype=np.float64)
    positions, _ = cube.get_world_poses()
    return np.asarray(positions[0], dtype=np.float64)


def get_geometry_bbox(prim_path):
    stage = get_current_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim:
        raise ValueError(f"Object prim does not exist for bbox measurement: {prim_path}")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    combined = None
    geometry_paths = []

    for prim in Usd.PrimRange(root_prim):
        path = str(prim.GetPath())
        if "/Materials" in path:
            continue
        if not prim.IsA(UsdGeom.Boundable):
            continue

        box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        if box.IsEmpty():
            continue

        geometry_paths.append(path)
        if combined is None:
            combined = Gf.Range3d(box.GetMin(), box.GetMax())
        else:
            combined.UnionWith(Gf.Range3d(box.GetMin(), box.GetMax()))

    if combined is None or combined.IsEmpty():
        raise ValueError(f"No geometry bbox found under {prim_path}")

    bbox_min = combined.GetMin()
    bbox_max = combined.GetMax()
    bbox_size = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) * 0.5
    return {
        "min": np.array([bbox_min[0], bbox_min[1], bbox_min[2]], dtype=np.float64),
        "max": np.array([bbox_max[0], bbox_max[1], bbox_max[2]], dtype=np.float64),
        "center": np.array([bbox_center[0], bbox_center[1], bbox_center[2]], dtype=np.float64),
        "size": np.array([bbox_size[0], bbox_size[1], bbox_size[2]], dtype=np.float64),
        "geometry_paths": geometry_paths,
    }


def find_tracked_object_prim_path(prim_path):
    stage = get_current_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim:
        raise ValueError(f"Object prim does not exist: {prim_path}")

    rigid_body_prims = [prim for prim in Usd.PrimRange(root_prim) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    if rigid_body_prims:
        tracked_prim_path = str(rigid_body_prims[0].GetPath())
        print(f"Tracking existing rigid body prim: {tracked_prim_path}")
        return tracked_prim_path

    print(f"No rigid body child found under {prim_path}; tracking reference root.")
    return prim_path


def create_pick_object(world, object_size, object_spec=None, position=None, orientation=None):
    object_spec = object_spec or {}
    prim_path = object_spec.get("prim_path", OBJECT_PRIM_PATH)
    position = np.array(args.cube_position if position is None else position, dtype=np.float64)
    orientation = euler_angles_to_quats(np.array([args.object_roll, args.object_pitch, 0.0])) if orientation is None else orientation

    if args.object_mode == "cuboid":
        return world.scene.add(
            DynamicCuboid(
                prim_path=prim_path,
                name="pick_object",
                position=position,
                scale=object_size,
                color=np.array([0.1, 0.45, 0.95]),
                mass=args.cube_mass,
            )
        )

    usd_path = str(object_spec.get("usd_path", args.object_usd_path))
    require_file(usd_path)
    print(f"Loading pick object USD: {usd_path}")
    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    print(f"Referenced pick object at {prim_path}")
    pick_object = ReferencedUsdObject(prim_path)
    print(f"Tracking pick object pose at {pick_object.tracked_prim_path}")
    pick_object.set_world_pose(
        position=position,
        orientation=orientation,
    )
    print(f"Placed pick object at {position.tolist()}")
    return pick_object


def get_scripted_object_pose(trial_index, base_position):
    if args.pose_sequence == "center":
        return {
            "position": base_position,
            "orientation": euler_angles_to_quats(np.array([args.object_roll, args.object_pitch, 0.0])),
            "yaw": 0.0,
        }

    offsets = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.06, 0.04, 0.0]),
        np.array([-0.05, 0.07, 0.0]),
        np.array([0.08, -0.03, 0.0]),
        np.array([-0.07, -0.04, 0.0]),
    ]
    yaws = [0.0, np.pi / 6.0, -np.pi / 4.0, np.pi / 3.0, -np.pi / 6.0]
    offset = offsets[trial_index % len(offsets)]
    yaw = yaws[trial_index % len(yaws)]
    position = base_position + offset
    orientation = euler_angles_to_quats(np.array([args.object_roll, args.object_pitch, yaw]))
    return {
        "position": position,
        "orientation": orientation,
        "yaw": yaw,
    }


def validate_range(name, values):
    if values[0] > values[1]:
        raise ValueError(f"{name} min must be <= max, got {values}")


def sample_random_pose(rng, object_spec):
    validate_range("--bin-random-x-range", args.bin_random_x_range)
    validate_range("--bin-random-y-range", args.bin_random_y_range)
    x = rng.uniform(args.bin_random_x_range[0], args.bin_random_x_range[1])
    y = rng.uniform(args.bin_random_y_range[0], args.bin_random_y_range[1])
    z = float(object_spec.get("spawn_z", args.cube_position[2]))
    yaw = rng.uniform(-np.pi, np.pi) if args.randomize_yaw else 0.0
    orientation = euler_angles_to_quats(np.array([object_spec.get("roll", 0.0), object_spec.get("pitch", 0.0), yaw]))
    return {
        "position": np.array([x, y, z], dtype=np.float64),
        "orientation": orientation,
        "yaw": yaw,
    }


def sample_random_place_position(rng):
    validate_range("--place-random-x-range", args.place_random_x_range)
    validate_range("--place-random-y-range", args.place_random_y_range)
    return np.array(
        [
            rng.uniform(args.place_random_x_range[0], args.place_random_x_range[1]),
            rng.uniform(args.place_random_y_range[0], args.place_random_y_range[1]),
            args.place_position[2],
        ],
        dtype=np.float64,
    )


def analyze_top_down_grasp(bbox, object_spec=None):
    object_spec = object_spec or {}
    size = bbox["size"]
    x_size = float(abs(size[0]))
    y_size = float(abs(size[1]))
    z_size = float(abs(size[2]))
    horizontal_dims = sorted([x_size, y_size])
    grasp_width = horizontal_dims[0]
    long_width = horizontal_dims[1]
    aspect_ratio = long_width / max(grasp_width, 1e-6)

    if grasp_width <= 0.0 or z_size <= 0.0:
        return {"feasible": False, "reason": f"invalid bbox size {size.tolist()}", "bbox_size": size}
    max_grasp_width = object_spec.get("max_top_down_grasp_width", args.max_top_down_grasp_width)
    if grasp_width > max_grasp_width:
        return {
            "feasible": False,
            "reason": f"top-down grasp width {grasp_width:.4f} exceeds limit {max_grasp_width:.4f}",
            "bbox_size": size,
        }
    if aspect_ratio > args.max_top_down_aspect_ratio:
        return {
            "feasible": False,
            "reason": f"horizontal aspect ratio {aspect_ratio:.2f} exceeds limit {args.max_top_down_aspect_ratio:.2f}",
            "bbox_size": size,
        }

    computed_closed_width = grasp_width / 2.0 - args.grasp_squeeze_margin
    prefer_catalog_grasp = object_spec.get("prefer_catalog_grasp", False)
    if prefer_catalog_grasp and "closed_width" in object_spec:
        closed_width = float(object_spec["closed_width"])
        width_source = "catalog"
    else:
        closed_width = computed_closed_width
        width_source = "bbox"
    if closed_width < args.min_computed_gripper_width:
        return {
            "feasible": False,
            "reason": (
                f"computed per-finger width {closed_width:.4f} is below "
                f"{args.min_computed_gripper_width:.4f}"
            ),
            "bbox_size": size,
        }
    if closed_width > args.max_computed_gripper_width:
        return {
            "feasible": False,
            "reason": (
                f"computed per-finger width {closed_width:.4f} exceeds "
                f"{args.max_computed_gripper_width:.4f}"
            ),
            "bbox_size": size,
        }

    grasp_yaw_offset = np.pi / 2.0 if y_size < x_size else 0.0
    return {
        "feasible": True,
        "reason": "ok",
        "bbox_size": size,
        "bbox_center": bbox["center"],
        "grasp_width": grasp_width,
        "aspect_ratio": aspect_ratio,
        "closed_width": float(np.clip(closed_width, 0.0, 0.04)),
        "computed_closed_width": float(np.clip(computed_closed_width, 0.0, 0.04)),
        "width_source": width_source,
        "grasp_yaw_offset": grasp_yaw_offset,
    }


def print_object_analysis(object_name, analysis):
    status = "feasible" if analysis["feasible"] else "skipped"
    bbox_size = analysis.get("bbox_size")
    bbox_text = bbox_size.tolist() if bbox_size is not None else "<none>"
    print(f"Object analysis for {object_name}: {status}")
    print(f"  bbox_size={bbox_text}")
    if analysis["feasible"]:
        print(
            f"  grasp_width={analysis['grasp_width']:.4f}, "
            f"closed_width={analysis['closed_width']:.4f}, "
            f"width_source={analysis.get('width_source', 'bbox')}, "
            f"grasp_yaw_offset={analysis['grasp_yaw_offset']:.3f}, "
            f"aspect_ratio={analysis['aspect_ratio']:.2f}"
        )
    else:
        print(f"  reason={analysis['reason']}")


def find_camera_prim(stage, camera_prim_path):
    prim = stage.GetPrimAtPath(camera_prim_path)
    if prim and prim.IsA(UsdGeom.Camera):
        return prim

    target_name = camera_prim_path.rstrip("/").split("/")[-1]
    for candidate in stage.Traverse():
        if candidate.IsA(UsdGeom.Camera) and candidate.GetName() == target_name:
            return candidate

    cameras = [str(candidate.GetPath()) for candidate in stage.Traverse() if candidate.IsA(UsdGeom.Camera)]
    raise ValueError(f"Camera not found at {camera_prim_path!r}. Available cameras: {cameras}")


def visible_rgb_stats(rgb):
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[2] >= 3:
        rgb = rgb[:, :, :3]
    if not rgb.size:
        return 0.0, 0, 0, 0.0
    return float(np.mean(rgb)), int(np.min(rgb)), int(np.max(rgb)), float(np.std(rgb))


def is_valid_rgb_frame(rgb):
    return perception_pose.is_valid_rgb_frame(rgb)


def save_rgb_png(rgb, output_path):
    perception_pose.save_rgb_png(rgb, output_path)


def compute_mask_bbox(mask):
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size == 0:
        return None
    return [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]


def bbox_center(bbox):
    return np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float64)


def bbox_is_in_pick_tray(bbox, image_shape):
    return perception_pose.bbox_is_in_pick_tray(bbox, image_shape)


def save_sam_debug_images(rgb, mask, output_dir):
    return perception_pose.save_sam_debug_images(rgb, mask, output_dir)


def create_perception_capture(stage):
    return perception_pose.create_perception_capture(args, stage, find_camera_prim)


def capture_rgbd_for_perception(world, perception_capture):
    return perception_pose.capture_rgbd_for_perception(args, world, perception_capture, step_world)


def make_sam3_predictor():
    return perception_pose.make_sam3_predictor(args, EXTRA_SITE_PACKAGES)


def extract_sam_masks(results):
    if not results or results[0].masks is None or results[0].masks.data is None:
        return []
    masks = results[0].masks.data
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    return [np.asarray(mask, dtype=bool) for mask in masks]


def run_sam3_mask(predictor, rgb_path, prompt):
    return perception_pose.run_sam3_mask(predictor, rgb_path, prompt)


def resize_mask_to_depth(mask, depth_shape):
    return perception_pose.resize_mask_to_depth(mask, depth_shape)


def project_depth_mask_to_world(depth, valid_mask, camera_prim):
    return perception_pose.project_depth_mask_to_world(depth, valid_mask, camera_prim)


def extract_obstacle_points(mask, depth, camera_prim):
    return perception_pose.extract_obstacle_points(args, mask, depth, camera_prim)


def estimate_object_geometry_from_mask_depth(mask, depth, camera_prim, object_spec):
    return perception_pose.estimate_object_geometry_from_mask_depth(args, mask, depth, camera_prim, object_spec)


def update_pose_from_sam3_rgbd(world, perception_capture, sam_predictor, object_pose, object_spec, trial_index):
    return perception_pose.update_pose_from_sam3_rgbd(
        args,
        world,
        perception_capture,
        sam_predictor,
        object_pose,
        object_spec,
        trial_index,
        step_world,
        euler_angles_to_quats,
    )


def perception_pose_inside_pick_bounds(object_pose):
    x_min, x_max = args.bin_random_x_range
    y_min, y_max = args.bin_random_y_range
    px, py = np.asarray(object_pose["position"], dtype=np.float64)[:2]
    return (x_min <= px <= x_max) and (y_min <= py <= y_max)


def validate_trial_object_names():
    requested = list(args.trial_objects)
    if args.scene_objects:
        requested.extend(args.scene_objects)
    if args.target_object:
        requested.append(args.target_object)
    unknown = [name for name in requested if name not in USD_OBJECT_CATALOG]
    if unknown:
        raise ValueError(f"Unknown --trial-objects entries: {unknown}. Valid names: {sorted(USD_OBJECT_CATALOG)}")


def infer_target_object_from_prompt(prompt):
    if not prompt:
        return None
    text = prompt.lower().replace("_", " ")
    if "rubik" in text or "cube" in text:
        return "rubiks_cube"
    if "mango" in text:
        return "mango"
    if "peach" in text:
        return "peach"
    if "mug" in text:
        return "mug"
    if "can" in text or "cylinder" in text:
        return "can"
    return None


def default_prompt_for_object(object_name):
    return USD_OBJECT_CATALOG[object_name].get("sam_prompt", object_name.replace("_", " "))


def ask_target_for_scene(scene_object_names, picked_objects=None):
    picked_objects = picked_objects or set()
    choices = ", ".join(scene_object_names)
    print("")
    print(f"Objects in tray: {choices}")
    print("Type which object to pick, for example: peach, cube, or mango")
    while True:
        answer = input("Pick object: ").strip().lower().replace(" ", "_")
        if answer in picked_objects:
            print(f"{answer} was already placed. Pick another object.")
            continue
        if answer in scene_object_names:
            return answer, default_prompt_for_object(answer)
        inferred = infer_target_object_from_prompt(answer)
        if inferred in scene_object_names:
            if inferred in picked_objects:
                print(f"{inferred} was already placed. Pick another object.")
                continue
            return inferred, default_prompt_for_object(inferred)
        print(f"Please enter one of: {choices}")


def build_trial_object_sequence(rng):
    validate_trial_object_names()
    if args.trial_object_source == "fixed":
        return ["fixed"] * args.trials

    supported_objects = []
    for object_name in args.trial_objects:
        skip_reason = USD_OBJECT_CATALOG[object_name].get("skip_reason")
        if skip_reason:
            print(f"Skipping {object_name}: {skip_reason}")
        else:
            supported_objects.append(object_name)
    if not supported_objects:
        raise ValueError("No supported objects remain after filtering --trial-objects.")

    if args.trial_object_source == "sequence-usd-list":
        return [supported_objects[index % len(supported_objects)] for index in range(args.trials)]

    sequence = []
    while len(sequence) < args.trials:
        cycle = list(supported_objects)
        rng.shuffle(cycle)
        sequence.extend(cycle)
    return sequence[: args.trials]


def resolve_trial_object_spec(object_name, prim_path=None):
    if args.trial_object_source == "fixed":
        return {
            "name": "fixed",
            "usd_path": Path(args.object_usd_path),
            "prim_path": prim_path or OBJECT_PRIM_PATH,
            "spawn_z": args.cube_position[2],
            "closed_width": args.closed_gripper_width,
            "roll": args.object_roll,
            "pitch": args.object_pitch,
        }

    spec = dict(USD_OBJECT_CATALOG[object_name])
    spec["name"] = object_name
    if prim_path:
        spec["prim_path"] = prim_path
    if args.sam_prompt:
        spec["sam_prompt"] = args.sam_prompt
    return spec


def scene_prim_path(object_name):
    return f"/World/random_pick_object_{object_name}"


def cleanup_scene_objects(object_names):
    for object_name in object_names:
        prim_path = scene_prim_path(object_name)
        if is_prim_path_valid(prim_path):
            delete_prim(prim_path)


def filter_persistent_clutter_objects(object_names):
    filtered = [name for name in object_names if name != "can"]
    if len(filtered) != len(object_names):
        print(f"Soup can is disabled for persistent clutter; using: {', '.join(filtered)}")
    return filtered


def sample_non_overlapping_place_position(rng, occupied_positions, min_distance=0.12):
    for _ in range(100):
        candidate = sample_random_place_position(rng)
        if all(np.linalg.norm(candidate[:2] - occupied[:2]) >= min_distance for occupied in occupied_positions):
            return candidate
    return sample_random_place_position(rng)


def place_slots():
    z = float(args.place_position[2])
    return [
        np.array([-0.235, 0.365, z], dtype=np.float64),
        np.array([-0.165, 0.365, z], dtype=np.float64),
        np.array([-0.235, 0.435, z], dtype=np.float64),
        np.array([-0.165, 0.435, z], dtype=np.float64),
        np.array([-0.200, 0.400, z], dtype=np.float64),
    ]


def sample_persistent_place_position(rng, occupied_positions):
    if args.use_place_slots:
        slots = place_slots()
        for slot_index in rng.permutation(len(slots)).tolist():
            slot = slots[slot_index]
            if any(np.linalg.norm(slot[:2] - occupied[:2]) < args.place_min_separation for occupied in occupied_positions):
                continue
            jitter_xy = rng.uniform(-args.place_slot_jitter, args.place_slot_jitter, size=2)
            candidate = slot.copy()
            candidate[:2] += jitter_xy
            return candidate
        print("Warning: all place slots are occupied; falling back to random non-overlap sampling.")
    return sample_non_overlapping_place_position(rng, occupied_positions, min_distance=args.place_min_separation)


def sample_perception_place_position(world, perception_capture, rng, occupied_positions, output_dir=None):
    return sample_perception_place_position_impl(
        args,
        world,
        perception_capture,
        rng,
        occupied_positions,
        capture_rgbd_for_perception,
        project_depth_mask_to_world,
        sample_persistent_place_position,
        output_dir=output_dir,
    )


def sample_clutter_pose(rng, object_spec):
    validate_range("--clutter-spawn-x-range", args.clutter_spawn_x_range)
    validate_range("--clutter-spawn-y-range", args.clutter_spawn_y_range)
    x = rng.uniform(args.clutter_spawn_x_range[0], args.clutter_spawn_x_range[1])
    y = rng.uniform(args.clutter_spawn_y_range[0], args.clutter_spawn_y_range[1])
    z = float(object_spec.get("spawn_z", args.cube_position[2]))
    yaw = rng.uniform(-np.pi, np.pi) if args.randomize_yaw else 0.0
    orientation = euler_angles_to_quats(np.array([object_spec.get("roll", 0.0), object_spec.get("pitch", 0.0), yaw]))
    return {
        "position": np.array([x, y, z], dtype=np.float64),
        "orientation": orientation,
        "yaw": yaw,
    }


def sample_spread_clutter_pose(rng, object_spec, occupied_footprints):
    best_pose = None
    best_score = -float("inf")
    candidate_count = max(1, int(args.clutter_spawn_candidates))
    for _ in range(candidate_count):
        pose = sample_clutter_pose(rng, object_spec)
        xy = pose["position"][:2]
        if not is_far_enough_from_tray_center(
            xy,
            args.clutter_spawn_x_range,
            args.clutter_spawn_y_range,
            args.clutter_center_exclusion_radius,
        ):
            continue
        if occupied_footprints:
            score = min(
                float(np.linalg.norm(xy - occupied["xy"]) - occupied["radius"])
                for occupied in occupied_footprints
            )
        else:
            center = tray_center_from_ranges(args.clutter_spawn_x_range, args.clutter_spawn_y_range)
            score = float(np.linalg.norm(xy - center))
        if score > best_score:
            best_score = score
            best_pose = pose
    return best_pose if best_pose is not None else sample_clutter_pose(rng, object_spec)


def tray_center_from_ranges(x_range, y_range):
    return np.array(
        [
            0.5 * (float(x_range[0]) + float(x_range[1])),
            0.5 * (float(y_range[0]) + float(y_range[1])),
        ],
        dtype=np.float64,
    )


def is_far_enough_from_tray_center(xy, x_range, y_range, radius):
    if radius <= 0.0:
        return True
    center = tray_center_from_ranges(x_range, y_range)
    return float(np.linalg.norm(np.asarray(xy, dtype=np.float64) - center)) >= radius


def is_xy_inside_range(xy, x_range, y_range, margin=0.0):
    xy = np.asarray(xy, dtype=np.float64)
    return (
        float(x_range[0]) + margin <= xy[0] <= float(x_range[1]) - margin
        and float(y_range[0]) + margin <= xy[1] <= float(y_range[1]) - margin
    )


def settled_pose_record(cube):
    position, orientation = cube.get_world_pose()
    return {
        "position": np.asarray(position, dtype=np.float64),
        "orientation": np.asarray(orientation, dtype=np.float64),
        "yaw": 0.0,
    }


def settled_xy_is_valid(final_xy, sampled_positions):
    in_tray = is_xy_inside_range(
        final_xy,
        args.clutter_spawn_x_range,
        args.clutter_spawn_y_range,
        margin=args.drop_spawn_tray_margin,
    )
    separated = all(
        np.linalg.norm(np.asarray(final_xy, dtype=np.float64) - existing_xy) >= args.clutter_min_separation
        for existing_xy in sampled_positions
    )
    return in_tray and separated


def footprint_radius_from_bbox(bbox):
    size = np.asarray(bbox["size"], dtype=np.float64)
    return float(0.5 * max(abs(size[0]), abs(size[1])) + args.clutter_footprint_padding)


def settled_footprint_record(scene_spec):
    bbox = get_geometry_bbox(scene_spec["prim_path"])
    return {"radius": footprint_radius_from_bbox(bbox), "bbox_size": bbox["size"]}


def settled_xy_validation(final_xy, footprint_radius, occupied_footprints):
    in_tray = is_xy_inside_range(
        final_xy,
        args.clutter_spawn_x_range,
        args.clutter_spawn_y_range,
        margin=args.drop_spawn_tray_margin,
    )
    if not in_tray:
        return False, "outside tray"

    final_xy = np.asarray(final_xy, dtype=np.float64)
    for occupied in occupied_footprints:
        distance = float(np.linalg.norm(final_xy - occupied["xy"]))
        required = max(
            args.clutter_min_separation,
            float(footprint_radius + occupied["radius"]),
        )
        if distance < required:
            return False, f"too close: distance={distance:.3f}, required={required:.3f}"
    return True, "ok"


def clutter_spawn_slots():
    if args.close_clutter_spawn:
        return [
            np.array([-0.160, -0.286], dtype=np.float64),
            np.array([-0.108, -0.286], dtype=np.float64),
            np.array([-0.134, -0.244], dtype=np.float64),
        ]
    x_min, x_max = args.clutter_spawn_x_range
    y_min, y_max = args.clutter_spawn_y_range
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    pad_x = 0.025
    pad_y = 0.025
    return [
        np.array([x_min + pad_x, y_min + pad_y], dtype=np.float64),
        np.array([x_max - pad_x, y_min + pad_y], dtype=np.float64),
        np.array([x_min + pad_x, y_max - pad_y], dtype=np.float64),
        np.array([x_max - pad_x, y_max - pad_y], dtype=np.float64),
        np.array([center_x, center_y], dtype=np.float64),
    ]


def sample_clutter_slot_pose(rng, object_spec, slot_xy):
    jitter = rng.uniform(-args.clutter_zone_jitter, args.clutter_zone_jitter, size=2)
    xy = np.asarray(slot_xy, dtype=np.float64) + jitter
    z = float(object_spec.get("spawn_z", args.cube_position[2]))
    yaw = rng.uniform(-np.pi, np.pi) if args.randomize_yaw else 0.0
    orientation = euler_angles_to_quats(np.array([object_spec.get("roll", 0.0), object_spec.get("pitch", 0.0), yaw]))
    return {
        "position": np.array([xy[0], xy[1], z], dtype=np.float64),
        "orientation": orientation,
        "yaw": yaw,
    }


def create_drop_spawn_object(world, object_size, scene_spec, scene_pose):
    drop_pose = dict(scene_pose)
    drop_pose["position"] = np.asarray(scene_pose["position"], dtype=np.float64).copy()
    drop_pose["position"][2] = args.bin_floor_z + args.drop_spawn_height
    return create_pick_object(
        world,
        object_size,
        object_spec=scene_spec,
        position=drop_pose["position"],
        orientation=drop_pose["orientation"],
    )


def spawn_clutter_scene(world, rng, object_size, scene_object_names):
    cleanup_scene_objects(scene_object_names)
    step_world(world, 10)
    records = {}
    sampled_positions = []
    occupied_footprints = []
    slots = clutter_spawn_slots()
    slot_order = rng.permutation(len(slots)).tolist()

    for scene_index, scene_object_name in enumerate(scene_object_names):
        scene_spec = resolve_trial_object_spec(
            scene_object_name,
            prim_path=scene_prim_path(scene_object_name),
        )
        if args.use_clutter_spawn_slots and scene_index < len(slot_order):
            scene_pose = sample_clutter_slot_pose(rng, scene_spec, slots[slot_order[scene_index]])
        else:
            scene_pose = sample_spread_clutter_pose(rng, scene_spec, occupied_footprints)
            for _ in range(200):
                xy = scene_pose["position"][:2]
                separated = all(np.linalg.norm(xy - occupied["xy"]) >= args.clutter_min_separation for occupied in occupied_footprints)
                away_from_center = is_far_enough_from_tray_center(
                    xy,
                    args.clutter_spawn_x_range,
                    args.clutter_spawn_y_range,
                    args.clutter_center_exclusion_radius,
                )
                if separated and away_from_center:
                    break
                scene_pose = sample_spread_clutter_pose(rng, scene_spec, occupied_footprints)
            else:
                print(
                    "Warning: could not satisfy clutter separation for "
                    f"{scene_object_name}; using last sampled pose."
                )
        if args.drop_spawn_clutter:
            scene_cube = None
            accepted_pose = None
            for drop_attempt in range(args.drop_spawn_max_retries):
                if scene_cube is not None and is_prim_path_valid(scene_spec["prim_path"]):
                    delete_prim(scene_spec["prim_path"])
                    step_world(world, 4)
                scene_cube = create_drop_spawn_object(world, object_size, scene_spec, scene_pose)
                step_world(world, args.drop_spawn_settle_steps, render=True)
                accepted_pose = settled_pose_record(scene_cube)
                settled_xy = accepted_pose["position"][:2]
                footprint = settled_footprint_record(scene_spec)
                is_valid, invalid_reason = settled_xy_validation(
                    settled_xy,
                    footprint["radius"],
                    occupied_footprints,
                )
                if is_valid:
                    break
                print(
                    f"  retry drop spawn {scene_object_name} {drop_attempt + 1}/{args.drop_spawn_max_retries}: "
                    f"settled_xy={np.round(settled_xy, 4).tolist()} {invalid_reason}"
                )
                scene_pose = sample_spread_clutter_pose(rng, scene_spec, occupied_footprints)
            else:
                print(f"Warning: accepting last settled pose for {scene_object_name} after drop retries.")
                footprint = settled_footprint_record(scene_spec)
            scene_pose = accepted_pose
            print(
                f"Drop spawn {scene_object_name}: settled_xy={np.round(scene_pose['position'][:2], 4).tolist()}, "
                f"z={float(scene_pose['position'][2]):.4f}, footprint_radius={footprint['radius']:.3f}"
            )
        else:
            print(f"Spawn {scene_object_name}: xy={np.round(scene_pose['position'][:2], 4).tolist()}")
            scene_cube = create_pick_object(
                world,
                object_size,
                object_spec=scene_spec,
                position=scene_pose["position"],
                orientation=scene_pose["orientation"],
            )
            scene_bbox = get_geometry_bbox(scene_spec["prim_path"])
            floor_delta = args.bin_floor_z + args.spawn_clearance - float(scene_bbox["min"][2])
            if abs(floor_delta) > 1e-5:
                scene_pose["position"] = scene_pose["position"] + np.array([0.0, 0.0, floor_delta])
                reset_cube_pose(world, scene_cube, scene_pose["position"], scene_pose["orientation"])
            footprint = settled_footprint_record(scene_spec)

        sampled_positions.append(scene_pose["position"][:2].copy())
        occupied_footprints.append(
            {
                "name": scene_object_name,
                "xy": scene_pose["position"][:2].copy(),
                "radius": footprint["radius"],
            }
        )

        records[scene_object_name] = {
            "spec": scene_spec,
            "pose": scene_pose,
            "cube": scene_cube,
        }

    if args.close_clutter_spawn:
        print("Close clutter spawn enabled: objects are intentionally near each other for grasp-filter testing.")

    reset_world_and_play(world)
    for record in records.values():
        reset_cube_pose(world, record["cube"], record["pose"]["position"], record["pose"]["orientation"])
    step_world(world, args.settle_steps, render=True)
    return records


def run_persistent_clutter_loop(
    world,
    robot,
    ik_solver,
    articulation_ik,
    perception_capture,
    sam_predictor,
    rng,
    object_size,
    scene_object_names,
):
    records = spawn_clutter_scene(world, rng, object_size, scene_object_names)
    picked_objects = set()
    occupied_place_positions = []
    trial_results = []
    attempt_index = 0

    print("Persistent clutter mode: successful objects stay in the place tray.")
    while len(picked_objects) < len(scene_object_names):
        target_name, prompt = ask_target_for_scene(scene_object_names, picked_objects=picked_objects)
        record = records[target_name]
        object_spec = record["spec"]
        object_spec["sam_prompt"] = prompt
        object_spec.pop("closed_width", None)
        object_spec["grasp_yaw_offset"] = 0.0
        object_pose = record["pose"]
        attempt_index += 1

        print(
            f"Persistent pick {attempt_index}: target={target_name}, "
            f"prompt={prompt!r}"
        )
        step_world(world, args.settle_steps, render=True)
        try:
            object_pose = update_pose_from_sam3_rgbd(
                world,
                perception_capture,
                sam_predictor,
                object_pose,
                object_spec,
                attempt_index - 1,
            )
        except RuntimeError as exc:
            print(f"Perception failed for {target_name}: {exc}")
            print("Try another object, or retry this object with a different prompt later.")
            continue
        if args.use_perception_place:
            place_position = sample_perception_place_position(
                world,
                perception_capture,
                rng,
                occupied_place_positions,
                output_dir=object_spec.get("perception_output_dir"),
            )
        else:
            place_position = sample_persistent_place_position(rng, occupied_place_positions)
        print(f"  place_position={place_position.tolist()}")
        result = run_pick_place_trial(
            world,
            robot,
            record["cube"],
            ik_solver,
            articulation_ik,
            object_pose["position"],
            object_pose["orientation"],
            object_pose["yaw"],
            place_position,
            object_spec=object_spec,
            reset_before_pick=False,
        )
        result["trial"] = attempt_index
        trial_results.append(result)

        if result["success"]:
            picked_objects.add(target_name)
            occupied_place_positions.append(place_position)
            record["pose"] = {
                "position": result.get("final_position", place_position),
                "orientation": object_pose["orientation"],
                "yaw": object_pose["yaw"],
            }
            print(f"{target_name} placed. Pick another object.")
        else:
            print(f"{target_name} pick/place failed. Try again or choose another object.")

    print_trial_summary(trial_results)
    return trial_results


def compute_top_down_grasp_candidate(object_pose):
    object_position = object_pose["position"]
    grasp_yaw = object_pose["yaw"] + args.grasp_yaw_offset + object_pose.get("grasp_yaw_offset", 0.0)
    position = np.asarray(object_position, dtype=np.float64)
    position = position + np.array([0.0, 0.0, args.grasp_z_offset])
    orientation = euler_angles_to_quats(np.array([0.0, args.tool_pitch, grasp_yaw]))
    width = object_pose.get("closed_width", args.closed_gripper_width)
    return {
        "position": position,
        "orientation": orientation,
        "width": width,
        "score": 1.0,
        "source": "perception-top-down" if object_pose.get("source") == "sam3-rgbd" else "scripted-top-down",
        "yaw": grasp_yaw,
    }


def normalize_grasp_yaw(yaw):
    return float(yaw % np.pi)


def gripper_finger_occupancy(points, center_xy, yaw, opening_width):
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return 0, float("inf")

    points_xy = points[:, :2]
    shifted = points_xy - np.asarray(center_xy, dtype=np.float64)
    c = np.cos(-yaw)
    s = np.sin(-yaw)
    local_x = c * shifted[:, 0] - s * shifted[:, 1]
    local_y = s * shifted[:, 0] + c * shifted[:, 1]

    half_length = args.gripper_finger_length * 0.5
    half_opening = opening_width * 0.5
    thickness = args.gripper_finger_thickness
    in_length = np.abs(local_x) <= half_length
    upper_finger = in_length & (local_y >= half_opening) & (local_y <= half_opening + thickness)
    lower_finger = in_length & (local_y <= -half_opening) & (local_y >= -(half_opening + thickness))
    collision_mask = upper_finger | lower_finger

    finger_distance = np.minimum(
        np.abs(local_y - half_opening),
        np.abs(local_y + half_opening),
    )
    near_length = np.abs(local_x) <= half_length + thickness
    clearance = float(np.min(finger_distance[near_length])) if np.any(near_length) else float("inf")
    return int(np.count_nonzero(collision_mask)), clearance


def build_grasp_yaw_candidates(object_pose):
    base_yaw = normalize_grasp_yaw(float(object_pose.get("yaw", 0.0)))
    offsets = np.deg2rad(np.array([0.0, 90.0, -15.0, 15.0, 75.0, 105.0, -30.0, 30.0]))
    candidates = [normalize_grasp_yaw(base_yaw + offset) for offset in offsets]
    for yaw in np.linspace(0.0, np.pi, max(2, int(args.grasp_yaw_samples)), endpoint=False):
        candidates.append(normalize_grasp_yaw(float(yaw)))
    unique = []
    for yaw in candidates:
        if not any(abs(np.arctan2(np.sin(yaw - existing), np.cos(yaw - existing))) < 1e-3 for existing in unique):
            unique.append(yaw)
    return unique


def compute_clutter_aware_grasp_candidate(object_pose):
    base = compute_top_down_grasp_candidate(object_pose)
    raw_obstacle_points = object_pose.get("obstacle_points")
    if raw_obstacle_points is None:
        raw_obstacle_points = np.zeros((0, 3), dtype=np.float64)
    obstacle_points = np.asarray(raw_obstacle_points, dtype=np.float64)
    raw_target_points = object_pose.get("target_points")
    if raw_target_points is None:
        raw_target_points = np.zeros((0, 3), dtype=np.float64)
    target_points = np.asarray(raw_target_points, dtype=np.float64)
    center_xy = np.asarray(object_pose["position"][:2], dtype=np.float64)
    target_width = float(object_pose.get("minor_extent", base["width"] * 2.0))
    opening_width = float(np.clip(target_width + 0.014, 0.025, 0.078))

    yaw_candidates = build_grasp_yaw_candidates(object_pose)
    best = None
    all_candidates = []
    for yaw in yaw_candidates:
        collisions, clearance = gripper_finger_occupancy(
            obstacle_points,
            center_xy,
            yaw,
            opening_width,
        )
        target_finger_points, _ = gripper_finger_occupancy(
            target_points,
            center_xy,
            yaw,
            opening_width,
        )
        no_finger_collision = collisions == 0
        accepted = (
            target_finger_points <= args.max_target_finger_points
            and (
                no_finger_collision
                or (collisions <= args.max_gripper_collision_points and clearance >= args.min_obstacle_clearance)
            )
        )
        yaw_alignment = abs(np.arctan2(np.sin(yaw - normalize_grasp_yaw(object_pose.get("yaw", 0.0))), np.cos(yaw - normalize_grasp_yaw(object_pose.get("yaw", 0.0)))))
        score = (
            float(clearance if np.isfinite(clearance) else 1.0)
            - 0.002 * collisions
            - 0.004 * target_finger_points
            - 0.001 * yaw_alignment
        )
        candidate = {
            "yaw": float(yaw),
            "collisions": collisions,
            "target_finger_points": target_finger_points,
            "clearance": clearance,
            "accepted": accepted,
            "score": score,
        }
        all_candidates.append(candidate)
        if accepted and (best is None or candidate["score"] > best["score"]):
            best = candidate

    if best is None:
        best = min(
            all_candidates,
            key=lambda item: (
                item["target_finger_points"],
                item["collisions"],
                -float(item["clearance"] if np.isfinite(item["clearance"]) else 1.0),
            ),
        )
        print(
            "Warning: no fully clear gripper yaw found; using best fallback candidate "
            f"yaw={best['yaw']:.3f}, target_finger_points={best['target_finger_points']}, "
            f"collisions={best['collisions']}, clearance={best['clearance']:.3f}"
        )
        best["safe"] = False
    else:
        best["safe"] = True

    grasp_yaw = best["yaw"] + args.grasp_yaw_offset + object_pose.get("grasp_yaw_offset", 0.0)
    position = np.asarray(object_pose["position"], dtype=np.float64) + np.array([0.0, 0.0, args.grasp_z_offset])
    orientation = euler_angles_to_quats(np.array([0.0, args.tool_pitch, grasp_yaw]))
    print(
        "Clutter-aware grasp filter: "
        f"yaw={grasp_yaw:.3f}, collisions={best['collisions']}, "
        f"target_finger_points={best['target_finger_points']}, "
        f"clearance={best['clearance']:.3f}, opening_width={opening_width:.3f}, "
        f"obstacle_points={len(obstacle_points)}"
    )
    grasp_candidate = {
        "position": position,
        "orientation": orientation,
        "width": base["width"],
        "score": best["score"],
        "source": "clutter-aware-top-down",
        "yaw": grasp_yaw,
        "selected_yaw": best["yaw"],
        "collision_count": best["collisions"],
        "target_finger_points": best["target_finger_points"],
        "clearance": best["clearance"],
        "safe": bool(best["safe"]),
        "opening_width": opening_width,
        "candidates": all_candidates,
    }
    save_grasp_candidates(args, object_pose, grasp_candidate, opening_width, obstacle_points, target_points)
    return grasp_candidate


def get_grasp_candidate(object_pose):
    if args.grasp_source == "clutter-aware-top-down":
        return compute_clutter_aware_grasp_candidate(object_pose)
    if args.grasp_source == "scripted-top-down":
        return compute_top_down_grasp_candidate(object_pose)
    raise ValueError(f"Unsupported grasp source: {args.grasp_source}")


def build_pick_place_waypoints(grasp_position, place_position):
    approach_height = args.approach_height
    place_approach_height = args.place_approach_height
    pre_place = place_position + np.array([0.0, 0.0, place_approach_height])
    safe_z = max(float(grasp_position[2] + approach_height), float(pre_place[2])) + args.transfer_height_margin
    safe_place_z = safe_z
    return {
        "safe_grasp": np.array([grasp_position[0], grasp_position[1], safe_z]),
        "pre_grasp": np.array([grasp_position[0], grasp_position[1], grasp_position[2] + approach_height]),
        "grasp": grasp_position,
        "safe_place": np.array([place_position[0], place_position[1], safe_place_z]),
        "pre_place": pre_place,
        "place": place_position,
        "post_place_safe": np.array([place_position[0], place_position[1], safe_place_z]),
    }


def execute_pick(
    world,
    robot,
    ik_solver,
    articulation_ik,
    cube,
    object_pose,
    grasp_candidate,
    waypoints,
    reset_object=True,
):
    if reset_object:
        reset_cube_pose(world, cube, object_pose["position"], object_pose["orientation"])

    print("Opening gripper")
    command_gripper(world, robot, 0.04, steps=args.gripper_steps)

    print("Moving from home to safe Z above object")
    move_to_pose(
        world,
        robot,
        ik_solver,
        articulation_ik,
        waypoints["safe_grasp"],
        grasp_candidate["orientation"],
        steps=args.long_move_steps,
    )

    print("Lowering vertically to pre-grasp")
    move_cartesian_line(
        world,
        robot,
        ik_solver,
        articulation_ik,
        waypoints["safe_grasp"],
        waypoints["pre_grasp"],
        grasp_candidate["orientation"],
        segments=12,
    )

    print("Lowering vertically to grasp")
    move_cartesian_line(
        world,
        robot,
        ik_solver,
        articulation_ik,
        waypoints["pre_grasp"],
        waypoints["grasp"],
        grasp_candidate["orientation"],
        segments=8,
    )

    print("Closing gripper")
    command_gripper(world, robot, grasp_candidate["width"], steps=args.gripper_steps)
    step_world(world, args.settle_steps)

    print("Moving up in Z only to safe Z")
    move_cartesian_line(
        world,
        robot,
        ik_solver,
        articulation_ik,
        waypoints["grasp"],
        waypoints["safe_grasp"],
        grasp_candidate["orientation"],
        segments=16,
        gripper_width=grasp_candidate["width"],
    )


def evaluate_lift(cube, object_pose):
    lifted_position = get_cube_position(cube)
    lift_delta = float(lifted_position[2] - object_pose["position"][2])
    success = lift_delta >= args.lift_success_threshold
    print(
        f"Lift check: {'success' if success else 'failed'} "
        f"(cube_z_delta={lift_delta:.3f}, threshold={args.lift_success_threshold:.3f})"
    )
    return {
        "success": success,
        "lift_delta": lift_delta,
    }


def evaluate_place(cube, place_position):
    final_position = get_cube_position(cube)
    place_position = np.asarray(place_position, dtype=np.float64)
    xy_error = float(np.linalg.norm(final_position[:2] - place_position[:2]))
    z_ok = float(final_position[2]) >= args.place_success_z_min
    success = xy_error <= args.place_success_xy_threshold and z_ok
    print(
        f"Place check: {'success' if success else 'failed'} "
        f"(xy_error={xy_error:.3f}, threshold={args.place_success_xy_threshold:.3f}, "
        f"final_z={float(final_position[2]):.3f})"
    )
    return {
        "success": success,
        "xy_error": xy_error,
        "final_position": final_position,
    }


def execute_place(world, robot, ik_solver, articulation_ik, grasp_candidate, waypoints):
    print("Moving X/Y only at safe Z to top of place")
    if not move_cartesian_line(
        world,
        robot,
        ik_solver,
        articulation_ik,
        waypoints["safe_grasp"],
        waypoints["safe_place"],
        grasp_candidate["orientation"],
        segments=24,
        gripper_width=grasp_candidate["width"],
    ):
        print(f"Place failed: IK could not reach safe place target {waypoints['safe_place'].tolist()}")
        return False

    print("Lowering in Z only to pre-place")
    if not move_cartesian_line(
        world,
        robot,
        ik_solver,
        articulation_ik,
        waypoints["safe_place"],
        waypoints["pre_place"],
        grasp_candidate["orientation"],
        segments=16,
        gripper_width=grasp_candidate["width"],
    ):
        print(f"Place failed: IK could not reach pre-place target {waypoints['pre_place'].tolist()}")
        return False

    print("Lowering in Z only to place")
    if not move_cartesian_line(
        world,
        robot,
        ik_solver,
        articulation_ik,
        waypoints["pre_place"],
        waypoints["place"],
        grasp_candidate["orientation"],
        segments=8,
        gripper_width=grasp_candidate["width"],
    ):
        print(f"Place failed: IK could not reach place target {waypoints['place'].tolist()}")
        return False

    print("Opening gripper and releasing object")
    command_gripper(world, robot, 0.04, steps=args.gripper_steps)
    step_world(world, args.settle_steps)

    print("Moving up in Z only to safe place Z")
    if not move_cartesian_line(
        world,
        robot,
        ik_solver,
        articulation_ik,
        waypoints["place"],
        waypoints["post_place_safe"],
        grasp_candidate["orientation"],
        segments=16,
    ):
        print(f"Warning: IK could not reach post-place safe target {waypoints['post_place_safe'].tolist()}")

    return True


def recover_to_home(world, robot):
    print("Returning arm to home")
    move_to_home(world, robot)


def run_pick_place_trial(
    world,
    robot,
    cube,
    ik_solver,
    articulation_ik,
    cube_position,
    cube_orientation,
    cube_yaw,
    place_position,
    object_spec=None,
    reset_before_pick=True,
):
    object_spec = object_spec or {}
    object_pose = {
        "position": cube_position,
        "orientation": cube_orientation,
        "yaw": cube_yaw,
        "name": object_spec.get("name", "pick_object"),
        "closed_width": object_spec.get("closed_width", args.closed_gripper_width),
        "grasp_yaw_offset": object_spec.get("grasp_yaw_offset", 0.0),
        "source": object_spec.get("pose_source", "scripted"),
        "target_points": object_spec.get("target_points"),
        "obstacle_points": object_spec.get("obstacle_points"),
        "minor_extent": object_spec.get("minor_extent"),
        "major_extent": object_spec.get("major_extent"),
        "perception_output_dir": object_spec.get("perception_output_dir"),
    }
    grasp_candidate = get_grasp_candidate(object_pose)
    if not grasp_candidate.get("safe", True) and not args.allow_unsafe_grasp_fallback:
        print(
            "No collision-free grasp candidate was found. "
            "Skipping execution so the gripper does not hit clutter."
        )
        return {
            "success": False,
            "skipped": True,
            "skip_reason": "no collision-free clutter-aware grasp",
            "lift_delta": 0.0,
            "cube_position": object_pose["position"],
            "cube_yaw": object_pose["yaw"],
            "object_name": object_pose["name"],
            "place_position": place_position,
            "grasp_source": grasp_candidate["source"],
            "grasp_score": grasp_candidate["score"],
        }
    print(
        f"Using grasp candidate for {object_pose['name']} from {grasp_candidate['source']}: "
        f"position={grasp_candidate['position'].tolist()}, "
        f"grasp_yaw_rad={grasp_candidate['yaw']:.3f}, "
        f"width={grasp_candidate['width']:.3f}, score={grasp_candidate['score']:.3f}"
    )
    waypoints = build_pick_place_waypoints(grasp_candidate["position"], place_position)

    execute_pick(
        world,
        robot,
        ik_solver,
        articulation_ik,
        cube,
        object_pose,
        grasp_candidate,
        waypoints,
        reset_object=reset_before_pick,
    )
    lift_result = evaluate_lift(cube, object_pose)

    if not lift_result["success"]:
        print("Opening gripper after failed pick")
        command_gripper(world, robot, 0.04, steps=args.gripper_steps)
        recover_to_home(world, robot)
        return {
            "success": False,
            "lift_delta": lift_result["lift_delta"],
            "cube_position": object_pose["position"],
            "cube_yaw": object_pose["yaw"],
            "object_name": object_pose["name"],
            "place_position": place_position,
            "grasp_source": grasp_candidate["source"],
            "grasp_score": grasp_candidate["score"],
        }

    place_motion_success = execute_place(world, robot, ik_solver, articulation_ik, grasp_candidate, waypoints)
    place_result = evaluate_place(cube, place_position)
    recover_to_home(world, robot)
    return {
        "success": place_motion_success and place_result["success"],
        "lift_delta": lift_result["lift_delta"],
        "cube_position": object_pose["position"],
        "cube_yaw": object_pose["yaw"],
        "object_name": object_pose["name"],
        "place_position": place_position,
        "place_xy_error": place_result["xy_error"],
        "final_position": place_result["final_position"],
        "grasp_source": grasp_candidate["source"],
        "grasp_score": grasp_candidate["score"],
    }


def apply_high_friction_material(cube):
    material = PhysicsMaterial(
        prim_path=HIGH_FRICTION_MATERIAL_PATH,
        static_friction=args.friction,
        dynamic_friction=args.friction,
        restitution=0.0,
    )
    if args.object_mode == "usd":
        print("Using sim-ready USD object physics/materials; tuning Panda finger friction only.")
    else:
        print("Applying high-friction material to pick object")
        cube.apply_physics_material(material)

    create_finger_pad_colliders(material)

    if GeometryPrim is None:
        carb.log_warn("GeometryPrim is unavailable; could not tune Panda finger friction.")
        return

    print("Applying high-friction material to Panda fingers")
    for finger_name in ("panda_leftfinger", "panda_rightfinger"):
        finger_path = f"{ROBOT_PRIM_PATH}/{finger_name}"
        if not is_prim_path_valid(finger_path):
            carb.log_warn(f"Finger prim not found for material tuning: {finger_path}")
            continue
        try:
            GeometryPrim(finger_path).apply_physics_material(material, weaker_than_descendants=False)
        except Exception as exc:
            carb.log_warn(f"Could not apply friction material to {finger_path}: {exc}")


def disable_existing_finger_colliders(finger_path):
    if args.keep_finger_mesh_colliders:
        return

    stage = get_current_stage()
    finger_prim = stage.GetPrimAtPath(finger_path)
    disabled_count = 0
    for prim in Usd.PrimRange(finger_prim):
        if str(prim.GetPath()).endswith(f"/{FINGER_PAD_COLLIDER_NAME}"):
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_api = UsdPhysics.CollisionAPI(prim)
            collision_api.CreateCollisionEnabledAttr().Set(False)
            disabled_count += 1
    if disabled_count:
        print(f"Disabled {disabled_count} existing collider prim(s) under {finger_path}")


def create_finger_pad_colliders(material):
    if args.disable_finger_pad_colliders:
        print("Finger pad box colliders disabled.")
        return

    stage = get_current_stage()
    material_prim_path = getattr(material, "prim_path", HIGH_FRICTION_MATERIAL_PATH)
    material_prim = stage.GetPrimAtPath(material_prim_path)
    if not material_prim:
        carb.log_warn(f"Physics material prim not found for finger pad colliders: {material_prim_path}")
        return

    usd_material = UsdShade.Material(material_prim)
    pad_size = np.asarray(args.finger_pad_size, dtype=np.float64)
    if np.any(pad_size <= 0.0):
        raise ValueError(f"--finger-pad-size must contain positive dimensions, got {pad_size.tolist()}")
    if args.finger_pad_inward_protrusion < 0.0 or args.finger_pad_inward_protrusion > pad_size[1]:
        raise ValueError(
            "--finger-pad-inward-protrusion must be between 0 and the pad y-size, "
            f"got {args.finger_pad_inward_protrusion} for y-size {pad_size[1]}"
        )

    pad_y_center = 0.5 * pad_size[1] - args.finger_pad_inward_protrusion
    pad_y_offsets = {
        "panda_leftfinger": pad_y_center,
        "panda_rightfinger": -pad_y_center,
    }
    print(
        "Adding invisible high-friction box colliders to Panda finger pads "
        f"(size={pad_size.tolist()}, inward_protrusion={args.finger_pad_inward_protrusion:.4f}, "
        f"z_offset={args.finger_pad_z_offset:.3f})"
    )
    for finger_name, pad_y_offset in pad_y_offsets.items():
        finger_path = f"{ROBOT_PRIM_PATH}/{finger_name}"
        if not is_prim_path_valid(finger_path):
            carb.log_warn(f"Finger prim not found for pad collider: {finger_path}")
            continue

        disable_existing_finger_colliders(finger_path)

        collider_path = f"{finger_path}/{FINGER_PAD_COLLIDER_NAME}"
        if is_prim_path_valid(collider_path):
            delete_prim(collider_path)

        cube = UsdGeom.Cube.Define(stage, collider_path)
        cube.CreateSizeAttr(1.0)
        cube.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        cube.CreatePurposeAttr(UsdGeom.Tokens.proxy)
        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(0.0, float(pad_y_offset), float(args.finger_pad_z_offset)))
        xform.AddScaleOp().Set(Gf.Vec3f(float(pad_size[0]), float(pad_size[1]), float(pad_size[2])))

        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(usd_material)


def print_trial_summary(trial_results):
    successes = sum(1 for result in trial_results if result["success"])
    skipped = sum(1 for result in trial_results if result.get("skipped"))
    print("Trial summary")
    for result in trial_results:
        status = "skipped" if result.get("skipped") else ("success" if result["success"] else "failed")
        print(
            f"  Trial {result['trial']}: {status}, "
            f"object={result.get('object_name', 'pick_object')}, "
            f"lift_delta={result['lift_delta']:.3f}, "
            f"cube_position={result['cube_position'].tolist()}, cube_yaw_rad={result['cube_yaw']:.3f}, "
            f"place_position={result.get('place_position', np.array(args.place_position)).tolist()}, "
            f"place_xy_error={result.get('place_xy_error', float('nan')):.3f}, "
            f"grasp_source={result['grasp_source']}, grasp_score={result['grasp_score']:.3f}"
        )
        if result.get("skip_reason"):
            print(f"    skip_reason={result['skip_reason']}")
    print(f"Successful trials: {successes}/{len(trial_results)}; skipped: {skipped}")


def main():
    require_file(args.usd_path)
    require_file(args.urdf_path)
    require_file(args.robot_description_path)

    open_stage(args.usd_path)
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    object_size = np.array(args.object_size, dtype=np.float64)
    if args.cube_size is not None:
        object_size = np.array([args.cube_size, args.cube_size, args.cube_size], dtype=np.float64)

    robot = world.scene.add(Articulation(prim_path=ROBOT_PRIM_PATH, name="franka"))

    rng = np.random.default_rng(args.random_seed)
    random_trial_objects = args.trial_object_source != "fixed"
    if random_trial_objects and args.object_mode != "usd":
        raise ValueError("--trial-object-source random-usd-list/sequence-usd-list requires --object-mode usd")

    cube = None
    if random_trial_objects:
        apply_high_friction_material(cube)
    else:
        if is_prim_path_valid(OBJECT_PRIM_PATH):
            delete_prim(OBJECT_PRIM_PATH)
        cube = create_pick_object(world, object_size)
        apply_high_friction_material(cube)

    reset_world_and_play(world)
    step_world(world, 80)

    ik_solver = LulaKinematicsSolver(
        robot_description_path=args.robot_description_path,
        urdf_path=args.urdf_path,
    )
    print("Lula frames:", ik_solver.get_all_frame_names())
    if EE_FRAME not in ik_solver.get_all_frame_names():
        raise ValueError(f"{EE_FRAME!r} is not a valid Lula frame. Pick one from the printed frame list.")

    articulation_ik = ArticulationKinematicsSolver(robot, ik_solver, EE_FRAME)

    base_cube_position = np.array(args.cube_position, dtype=np.float64)
    base_place_position = np.array(args.place_position, dtype=np.float64)
    trial_object_sequence = build_trial_object_sequence(rng)
    if random_trial_objects:
        print(f"Trial object sequence: {trial_object_sequence}")
    scene_object_names = list(args.scene_objects or [])
    if args.persistent_clutter:
        scene_object_names = filter_persistent_clutter_objects(scene_object_names)
    scene_target_name = None
    scene_prompt = args.sam_prompt
    if scene_object_names:
        scene_target_name = args.target_object or infer_target_object_from_prompt(scene_prompt)
        if scene_target_name not in scene_object_names:
            if scene_target_name is not None:
                raise ValueError(f"--target-object {scene_target_name!r} must be one of --scene-objects {scene_object_names}.")
            print("Scene clutter mode will ask which object to pick after spawning the tray.")
        if args.object_mode != "usd":
            raise ValueError("--scene-objects requires --object-mode usd.")
        print(
            f"Scene clutter mode: objects={scene_object_names}, "
            f"target={scene_target_name or '<ask>'}, sam_prompt={scene_prompt or '<ask>'}"
        )
        if args.persistent_clutter and args.pose_source != "sam3-rgbd":
            raise ValueError("--persistent-clutter currently requires --pose-source sam3-rgbd.")
    trial_results = []
    perception_capture = None
    sam_predictor = None
    if args.pose_source == "sam3-rgbd":
        if not random_trial_objects:
            raise ValueError("--pose-source sam3-rgbd currently expects --object-mode usd with a trial object list.")
        stage = omni.usd.get_context().get_stage()
        perception_capture = create_perception_capture(stage)
        sam_predictor = make_sam3_predictor()
        print(f"Using SAM 3 RGB-D perception from camera {args.camera_prim_path}")

    if scene_object_names and args.persistent_clutter:
        run_persistent_clutter_loop(
            world,
            robot,
            ik_solver,
            articulation_ik,
            perception_capture,
            sam_predictor,
            rng,
            object_size,
            scene_object_names,
        )
        print("Done. Close the Isaac Sim window to exit.")
        while simulation_app.is_running():
            step_world(world)
        return

    for trial_index in range(args.trials):
        reset_before_pick = True
        if random_trial_objects:
            spawned_scene = []
            if scene_object_names:
                cleanup_scene_objects(scene_object_names)
                step_world(world, 10)
                place_position = sample_random_place_position(rng)
                target_pose = None
                target_spec = None
                target_cube = None
                sampled_positions = []

                for scene_index, scene_object_name in enumerate(scene_object_names):
                    scene_spec = resolve_trial_object_spec(
                        scene_object_name,
                        prim_path=scene_prim_path(scene_object_name),
                    )
                    scene_pose = sample_random_pose(rng, scene_spec)
                    for _ in range(50):
                        xy = scene_pose["position"][:2]
                        if all(np.linalg.norm(xy - existing_xy) >= 0.10 for existing_xy in sampled_positions):
                            break
                        scene_pose = sample_random_pose(rng, scene_spec)
                    sampled_positions.append(scene_pose["position"][:2].copy())

                    scene_cube = create_pick_object(
                        world,
                        object_size,
                        object_spec=scene_spec,
                        position=scene_pose["position"],
                        orientation=scene_pose["orientation"],
                    )
                    scene_bbox = get_geometry_bbox(scene_spec["prim_path"])
                    floor_delta = args.bin_floor_z + args.spawn_clearance - float(scene_bbox["min"][2])
                    if abs(floor_delta) > 1e-5:
                        scene_pose["position"] = scene_pose["position"] + np.array([0.0, 0.0, floor_delta])
                        reset_cube_pose(world, scene_cube, scene_pose["position"], scene_pose["orientation"])

                    spawned_scene.append((scene_object_name, scene_spec, scene_pose, scene_cube))
                    if scene_target_name is not None and scene_object_name == scene_target_name:
                        target_pose = scene_pose
                        target_spec = scene_spec
                        target_cube = scene_cube

                reset_world_and_play(world)
                for _, scene_spec, scene_pose, scene_cube in spawned_scene:
                    reset_cube_pose(world, scene_cube, scene_pose["position"], scene_pose["orientation"])

                current_target_name = scene_target_name
                current_prompt = scene_prompt
                if current_target_name is None or current_prompt is None:
                    step_world(world, args.settle_steps, render=True)
                    current_target_name, current_prompt = ask_target_for_scene(scene_object_names)
                    print(f"Using target={current_target_name}, SAM prompt={current_prompt!r}")
                    for scene_object_name, scene_spec, scene_pose, scene_cube in spawned_scene:
                        if scene_object_name == current_target_name:
                            target_pose = scene_pose
                            target_spec = scene_spec
                            target_cube = scene_cube
                            break
                target_spec["sam_prompt"] = current_prompt
                object_spec = target_spec
                object_pose = target_pose
                cube = target_cube
                prim_path = object_spec["prim_path"]
                print(
                    f"Trial {trial_index + 1}: spawned clutter scene, "
                    f"target={current_target_name}, prompt={current_prompt!r}"
                )
                object_spec.pop("closed_width", None)
                object_spec["grasp_yaw_offset"] = 0.0
                if args.pose_source == "sam3-rgbd":
                    step_world(world, args.settle_steps, render=True)
                    object_pose = update_pose_from_sam3_rgbd(
                        world,
                        perception_capture,
                        sam_predictor,
                        object_pose,
                        object_spec,
                        trial_index,
                    )
                    reset_before_pick = False
            else:
                object_spec = resolve_trial_object_spec(trial_object_sequence[trial_index])
                object_pose = sample_random_pose(rng, object_spec)
                place_position = sample_random_place_position(rng)
                prim_path = object_spec["prim_path"]
                if is_prim_path_valid(prim_path):
                    delete_prim(prim_path)
                    step_world(world, 10)
                cube = create_pick_object(
                    world,
                    object_size,
                    object_spec=object_spec,
                    position=object_pose["position"],
                    orientation=object_pose["orientation"],
                )
                bbox = get_geometry_bbox(prim_path)
                floor_delta = args.bin_floor_z + args.spawn_clearance - float(bbox["min"][2])
                if abs(floor_delta) > 1e-5:
                    object_pose["position"] = object_pose["position"] + np.array([0.0, 0.0, floor_delta])
                    reset_cube_pose(world, cube, object_pose["position"], object_pose["orientation"])
                    bbox = get_geometry_bbox(prim_path)

                if args.pose_source == "sam3-rgbd":
                    print(
                        f"Skipping Isaac bbox grasp analysis for {object_spec.get('name', 'pick_object')}; "
                        "SAM RGB-D will estimate pose and gripper width."
                    )
                    object_spec.pop("closed_width", None)
                    object_spec["grasp_yaw_offset"] = 0.0
                else:
                    analysis = analyze_top_down_grasp(bbox, object_spec)
                    print_object_analysis(object_spec.get("name", "pick_object"), analysis)
                    if not analysis["feasible"]:
                        result = {
                            "trial": trial_index + 1,
                            "success": False,
                            "skipped": True,
                            "skip_reason": analysis["reason"],
                            "lift_delta": 0.0,
                            "cube_position": object_pose["position"],
                            "cube_yaw": object_pose["yaw"],
                            "object_name": object_spec.get("name", "pick_object"),
                            "place_position": place_position,
                            "grasp_source": "analysis",
                            "grasp_score": 0.0,
                        }
                        trial_results.append(result)
                        delete_prim(prim_path)
                        step_world(world, 20)
                        reset_world_and_play(world)
                        step_world(world, 40)
                        continue

                    object_spec["closed_width"] = analysis["closed_width"]
                    object_spec["grasp_yaw_offset"] = analysis["grasp_yaw_offset"]
                reset_world_and_play(world)
                reset_cube_pose(world, cube, object_pose["position"], object_pose["orientation"])
                if args.pose_source == "sam3-rgbd":
                    step_world(world, args.settle_steps, render=True)
                    object_pose = update_pose_from_sam3_rgbd(
                        world,
                        perception_capture,
                        sam_predictor,
                        object_pose,
                        object_spec,
                        trial_index,
                    )
                    reset_before_pick = False
        else:
            object_spec = {"name": "fixed", "closed_width": args.closed_gripper_width}
            object_pose = get_scripted_object_pose(trial_index, base_cube_position)
            place_position = base_place_position
            reset_world_and_play(world)
            reset_cube_pose(world, cube, object_pose["position"], object_pose["orientation"])

        print(
            f"Trial {trial_index + 1}/{args.trials}: "
            f"object={object_spec.get('name', 'pick_object')}, "
            f"object_position={object_pose['position'].tolist()}, "
            f"object_yaw_rad={object_pose['yaw']:.3f}, "
            f"place_position={place_position.tolist()}, object_size={object_size.tolist()}"
        )
        result = run_pick_place_trial(
            world,
            robot,
            cube,
            ik_solver,
            articulation_ik,
            object_pose["position"],
            object_pose["orientation"],
            object_pose["yaw"],
            place_position,
            object_spec=object_spec,
            reset_before_pick=reset_before_pick,
        )
        result["trial"] = trial_index + 1
        trial_results.append(result)

        if random_trial_objects:
            if scene_object_names:
                cleanup_scene_objects(scene_object_names)
                step_world(world, 20)
            else:
                prim_path = object_spec["prim_path"]
                if is_prim_path_valid(prim_path):
                    delete_prim(prim_path)
                    step_world(world, 20)
            reset_world_and_play(world)
            step_world(world, 40)

    print_trial_summary(trial_results)

    print("Done. Close the Isaac Sim window to exit.")
    while simulation_app.is_running():
        step_world(world)


try:
    main()
except Exception:
    print("Fatal error in pick/place script:")
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
