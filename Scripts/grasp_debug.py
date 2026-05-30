import json
from pathlib import Path

import numpy as np


def finite_json_float(value):
    value = float(value)
    if np.isfinite(value):
        return value
    return None


def save_grasp_candidates(args, object_pose, grasp_candidate, opening_width, obstacle_points, target_points):
    output_dir = object_pose.get("perception_output_dir")
    if not output_dir:
        return
    path = Path(output_dir) / "grasp_candidates.json"
    candidates = []
    for candidate in grasp_candidate["candidates"]:
        yaw_delta = np.arctan2(
            np.sin(candidate["yaw"] - grasp_candidate["selected_yaw"]),
            np.cos(candidate["yaw"] - grasp_candidate["selected_yaw"]),
        )
        candidates.append(
            {
                "yaw_rad": finite_json_float(candidate["yaw"]),
                "yaw_deg": finite_json_float(np.rad2deg(candidate["yaw"])),
                "collisions": int(candidate["collisions"]),
                "target_finger_points": int(candidate["target_finger_points"]),
                "clearance_m": finite_json_float(candidate["clearance"]),
                "accepted": bool(candidate["accepted"]),
                "score": finite_json_float(candidate["score"]),
                "chosen": bool(abs(yaw_delta) < 1e-6),
            }
        )

    payload = {
        "object_name": object_pose.get("name", "pick_object"),
        "target_position": np.asarray(object_pose["position"], dtype=np.float64).tolist(),
        "target_yaw_rad": finite_json_float(object_pose.get("yaw", 0.0)),
        "selected_yaw_rad": finite_json_float(grasp_candidate["selected_yaw"]),
        "selected_yaw_deg": finite_json_float(np.rad2deg(grasp_candidate["selected_yaw"])),
        "commanded_yaw_rad": finite_json_float(grasp_candidate["yaw"]),
        "opening_width_m": finite_json_float(opening_width),
        "closed_width_m": finite_json_float(grasp_candidate["width"]),
        "obstacle_point_count": int(len(obstacle_points)),
        "target_point_count": int(len(target_points)),
        "gripper_finger_length_m": finite_json_float(args.gripper_finger_length),
        "gripper_finger_thickness_m": finite_json_float(args.gripper_finger_thickness),
        "min_obstacle_clearance_m": finite_json_float(args.min_obstacle_clearance),
        "max_gripper_collision_points": int(args.max_gripper_collision_points),
        "max_target_finger_points": int(args.max_target_finger_points),
        "candidates": candidates,
    }
    with open(path, "w", encoding="utf-8") as grasp_file:
        json.dump(payload, grasp_file, indent=2)
