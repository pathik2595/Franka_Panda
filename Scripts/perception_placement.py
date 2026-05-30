import json
from pathlib import Path

import numpy as np


def point_distance_to_set_xy(candidate_xy, points_xy):
    if points_xy.size == 0:
        return float("inf")
    distances = np.linalg.norm(points_xy - candidate_xy, axis=1)
    return float(np.min(distances))


def save_perception_place_debug(output_dir, payload):
    if not output_dir:
        return
    path = Path(output_dir) / "perception_place.json"
    with open(path, "w", encoding="utf-8") as place_file:
        json.dump(payload, place_file, indent=2)


def sample_perception_place_position(
    args,
    world,
    perception_capture,
    rng,
    occupied_positions,
    capture_rgbd_fn,
    project_depth_mask_to_world_fn,
    fallback_place_fn,
    output_dir=None,
):
    _, depth = capture_rgbd_fn(world, perception_capture)
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    points = project_depth_mask_to_world_fn(depth, valid, perception_capture["camera_prim"])
    x_min, x_max = args.place_random_x_range
    y_min, y_max = args.place_random_y_range
    place_points = points[
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
    ]
    if place_points.size:
        place_floor_z = float(np.percentile(place_points[:, 2], 10))
    else:
        place_floor_z = float(args.bin_floor_z)
    occupied_points = place_points[place_points[:, 2] > place_floor_z + args.place_occupied_z_margin]
    occupied_xy = occupied_points[:, :2] if occupied_points.size else np.zeros((0, 2), dtype=np.float64)

    candidates = []
    xs = np.arange(x_min, x_max + 1e-9, args.place_grid_step)
    ys = np.arange(y_min, y_max + 1e-9, args.place_grid_step)
    memory_points = (
        np.asarray([pos[:2] for pos in occupied_positions], dtype=np.float64)
        if occupied_positions
        else np.zeros((0, 2), dtype=np.float64)
    )
    for x in xs:
        for y in ys:
            candidate_xy = np.array([x, y], dtype=np.float64)
            perception_clearance = point_distance_to_set_xy(candidate_xy, occupied_xy)
            memory_clearance = point_distance_to_set_xy(candidate_xy, memory_points)
            clearance = min(perception_clearance, memory_clearance)
            edge_clearance = min(x - x_min, x_max - x, y - y_min, y_max - y)
            accepted = clearance >= args.place_clearance_radius
            score = clearance + 0.25 * edge_clearance
            candidates.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "clearance": float(clearance if np.isfinite(clearance) else 1.0),
                    "perception_clearance": float(perception_clearance if np.isfinite(perception_clearance) else 1.0),
                    "memory_clearance": float(memory_clearance if np.isfinite(memory_clearance) else 1.0),
                    "edge_clearance": float(edge_clearance),
                    "accepted": bool(accepted),
                    "score": float(score if np.isfinite(score) else 1.0 + 0.25 * edge_clearance),
                }
            )

    accepted_candidates = [candidate for candidate in candidates if candidate["accepted"]]
    if not accepted_candidates:
        save_perception_place_debug(
            output_dir,
            {
                "source": "perception-place",
                "selected": None,
                "fallback": "slot-tracker",
                "place_floor_z": float(place_floor_z),
                "place_point_count": int(len(place_points)),
                "occupied_point_count": int(len(occupied_points)),
                "candidate_count": int(len(candidates)),
                "place_clearance_radius": float(args.place_clearance_radius),
                "candidates": candidates,
            },
        )
        print("Warning: perception place search found no free cell; using slot fallback.")
        return fallback_place_fn(rng, occupied_positions)

    selected = accepted_candidates[int(rng.integers(0, len(accepted_candidates)))]
    jitter = rng.uniform(-args.place_grid_step * 0.35, args.place_grid_step * 0.35, size=2)
    candidate = np.array(
        [
            np.clip(selected["x"] + jitter[0], x_min, x_max),
            np.clip(selected["y"] + jitter[1], y_min, y_max),
            args.place_position[2],
        ],
        dtype=np.float64,
    )
    save_perception_place_debug(
        output_dir,
        {
            "source": "perception-place",
            "selected": selected,
            "selected_with_jitter": candidate.tolist(),
            "place_floor_z": float(place_floor_z),
            "place_point_count": int(len(place_points)),
            "occupied_point_count": int(len(occupied_points)),
            "candidate_count": int(len(candidates)),
            "accepted_candidate_count": int(len(accepted_candidates)),
            "place_clearance_radius": float(args.place_clearance_radius),
            "place_bounds": {
                "x_range": [float(x_min), float(x_max)],
                "y_range": [float(y_min), float(y_max)],
            },
            "candidates": candidates,
        },
    )
    print(
        "Perception place search: "
        f"selected={[round(float(candidate[0]), 4), round(float(candidate[1]), 4), round(float(candidate[2]), 4)]}, "
        f"clearance={selected['clearance']:.3f}, occupied_points={len(occupied_points)}"
    )
    return candidate
