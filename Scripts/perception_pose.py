import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import omni.replicator.core as rep
from pxr import Gf, Usd, UsdGeom


def visible_rgb_stats(rgb):
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[2] >= 3:
        rgb = rgb[:, :, :3]
    if not rgb.size:
        return 0.0, 0, 0, 0.0
    return float(np.mean(rgb)), int(np.min(rgb)), int(np.max(rgb)), float(np.std(rgb))


def is_valid_rgb_frame(rgb):
    rgb_mean, rgb_min, rgb_max, rgb_std = visible_rgb_stats(rgb)
    return rgb_max > 0 and rgb_mean > 5.0 and rgb_std > 15.0 and (rgb_max - rgb_min) > 60


def save_rgb_png(rgb, output_path):
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    Image.fromarray(rgb.astype(np.uint8)).save(output_path)


def compute_mask_bbox(mask):
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size == 0:
        return None
    return [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]


def bbox_center(bbox):
    return np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float64)


def bbox_is_in_pick_tray(bbox, image_shape):
    if bbox is None:
        return False
    height, width = image_shape[:2]
    center = bbox_center(bbox)
    x_min = width * 0.20
    x_max = width * 0.70
    y_min = height * 0.55
    y_max = height * 0.98
    return x_min <= center[0] <= x_max and y_min <= center[1] <= y_max


def save_sam_debug_images(rgb, mask, output_dir):
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    rgb_image = Image.fromarray(rgb.astype(np.uint8)).convert("RGBA")
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (rgb_image.height, rgb_image.width):
        mask = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (rgb_image.width, rgb_image.height),
                resample=Image.Resampling.NEAREST,
            ),
            dtype=np.uint8,
        ) > 0

    overlay_color = np.zeros((rgb_image.height, rgb_image.width, 4), dtype=np.uint8)
    overlay_color[mask] = np.array([0, 255, 80, 110], dtype=np.uint8)
    overlay = Image.alpha_composite(rgb_image, Image.fromarray(overlay_color))
    overlay.save(output_dir / "sam_overlay.png")

    bbox_image = overlay.copy()
    bbox = compute_mask_bbox(mask)
    if bbox:
        draw = ImageDraw.Draw(bbox_image)
        draw.rectangle(bbox, outline=(255, 40, 40, 255), width=3)
        label = f"SAM bbox {bbox[2] - bbox[0] + 1}x{bbox[3] - bbox[1] + 1}"
        draw.rectangle([bbox[0], max(0, bbox[1] - 18), bbox[0] + 150, bbox[1]], fill=(255, 40, 40, 220))
        draw.text((bbox[0] + 4, max(0, bbox[1] - 16)), label, fill=(255, 255, 255, 255))
    bbox_image.save(output_dir / "sam_bbox.png")
    return bbox


def save_sam_prompt_attempt_debug(rgb, mask, output_dir, attempt_index, prompt):
    safe_prompt = "".join(ch if ch.isalnum() else "_" for ch in prompt.lower()).strip("_")
    attempt_dir = Path(output_dir) / "sam_prompt_attempts"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{attempt_index:02d}_{safe_prompt or 'prompt'}"
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    rgb_image = Image.fromarray(rgb.astype(np.uint8)).convert("RGBA")
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (rgb_image.height, rgb_image.width):
        mask = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (rgb_image.width, rgb_image.height),
                resample=Image.Resampling.NEAREST,
            ),
            dtype=np.uint8,
        ) > 0
    Image.fromarray(mask.astype(np.uint8) * 255).save(attempt_dir / f"{prefix}_mask.png")
    overlay_color = np.zeros((rgb_image.height, rgb_image.width, 4), dtype=np.uint8)
    overlay_color[mask] = np.array([255, 210, 0, 110], dtype=np.uint8)
    Image.alpha_composite(rgb_image, Image.fromarray(overlay_color)).save(attempt_dir / f"{prefix}_overlay.png")


def create_perception_capture(args, stage, find_camera_prim_fn):
    camera_prim = find_camera_prim_fn(stage, args.camera_prim_path)
    camera_path = str(camera_prim.GetPath())
    render_product = rep.create.render_product(
        camera_path,
        (args.perception_width, args.perception_height),
        name="pick_place_rgbd_capture",
    )
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    rgb_annotator.attach([render_product])
    depth_annotator.attach([render_product])
    return {
        "camera_prim": camera_prim,
        "render_product": render_product,
        "rgb": rgb_annotator,
        "depth": depth_annotator,
    }


def capture_rgbd_for_perception(args, world, perception_capture, step_world_fn):
    last_rgb = None
    last_depth = None
    for attempt in range(args.perception_capture_retries):
        step_world_fn(world, max(1, args.perception_warmup_frames), render=True)
        rep.orchestrator.step(rt_subframes=args.perception_rt_subframes)
        rgb = np.asarray(perception_capture["rgb"].get_data())
        depth = np.asarray(perception_capture["depth"].get_data(), dtype=np.float32)
        last_rgb = rgb
        last_depth = depth
        if is_valid_rgb_frame(rgb):
            return rgb, depth
        rgb_mean, rgb_min, rgb_max, rgb_std = visible_rgb_stats(rgb)
        print(
            f"  perception capture retry {attempt + 1}/{args.perception_capture_retries}: "
            f"RGB frame invalid (mean={rgb_mean:.1f}, min={rgb_min}, max={rgb_max}, std={rgb_std:.2f})"
        )
    return last_rgb, last_depth


def make_sam3_predictor(args, extra_site_packages):
    model_path = Path(args.sam3_model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    missing_modules = []
    for module_name in ("clip", "timm"):
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            missing_modules.append(module_name)
    if missing_modules:
        raise ModuleNotFoundError(
            "Missing SAM 3 dependency module(s) in Isaac Python: "
            f"{', '.join(missing_modules)}. Install them into {extra_site_packages}."
        )
    from ultralytics.models.sam import SAM3SemanticPredictor

    return SAM3SemanticPredictor(
        overrides={
            "conf": args.sam_conf,
            "task": "segment",
            "mode": "predict",
            "model": str(model_path),
            "save": False,
            "verbose": False,
        }
    )


def extract_sam_masks(results):
    if not results or results[0].masks is None or results[0].masks.data is None:
        return []
    masks = results[0].masks.data
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    return [np.asarray(mask, dtype=bool) for mask in masks]


def save_sam_prompt_failure_debug(rgb, masks, output_dir, attempt_index, prompt, reason):
    safe_prompt = "".join(ch if ch.isalnum() else "_" for ch in prompt.lower()).strip("_")
    attempt_dir = Path(output_dir) / "sam_prompt_attempts"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{attempt_index:02d}_{safe_prompt or 'prompt'}_failed"
    with open(attempt_dir / f"{prefix}.txt", "w", encoding="utf-8") as debug_file:
        debug_file.write(str(reason))
    if not masks:
        return
    masks = [np.asarray(mask, dtype=bool) for mask in masks if np.asarray(mask, dtype=bool).any()]
    for mask_index, mask in enumerate(masks[:5]):
        save_sam_prompt_attempt_debug(rgb, mask, output_dir, attempt_index, f"{prompt}_failed_{mask_index}")


def run_sam3_mask(predictor, rgb_path, prompt):
    predictor.set_image(str(rgb_path))
    results = predictor(text=[prompt])
    masks = extract_sam_masks(results)
    if not masks:
        raise RuntimeError(f"SAM 3 returned no masks for prompt {prompt!r}.")
    areas = np.array([int(mask.sum()) for mask in masks], dtype=np.int64)
    return masks[int(np.argmax(areas))]


def geometry_inside_pick_bounds(args, geometry):
    x_min, x_max = args.bin_random_x_range
    y_min, y_max = args.bin_random_y_range
    px, py = np.asarray(geometry["position"], dtype=np.float64)[:2]
    return (x_min <= px <= x_max) and (y_min <= py <= y_max)


def geometry_pick_tray_overlap(args, geometry):
    points = np.asarray(geometry.get("target_points", []), dtype=np.float64)
    if points.size == 0:
        return 0.0
    margin = float(getattr(args, "pick_tray_mask_margin", 0.0))
    x_min, x_max = [float(value) for value in args.bin_random_x_range]
    y_min, y_max = [float(value) for value in args.bin_random_y_range]
    inside = (
        (points[:, 0] >= x_min - margin)
        & (points[:, 0] <= x_max + margin)
        & (points[:, 1] >= y_min - margin)
        & (points[:, 1] <= y_max + margin)
    )
    return float(np.count_nonzero(inside) / max(len(points), 1))


def choose_sam_mask_in_pick_bounds(args, predictor, rgb_path, prompt, rgb_shape, depth, camera_prim, object_spec):
    predictor.set_image(str(rgb_path))
    results = predictor(text=[prompt])
    masks = extract_sam_masks(results)
    if not masks:
        error = RuntimeError(f"SAM 3 returned no masks for prompt {prompt!r}.")
        error.sam_masks = []
        raise error

    candidates = []
    for mask_index, mask in enumerate(masks):
        area = int(np.asarray(mask, dtype=bool).sum())
        if area <= 0:
            continue
        bbox = compute_mask_bbox(mask)
        if args.reject_mask_outside_pick_tray and not bbox_is_in_pick_tray(bbox, rgb_shape):
            continue
        try:
            geometry = estimate_object_geometry_from_mask_depth(args, mask, depth, camera_prim, object_spec)
        except RuntimeError:
            continue
        overlap_fraction = geometry_pick_tray_overlap(args, geometry)
        min_fraction = float(getattr(args, "min_pick_tray_mask_fraction", 0.0))
        inside_pick_bounds = (not args.reject_mask_outside_pick_tray) or overlap_fraction >= min_fraction
        candidates.append(
            {
                "mask_index": mask_index,
                "mask": mask,
                "bbox": bbox,
                "geometry": geometry,
                "area": area,
                "inside_pick_bounds": inside_pick_bounds,
                "pick_tray_overlap_fraction": overlap_fraction,
            }
        )

    valid_candidates = [candidate for candidate in candidates if candidate["inside_pick_bounds"]]
    if valid_candidates:
        return max(valid_candidates, key=lambda candidate: candidate["area"])

    if candidates:
        best = max(candidates, key=lambda candidate: candidate["area"])
        position = np.asarray(best["geometry"]["position"], dtype=np.float64).tolist()
        error = RuntimeError(
            "SAM returned masks, but none overlapped the configured pick tray enough. "
            f"prompt={prompt!r}, best_world_position={position}, "
            f"best_overlap_fraction={best['pick_tray_overlap_fraction']:.3f}, "
            f"required_fraction={float(getattr(args, 'min_pick_tray_mask_fraction', 0.0)):.3f}, "
            f"margin={float(getattr(args, 'pick_tray_mask_margin', 0.0)):.3f}, "
            f"x_range={args.bin_random_x_range}, y_range={args.bin_random_y_range}, bbox={best['bbox']}"
        )
        error.sam_masks = [candidate["mask"] for candidate in candidates]
        raise error

    error = RuntimeError(
        "SAM returned masks, but none passed the pick-tray image/depth filters. "
        f"prompt={prompt!r}, x_range={args.bin_random_x_range}, y_range={args.bin_random_y_range}"
    )
    error.sam_masks = masks
    raise error


def prompt_candidates_for_object(object_spec, fallback_prompt):
    candidates = []
    for prompt in object_spec.get("sam_prompt_candidates", []):
        if prompt and prompt not in candidates:
            candidates.append(prompt)
    if fallback_prompt and fallback_prompt not in candidates:
        candidates.insert(0, fallback_prompt)
    return candidates or [object_spec.get("name", "object")]


def resize_mask_to_depth(mask, depth_shape):
    mask = np.asarray(mask, dtype=bool)
    if mask.shape == depth_shape:
        return mask
    return (
        np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (depth_shape[1], depth_shape[0]),
                resample=Image.Resampling.NEAREST,
            ),
            dtype=np.uint8,
        )
        > 0
    )


def project_depth_mask_to_world(depth, valid_mask, camera_prim):
    depth = np.asarray(depth, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if int(valid_mask.sum()) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    valid = valid_mask & np.isfinite(depth) & (depth > 0.0)
    if int(valid.sum()) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    camera = UsdGeom.Camera(camera_prim)
    focal_length = float(camera.GetFocalLengthAttr().Get(Usd.TimeCode.Default()))
    horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get(Usd.TimeCode.Default()))
    height, width = depth.shape
    fx = focal_length / horizontal_aperture * width
    fy = fx
    cx = width * 0.5
    cy = height * 0.5

    rows, cols = np.nonzero(valid)
    d = depth[rows, cols].astype(np.float64)
    x_norm = (cols.astype(np.float64) - cx) / fx
    y_norm = -(rows.astype(np.float64) - cy) / fy
    ray_norm = np.sqrt(x_norm * x_norm + y_norm * y_norm + 1.0)
    x_camera = x_norm / ray_norm * d
    y_camera = y_norm / ray_norm * d
    z_camera_forward = -d / ray_norm

    camera_matrix = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    world_points = np.array(
        [
            camera_matrix.Transform(Gf.Vec3d(float(x), float(y), float(z)))
            for x, y, z in zip(x_camera, y_camera, z_camera_forward)
        ],
        dtype=np.float64,
    )
    return world_points


def extract_obstacle_points(args, mask, depth, camera_prim):
    mask = resize_mask_to_depth(mask, np.asarray(depth).shape)
    valid = (~mask) & np.isfinite(depth) & (depth > 0.0)
    points = project_depth_mask_to_world(depth, valid, camera_prim)
    if points.size == 0:
        return points
    object_like = points[:, 2] > (args.bin_floor_z + args.gripper_obstacle_z_margin)
    return points[object_like]


def estimate_object_geometry_from_mask_depth(args, mask, depth, camera_prim, object_spec):
    depth = np.asarray(depth, dtype=np.float32)
    mask = resize_mask_to_depth(mask, depth.shape)

    valid = mask & np.isfinite(depth) & (depth > 0.0)
    if int(valid.sum()) < 20:
        raise RuntimeError(f"Not enough valid masked depth pixels: {int(valid.sum())}")

    points_xyz = project_depth_mask_to_world(depth, valid, camera_prim)
    obstacle_points = extract_obstacle_points(args, mask, depth, camera_prim)
    world_z = points_xyz[:, 2]

    points_xy = points_xyz[:, :2]
    center_xy = np.median(points_xy, axis=0)
    top_z = float(np.percentile(world_z, 95))
    estimated_height = max(top_z - args.bin_floor_z, 0.01)
    center_z = float(args.bin_floor_z + estimated_height * 0.5)

    centered_xy = points_xy - center_xy
    if centered_xy.shape[0] >= 3:
        covariance = np.cov(centered_xy, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        axes = eigenvectors[:, order]
    else:
        axes = np.eye(2)

    projected = centered_xy @ axes
    low = np.percentile(projected, 5, axis=0)
    high = np.percentile(projected, 95, axis=0)
    extents = np.maximum(high - low, 1e-6)
    major_index = int(np.argmax(extents))
    minor_index = int(np.argmin(extents))
    major_extent = float(extents[major_index])
    minor_extent = float(extents[minor_index])
    major_axis = axes[:, major_index]
    yaw = float(np.arctan2(major_axis[1], major_axis[0]))

    squeeze_margin = float(object_spec.get("perception_squeeze_margin", args.grasp_squeeze_margin))
    close_percentiles = getattr(args, "grasp_close_percentiles", [30.0, 70.0])
    close_low, close_high = sorted(float(value) for value in close_percentiles)
    close_low = float(np.clip(close_low, 0.0, 49.0))
    close_high = float(np.clip(close_high, 51.0, 100.0))
    contact_low = float(np.percentile(projected[:, minor_index], close_low))
    contact_high = float(np.percentile(projected[:, minor_index], close_high))
    contact_extent = max(contact_high - contact_low, 1e-6)
    silhouette_closed_width = minor_extent / 2.0 - squeeze_margin
    contact_closed_width = contact_extent / 2.0 - args.grasp_squeeze_margin
    closed_width = min(silhouette_closed_width, contact_closed_width)
    closed_width = float(np.clip(closed_width, args.min_computed_gripper_width, args.max_computed_gripper_width))
    if object_spec.get("name") == "peach":
        yaw = 0.0

    return {
        "position": np.array([float(center_xy[0]), float(center_xy[1]), center_z], dtype=np.float64),
        "yaw": yaw,
        "closed_width": closed_width,
        "silhouette_closed_width": float(silhouette_closed_width),
        "contact_closed_width": float(contact_closed_width),
        "contact_extent": float(contact_extent),
        "close_percentiles": [close_low, close_high],
        "squeeze_margin": squeeze_margin,
        "major_extent": major_extent,
        "minor_extent": minor_extent,
        "top_z": top_z,
        "estimated_height": float(estimated_height),
        "z_percentiles": [
            float(np.percentile(world_z, 5)),
            float(np.percentile(world_z, 50)),
            top_z,
        ],
        "target_points": points_xyz,
        "obstacle_points": obstacle_points,
        "valid_pixels": int(valid.sum()),
    }


def update_pose_from_sam3_rgbd(
    args,
    world,
    perception_capture,
    sam_predictor,
    object_pose,
    object_spec,
    trial_index,
    step_world_fn,
    euler_angles_to_quats_fn,
):
    prompt = object_spec.get("sam_prompt") or object_spec.get("name", "object")
    output_dir = Path(args.perception_output_dir) / f"trial_{trial_index + 1:03d}_{object_spec.get('name', 'object')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb, depth = capture_rgbd_for_perception(args, world, perception_capture, step_world_fn)
    rgb_path = output_dir / "rgb.png"
    save_rgb_png(rgb, rgb_path)
    np.save(output_dir / "depth_meters.npy", depth)

    if not is_valid_rgb_frame(rgb):
        raise RuntimeError(f"Perception RGB frame is invalid after retries: {rgb_path}")

    selected = None
    prompt_errors = []
    for prompt_index, candidate_prompt in enumerate(prompt_candidates_for_object(object_spec, prompt)):
        try:
            selected = choose_sam_mask_in_pick_bounds(
                args,
                sam_predictor,
                rgb_path,
                candidate_prompt,
                np.asarray(rgb).shape,
                depth,
                perception_capture["camera_prim"],
                object_spec,
            )
            prompt = candidate_prompt
            save_sam_prompt_attempt_debug(rgb, selected["mask"], output_dir, prompt_index, candidate_prompt)
            if prompt_index > 0:
                print(f"  SAM prompt fallback succeeded: {candidate_prompt!r}")
            break
        except RuntimeError as exc:
            prompt_errors.append(f"{candidate_prompt!r}: {exc}")
            save_sam_prompt_failure_debug(
                rgb,
                getattr(exc, "sam_masks", []),
                output_dir,
                prompt_index,
                candidate_prompt,
                exc,
            )
            continue
    if selected is None:
        with open(output_dir / "sam_prompt_failures.txt", "w", encoding="utf-8") as failure_file:
            failure_file.write("\n\n".join(prompt_errors))
        raise RuntimeError(
            "All SAM prompt candidates failed for "
            f"{object_spec.get('name', 'object')}: " + "; ".join(prompt_errors)
        )
    mask = selected["mask"]
    Image.fromarray(mask.astype(np.uint8) * 255).save(output_dir / "sam_mask.png")
    bbox = save_sam_debug_images(rgb, mask, output_dir)
    geometry = selected["geometry"]
    computed_closed_width = geometry["closed_width"]
    np.save(output_dir / "target_points_world.npy", geometry["target_points"])
    np.save(output_dir / "obstacle_points_world.npy", geometry["obstacle_points"])
    with open(output_dir / "perception_geometry.json", "w", encoding="utf-8") as geometry_file:
        json.dump(
            {
                "prompt": prompt,
                "valid_pixels": geometry["valid_pixels"],
                "target_point_count": int(len(geometry["target_points"])),
                "obstacle_point_count": int(len(geometry["obstacle_points"])),
                "bbox_xyxy": bbox,
                "position": geometry["position"].tolist(),
                "yaw": geometry["yaw"],
                "closed_width": computed_closed_width,
                "silhouette_closed_width": geometry["silhouette_closed_width"],
                "contact_closed_width": geometry["contact_closed_width"],
                "contact_extent": geometry["contact_extent"],
                "close_percentiles": geometry["close_percentiles"],
                "squeeze_margin": geometry["squeeze_margin"],
                "major_extent": geometry["major_extent"],
                "minor_extent": geometry["minor_extent"],
                "top_z": geometry["top_z"],
                "estimated_height": geometry["estimated_height"],
                "z_percentiles": geometry["z_percentiles"],
                "camera_prim_path": str(perception_capture["camera_prim"].GetPath()),
            },
            geometry_file,
            indent=2,
        )

    perceived_pose = dict(object_pose)
    perceived_pose["position"] = geometry["position"]
    perceived_pose["yaw"] = geometry["yaw"]
    perceived_pose["orientation"] = euler_angles_to_quats_fn(
        np.array([object_spec.get("roll", 0.0), object_spec.get("pitch", 0.0), geometry["yaw"]])
    )
    perceived_pose["closed_width"] = computed_closed_width
    perceived_pose["silhouette_closed_width"] = geometry["silhouette_closed_width"]
    perceived_pose["contact_closed_width"] = geometry["contact_closed_width"]
    perceived_pose["target_points"] = geometry["target_points"]
    perceived_pose["obstacle_points"] = geometry["obstacle_points"]
    perceived_pose["minor_extent"] = geometry["minor_extent"]
    perceived_pose["major_extent"] = geometry["major_extent"]
    perceived_pose["perception_output_dir"] = str(output_dir)
    perceived_pose["source"] = "sam3-rgbd"
    object_spec["closed_width"] = computed_closed_width
    object_spec["silhouette_closed_width"] = geometry["silhouette_closed_width"]
    object_spec["contact_closed_width"] = geometry["contact_closed_width"]
    object_spec["target_points"] = geometry["target_points"]
    object_spec["obstacle_points"] = geometry["obstacle_points"]
    object_spec["minor_extent"] = geometry["minor_extent"]
    object_spec["major_extent"] = geometry["major_extent"]
    object_spec["perception_output_dir"] = str(output_dir)
    object_spec["grasp_yaw_offset"] = 0.0
    object_spec["pose_source"] = "sam3-rgbd"

    x_min, x_max = args.bin_random_x_range
    y_min, y_max = args.bin_random_y_range
    px, py = perceived_pose["position"][:2]
    in_configured_bin = (x_min <= px <= x_max) and (y_min <= py <= y_max)
    if not in_configured_bin:
        print(
            "Warning: perception center is outside configured pick bounds: "
            f"xy={[float(px), float(py)]}, "
            f"x_range={args.bin_random_x_range}, y_range={args.bin_random_y_range}"
        )
    print(
        "Perception pose estimate: "
        f"prompt={prompt!r}, pixels={geometry['valid_pixels']}, "
        f"perceived_pos={np.round(perceived_pose['position'], 4).tolist()}, "
        f"yaw={geometry['yaw']:.3f}, width={computed_closed_width:.4f}, "
        f"silhouette_width={geometry['silhouette_closed_width']:.4f}, "
        f"contact_width={geometry['contact_closed_width']:.4f}, "
        f"squeeze={geometry['squeeze_margin']:.4f}, "
        f"extents=[{geometry['major_extent']:.4f}, {geometry['minor_extent']:.4f}], "
        f"height={geometry['estimated_height']:.4f}"
    )
    return perceived_pose
