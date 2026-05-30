import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


WORKSPACE = Path(__file__).resolve().parents[1]
EXTRA_SITE_PACKAGES = WORKSPACE / ".isaac_python_packages"
if EXTRA_SITE_PACKAGES.exists():
    sys.path.append(str(EXTRA_SITE_PACKAGES))


def parse_args():
    workspace = Path(__file__).resolve().parents[1]
    asset_root = workspace / "assets"
    parser = argparse.ArgumentParser(
        description="Generate randomized single-object RGB-D scenes and optional SAM 3 segmentation metrics."
    )
    parser.add_argument("--usd-path", default=str(asset_root / "scene.usd"))
    parser.add_argument("--camera-prim-path", default="/World/RGBD_Camera")
    parser.add_argument("--output-dir", default=str(workspace / "isaac_bin_picking" / "perception_dataset" / "sam3_single_object"))
    parser.add_argument("--model-path", default=str(workspace / "sam3.1_multiplex.pt"))
    parser.add_argument("--objects", nargs="+", default=["mug", "peach", "can", "rubiks_cube"])
    parser.add_argument("--scenes-per-object", type=int, default=25)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--warmup-frames", type=int, default=8)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--capture-retries", type=int, default=5)
    parser.add_argument("--rt-subframes", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--bin-random-x-range", nargs=2, type=float, default=[-0.21, -0.09])
    parser.add_argument("--bin-random-y-range", nargs=2, type=float, default=[-0.51, -0.39])
    parser.add_argument("--bin-floor-z", type=float, default=0.25)
    parser.add_argument("--spawn-clearance", type=float, default=0.002)
    parser.add_argument("--max-roll-deg", type=float, default=0.0)
    parser.add_argument("--max-pitch-deg", type=float, default=0.0)
    parser.add_argument("--lay-down-objects", nargs="+", default=["mug", "can"])
    parser.add_argument("--can-label-pitch-deg", type=float, default=90.0)
    parser.add_argument("--use-generated-rubiks-fallback", action="store_true")
    parser.add_argument("--run-sam", action="store_true", help="Run SAM 3 on every generated RGB image.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": args.headless})

import carb.settings
import omni.replicator.core as rep
import omni.usd
from PIL import Image
from pxr import Gf, Usd, UsdGeom
from isaacsim.core.api.world import World
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.core.utils.prims import delete_prim, is_prim_path_valid
from isaacsim.core.utils.semantics import add_labels
from isaacsim.core.utils.stage import add_reference_to_stage, open_stage


OBJECT_CATALOG = {
    "can": {
        "usd_path": WORKSPACE / "assets" / "objects" / "tomatosoupcan" / "tomato_soup_can.usd",
        "prompt": "red and white small cylinder",
    },
    "peach": {
        "usd_path": WORKSPACE / "assets" / "objects" / "peach" / "peach.usd",
        "prompt": "fruit",
    },
    "mug": {
        "usd_path": WORKSPACE / "assets" / "objects" / "mug" / "mug.usd",
        "prompt": "mug",
    },
    "rubiks_cube": {
        "usd_path": WORKSPACE / "assets" / "objects" / "rubiks_cube.usd",
        "prompt": "rubiks cube",
        "cube_size": 0.055,
    },
}
OBJECT_ROOT = "/World/sam3_benchmark_object"
TARGET_LABEL = "sam3_target_object"


def step_world(world, steps=1, render=True):
    for _ in range(steps):
        world.step(render=render)


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


def matrix4_to_list(matrix):
    return [[float(matrix[row][col]) for col in range(4)] for row in range(4)]


def get_camera_metadata(camera_prim, width, height):
    camera = UsdGeom.Camera(camera_prim)
    time_code = Usd.TimeCode.Default()
    focal_length = float(camera.GetFocalLengthAttr().Get(time_code))
    horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get(time_code))
    vertical_aperture = float(camera.GetVerticalApertureAttr().Get(time_code))
    return {
        "camera_prim_path": str(camera_prim.GetPath()),
        "resolution": {"width": width, "height": height},
        "intrinsics": {
            "fx": focal_length / horizontal_aperture * width,
            "fy": focal_length / vertical_aperture * height,
            "cx": width * 0.5,
            "cy": height * 0.5,
            "focal_length": focal_length,
            "horizontal_aperture": horizontal_aperture,
            "vertical_aperture": vertical_aperture,
        },
        "camera_to_world": matrix4_to_list(UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(time_code)),
    }


def set_world_pose(prim_path, position, orientation_wxyz, scale=None):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*np.asarray(position, dtype=np.float64)))
    quat = np.asarray(orientation_wxyz, dtype=np.float64)
    xform.AddOrientOp().Set(Gf.Quatf(float(quat[0]), Gf.Vec3f(float(quat[1]), float(quat[2]), float(quat[3]))))
    if scale is not None:
        scale_values = [float(value) for value in np.asarray(scale, dtype=np.float32)]
        xform.AddScaleOp().Set(Gf.Vec3f(scale_values[0], scale_values[1], scale_values[2]))


def add_target_labels(root_prim):
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Imageable):
            add_labels(prim, labels=[TARGET_LABEL], instance_name="class")


def get_geometry_bbox(prim_path):
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    combined = None
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Boundable):
            continue
        box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        if box.IsEmpty():
            continue
        if combined is None:
            combined = Gf.Range3d(box.GetMin(), box.GetMax())
        else:
            combined.UnionWith(Gf.Range3d(box.GetMin(), box.GetMax()))
    if combined is None or combined.IsEmpty():
        raise ValueError(f"No geometry bbox found under {prim_path}")
    return {
        "min": np.array([combined.GetMin()[0], combined.GetMin()[1], combined.GetMin()[2]], dtype=np.float64),
        "max": np.array([combined.GetMax()[0], combined.GetMax()[1], combined.GetMax()[2]], dtype=np.float64),
        "center": np.array(
            [
                ((combined.GetMin() + combined.GetMax()) * 0.5)[0],
                ((combined.GetMin() + combined.GetMax()) * 0.5)[1],
                ((combined.GetMin() + combined.GetMax()) * 0.5)[2],
            ],
            dtype=np.float64,
        ),
        "size": np.array(
            [
                (combined.GetMax() - combined.GetMin())[0],
                (combined.GetMax() - combined.GetMin())[1],
                (combined.GetMax() - combined.GetMin())[2],
            ],
            dtype=np.float64,
        ),
    }


def spawn_object(object_name, rng):
    if object_name not in OBJECT_CATALOG:
        raise ValueError(f"Unknown object {object_name!r}. Choices: {sorted(OBJECT_CATALOG)}")

    spec = OBJECT_CATALOG[object_name]
    if is_prim_path_valid(OBJECT_ROOT):
        delete_prim(OBJECT_ROOT)

    stage = omni.usd.get_context().get_stage()
    usd_path = Path(spec["usd_path"])
    scale = None

    if object_name == "rubiks_cube" and args.use_generated_rubiks_fallback:
        cube = UsdGeom.Cube.Define(stage, OBJECT_ROOT)
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([(0.9, 0.05, 0.05)])
        scale = np.array([spec["cube_size"], spec["cube_size"], spec["cube_size"]], dtype=np.float64)
    else:
        if not usd_path.exists():
            raise FileNotFoundError(usd_path)
        add_reference_to_stage(usd_path=str(usd_path), prim_path=OBJECT_ROOT)

    root_prim = stage.GetPrimAtPath(OBJECT_ROOT)
    add_target_labels(root_prim)

    x = rng.uniform(args.bin_random_x_range[0], args.bin_random_x_range[1])
    y = rng.uniform(args.bin_random_y_range[0], args.bin_random_y_range[1])
    yaw = rng.uniform(-np.pi, np.pi)
    if object_name in args.lay_down_objects:
        roll = np.pi / 2.0 + np.deg2rad(rng.uniform(-args.max_roll_deg, args.max_roll_deg))
        pitch = np.deg2rad(rng.uniform(-args.max_pitch_deg, args.max_pitch_deg))
        if object_name == "can":
            pitch += np.deg2rad(args.can_label_pitch_deg)
    else:
        roll = np.deg2rad(rng.uniform(-args.max_roll_deg, args.max_roll_deg))
        pitch = np.deg2rad(rng.uniform(-args.max_pitch_deg, args.max_pitch_deg))
    orientation = euler_angles_to_quats(np.array([roll, pitch, yaw], dtype=np.float64))
    position = np.array([x, y, args.bin_floor_z + 0.10], dtype=np.float64)
    set_world_pose(OBJECT_ROOT, position, orientation, scale=scale)

    bbox = get_geometry_bbox(OBJECT_ROOT)
    floor_delta = args.bin_floor_z + args.spawn_clearance - float(bbox["min"][2])
    position = position + np.array([0.0, 0.0, floor_delta], dtype=np.float64)
    set_world_pose(OBJECT_ROOT, position, orientation, scale=scale)
    bbox = get_geometry_bbox(OBJECT_ROOT)

    return {
        "object": object_name,
        "prompt": spec["prompt"],
        "usd_path": str(usd_path),
        "prim_path": OBJECT_ROOT,
        "position": position.tolist(),
        "orientation_wxyz": np.asarray(orientation, dtype=np.float64).tolist(),
        "yaw_rad": float(yaw),
        "roll_rad": float(roll),
        "pitch_rad": float(pitch),
        "bbox": {key: value.tolist() for key, value in bbox.items()},
    }


def semantic_mask_from_data(semantic_data):
    data = np.asarray(semantic_data["data"])
    info = semantic_data.get("info", {})
    id_to_labels = info.get("idToLabels", {})
    target_ids = []

    for raw_id, labels in id_to_labels.items():
        label_values = []
        if isinstance(labels, dict):
            label_values = list(labels.values())
        elif isinstance(labels, (list, tuple)):
            label_values = list(labels)
        elif labels is not None:
            label_values = [labels]
        if any(str(label) == TARGET_LABEL for label in label_values):
            target_ids.append(int(raw_id))

    if data.ndim == 3:
        data = data[:, :, 0]
    mask = np.isin(data, np.asarray(target_ids, dtype=data.dtype)) if target_ids else np.zeros(data.shape, dtype=bool)
    return mask, target_ids, id_to_labels


def save_rgb_png(rgb, output_path):
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    Image.fromarray(rgb.astype(np.uint8)).save(output_path)


def visible_rgb_stats(rgb):
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[2] >= 3:
        rgb = rgb[:, :, :3]
    if not rgb.size:
        return 0.0, 0, 0, 0.0
    return float(np.mean(rgb)), int(np.min(rgb)), int(np.max(rgb)), float(np.std(rgb))


def is_valid_rgb_frame(rgb):
    rgb_mean, rgb_min, rgb_max, rgb_std = visible_rgb_stats(rgb)
    rgb_range = rgb_max - rgb_min
    return rgb_max > 0 and rgb_mean > 5.0 and rgb_std > 15.0 and rgb_range > 60


def save_mask_png(mask, output_path):
    Image.fromarray(mask.astype(np.uint8) * 255).save(output_path)


def save_overlay(rgb, mask, output_path):
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    overlay = rgb.astype(np.uint8).copy()
    color = np.array([0, 220, 80], dtype=np.uint8)
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
    Image.fromarray(overlay).save(output_path)


def capture_annotators(world, rgb_annotator, depth_annotator, semantic_annotator):
    last_capture = None
    for attempt in range(args.capture_retries):
        step_world(world, max(1, args.warmup_frames // 2), render=True)
        rep.orchestrator.step(rt_subframes=args.rt_subframes)
        rgb = np.asarray(rgb_annotator.get_data())
        depth = np.asarray(depth_annotator.get_data(), dtype=np.float32)
        semantic = semantic_annotator.get_data()
        last_capture = (rgb, depth, semantic)
        rgb_mean, rgb_min, rgb_max, rgb_std = visible_rgb_stats(rgb)
        if is_valid_rgb_frame(rgb):
            return last_capture
        print(
            f"  capture retry {attempt + 1}/{args.capture_retries}: "
            f"RGB frame was blank/flat "
            f"(mean={rgb_mean:.1f}, min={rgb_min}, max={rgb_max}, std={rgb_std:.2f})"
        )
    return last_capture


def make_sam_predictor():
    if not args.run_sam:
        return None
    model_path = Path(args.model_path)
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
            f"{', '.join(missing_modules)}. Install them into "
            f"{EXTRA_SITE_PACKAGES} with Isaac Python, for example:\n"
            f'& "C:\\isaacsim\\python.bat" -m pip install --target "{EXTRA_SITE_PACKAGES}" --no-deps timm\n'
            "Use --no-deps so pip does not install another Torch/NumPy next to Isaac's runtime."
        )
    from ultralytics.models.sam import SAM3SemanticPredictor

    overrides = {
        "conf": args.conf,
        "task": "segment",
        "mode": "predict",
        "model": str(model_path),
        "save": False,
        "verbose": False,
    }
    if args.device is not None:
        overrides["device"] = args.device
    return SAM3SemanticPredictor(overrides=overrides)


def extract_sam_masks(results):
    if not results or results[0].masks is None or results[0].masks.data is None:
        return []
    masks = results[0].masks.data
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    return [np.asarray(mask, dtype=bool) for mask in masks]


def compute_mask_metrics(pred_mask, gt_mask):
    pred = np.asarray(pred_mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)
    intersection = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    pred_pixels = int(pred.sum())
    gt_pixels = int(gt.sum())
    precision = intersection / pred_pixels if pred_pixels else 0.0
    recall = intersection / gt_pixels if gt_pixels else 0.0
    iou = intersection / union if union else 0.0
    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "pred_pixels": pred_pixels,
        "gt_pixels": gt_pixels,
    }


def run_sam_on_scene(predictor, scene_dir, prompt, gt_mask):
    rgb_path = scene_dir / "rgb.png"
    predictor.set_image(str(rgb_path))
    results = predictor(text=[prompt])
    masks = extract_sam_masks(results)
    if not masks:
        return {"ran": True, "mask_count": 0, "error": "no_masks"}

    metrics_by_mask = [compute_mask_metrics(mask, gt_mask) for mask in masks]
    best_index = int(np.argmax([metrics["iou"] for metrics in metrics_by_mask]))
    best_mask = masks[best_index]
    save_mask_png(best_mask, scene_dir / "sam_mask.png")
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    save_overlay(rgb, best_mask, scene_dir / "sam_overlay.png")
    result = {
        "ran": True,
        "prompt": prompt,
        "mask_count": len(masks),
        "selected_mask_index": best_index,
    }
    result.update(metrics_by_mask[best_index])
    return result


def main():
    usd_path = Path(args.usd_path)
    if not usd_path.exists():
        raise FileNotFoundError(usd_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)
    open_stage(str(usd_path))
    world = World(stage_units_in_meters=1.0)
    world.reset()

    stage = omni.usd.get_context().get_stage()
    camera_prim = find_camera_prim(stage, args.camera_prim_path)
    camera_path = str(camera_prim.GetPath())
    render_product = rep.create.render_product(camera_path, (args.width, args.height), name="sam3_benchmark_capture")
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    semantic_annotator = rep.AnnotatorRegistry.get_annotator(
        "semantic_segmentation", init_params={"semanticTypes": ["class"], "colorize": False}
    )
    for annotator in (rgb_annotator, depth_annotator, semantic_annotator):
        annotator.attach([render_product])

    for _ in range(args.warmup_frames):
        step_world(world)

    predictor = make_sam_predictor()
    scene_records = []
    total_scenes = len(args.objects) * args.scenes_per_object
    scene_counter = 0

    for object_name in args.objects:
        for index in range(args.scenes_per_object):
            scene_counter += 1
            scene_id = f"{object_name}_{index:04d}"
            scene_dir = output_dir / object_name / scene_id
            scene_dir.mkdir(parents=True, exist_ok=True)

            object_record = spawn_object(object_name, rng)
            step_world(world, args.settle_steps, render=True)
            rgb, depth, semantic_data = capture_annotators(world, rgb_annotator, depth_annotator, semantic_annotator)
            rgb_mean, rgb_min, rgb_max, rgb_std = visible_rgb_stats(rgb)
            rgb_is_blank = not is_valid_rgb_frame(rgb)
            gt_mask, target_ids, id_to_labels = semantic_mask_from_data(semantic_data)

            save_rgb_png(rgb, scene_dir / "rgb.png")
            np.save(scene_dir / "depth_meters.npy", depth)
            np.save(scene_dir / "gt_mask.npy", gt_mask.astype(np.uint8))
            save_mask_png(gt_mask, scene_dir / "gt_mask.png")
            save_overlay(rgb, gt_mask, scene_dir / "gt_overlay.png")

            metadata = {
                "scene_id": scene_id,
                "scene_index": scene_counter,
                "total_scenes": total_scenes,
                "object": object_record,
                "camera": get_camera_metadata(camera_prim, args.width, args.height),
                "semantic_target_label": TARGET_LABEL,
                "semantic_target_ids": target_ids,
                "gt_mask_pixels": int(gt_mask.sum()),
                "rgb_is_blank": bool(rgb_is_blank),
                "rgb_mean": rgb_mean,
                "rgb_min": rgb_min,
                "rgb_max": rgb_max,
                "rgb_std": rgb_std,
                "id_to_labels": id_to_labels,
            }

            sam_metrics = {"ran": False}
            if rgb_is_blank:
                sam_metrics = {"ran": False, "error": "blank_rgb"}
            elif predictor is not None:
                sam_metrics = run_sam_on_scene(predictor, scene_dir, object_record["prompt"], gt_mask)
            metadata["sam3"] = sam_metrics

            with open(scene_dir / "metadata.json", "w", encoding="utf-8") as file:
                json.dump(metadata, file, indent=2)

            scene_records.append(
                {
                    "scene_id": scene_id,
                    "object": object_name,
                    "path": str(scene_dir),
                    "gt_mask_pixels": int(gt_mask.sum()),
                    "sam3": sam_metrics,
                }
            )
            print(
                f"[{scene_counter:03d}/{total_scenes:03d}] {scene_id}: "
                f"gt_pixels={int(gt_mask.sum())}, sam_iou={sam_metrics.get('iou', 'not_run')}"
            )

    summary = {
        "usd_path": str(usd_path),
        "camera_prim_path": camera_path,
        "objects": args.objects,
        "scenes_per_object": args.scenes_per_object,
        "run_sam": args.run_sam,
        "records": scene_records,
    }
    if args.run_sam:
        valid = [record["sam3"] for record in scene_records if record["sam3"].get("iou") is not None]
        if valid:
            summary["sam3_mean_iou"] = float(np.mean([item["iou"] for item in valid]))
            summary["sam3_mean_precision"] = float(np.mean([item["precision"] for item in valid]))
            summary["sam3_mean_recall"] = float(np.mean([item["recall"] for item in valid]))
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(f"Wrote dataset summary to {output_dir / 'summary.json'}")


try:
    main()
except Exception:
    print("Fatal error while generating SAM 3 RGB-D benchmark:")
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
