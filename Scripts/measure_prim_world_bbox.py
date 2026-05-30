import argparse
from pathlib import Path

from isaacsim import SimulationApp


def main():
    parser = argparse.ArgumentParser(description="Measure a prim world-space bounding box in an Isaac USD scene.")
    parser.add_argument("--usd-path", default=str(Path(__file__).resolve().parents[1] / "assets" / "scene.usd"))
    parser.add_argument("--prim-path", default=None)
    parser.add_argument("--list-candidates", action="store_true")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    simulation_app = SimulationApp({"headless": args.headless})

    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(args.usd_path)
    def measure_prim(prim):
        combined = None
        for child in Usd.PrimRange(prim):
            if not child.IsA(UsdGeom.Boundable):
                continue
            box = bbox_cache.ComputeWorldBound(child).ComputeAlignedBox()
            if box.IsEmpty():
                continue
            if combined is None:
                combined = Gf.Range3d(box.GetMin(), box.GetMax())
            else:
                combined.UnionWith(Gf.Range3d(box.GetMin(), box.GetMax()))
        return combined

    if args.list_candidates:
        keywords = ("container", "tray", "bin")
        for candidate in stage.Traverse():
            path = str(candidate.GetPath())
            if not any(keyword in path.lower() for keyword in keywords):
                continue
            box = measure_prim(candidate)
            if box is None or box.IsEmpty():
                continue
            bbox_min = box.GetMin()
            bbox_max = box.GetMax()
            center = (bbox_min + bbox_max) * 0.5
            size = bbox_max - bbox_min
            print(
                f"{path}: center=[{center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}], "
                f"size=[{size[0]:.6f}, {size[1]:.6f}, {size[2]:.6f}], "
                f"min=[{bbox_min[0]:.6f}, {bbox_min[1]:.6f}, {bbox_min[2]:.6f}], "
                f"max=[{bbox_max[0]:.6f}, {bbox_max[1]:.6f}, {bbox_max[2]:.6f}]"
            )
        simulation_app.close()
        return

    if not args.prim_path:
        raise ValueError("Pass --prim-path /Some/Prim or use --list-candidates.")

    prim = stage.GetPrimAtPath(args.prim_path)
    if not prim:
        raise ValueError(f"Prim not found: {args.prim_path}")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )

    combined = measure_prim(prim)
    if combined is None or combined.IsEmpty():
        raise ValueError(f"No boundable geometry under: {args.prim_path}")

    bbox_min = combined.GetMin()
    bbox_max = combined.GetMax()
    center = (bbox_min + bbox_max) * 0.5
    size = bbox_max - bbox_min
    print(f"prim_path: {args.prim_path}")
    print(f"world_min: [{bbox_min[0]:.6f}, {bbox_min[1]:.6f}, {bbox_min[2]:.6f}]")
    print(f"world_max: [{bbox_max[0]:.6f}, {bbox_max[1]:.6f}, {bbox_max[2]:.6f}]")
    print(f"world_center: [{center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}]")
    print(f"world_size: [{size[0]:.6f}, {size[1]:.6f}, {size[2]:.6f}]")
    print(f"suggested_outer_xy_size: [{size[0]:.6f}, {size[1]:.6f}]")
    print(f"suggested_inner_xy_size_margin_0.05: [{max(size[0] - 0.10, 0.0):.6f}, {max(size[1] - 0.10, 0.0):.6f}]")

    simulation_app.close()


if __name__ == "__main__":
    main()
