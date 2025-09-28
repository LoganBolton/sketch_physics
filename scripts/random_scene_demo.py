"""Generate semi-random PHYRE scenes with tracked trajectories."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
from typing import Iterable, List, Tuple

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
GEN_PY = REPO_ROOT / "cmake_build/gen-py"
SRC_PY = REPO_ROOT / "src/python"

if SRC_PY.exists():
    sys.path.insert(0, str(SRC_PY))
if GEN_PY.exists():
    sys.path.insert(0, str(GEN_PY))

from phyre import simulator  # noqa: E402
from phyre.creator import creator as creator_lib  # noqa: E402
from phyre.creator import constants as creator_constants  # noqa: E402
from phyre.creator import shapes as shapes_lib  # noqa: E402
from phyre.interface.scene import ttypes as scene_if  # noqa: E402

import simple_scene_demo as renderer  # noqa: E402

OUTPUT_DIR = pathlib.Path("output/random_scene")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a random PHYRE scene with trajectory overlays.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: None -> system entropy)")
    parser.add_argument("--pixel-scale", type=int, default=4,
                        help="Upsampling factor for output frames (default: 4)")
    parser.add_argument("--fps", type=int, default=60,
                        help="Playback frames-per-second (default: 60)")
    parser.add_argument("--frame-stride", type=int, default=2,
                        help="Keep every Nth physics frame (default: 2)")
    parser.add_argument("--playback-speed", type=float, default=10.0,
                        help="Playback speed multiplier (default: 10)")
    parser.add_argument("--steps", type=int, default=500,
                        help="Number of physics steps to simulate (default: 500)")
    parser.add_argument("--num-bars", type=int, default=4,
                        help="Number of random static bars to add (default: 4)")
    parser.add_argument("--num-polys", type=int, default=0,
                        help="Number of random convex polygons to add (default: 0)")
    parser.add_argument("--runs", type=int, default=1,
                        help="How many scenes to generate (default: 1)")
    parser.add_argument("--num-buckets", type=int, default=4,
                        help="Number of static buckets along the bottom (default: 4)")
    parser.add_argument("--ball-radius", type=float, default=0.1,
                        help="Radius scale for the dynamic ball (default: 0.1)")
    parser.add_argument("--output-dir", type=pathlib.Path, default=OUTPUT_DIR,
                        help="Directory for outputs (default: output/random_scene)")
    parser.add_argument("--video-format", choices=("gif", "mp4"), default="gif",
                        help="Output animation format (default: gif)")
    return parser.parse_args()


def _random_convex_polygon(center_x: float, center_y: float, radius: float,
                           num_vertices: int) -> List[Tuple[float, float]]:
    angles = sorted(random.uniform(0, 2 * math.pi) for _ in range(num_vertices))
    vertices = []
    for angle in angles:
        r = radius * random.uniform(0.6, 1.0)
        x = center_x + r * math.cos(angle)
        y = center_y + r * math.sin(angle)
        vertices.append((x, y))
    return vertices


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _build_random_scene(args: argparse.Namespace) -> Tuple[creator_lib.TaskCreator, List[float], float]:
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    creator = creator_lib.TaskCreator()

    bucket_centers, bucket_top = _add_buckets(creator, max(1, args.num_buckets))

    # Random static bars.
    safe_height = creator.scene.height - 80
    max_bar_width = creator.scene.width / 3
    for _ in range(max(0, args.num_bars)):
        width = random.uniform(30, max_bar_width)
        height = random.uniform(4, 10)
        bar = creator.add_box(width=width, height=height, dynamic=False)
        cx = random.uniform(width / 2 + 10, creator.scene.width - width / 2 - 10)
        angle = random.uniform(-60, 60)
        angle_rad = math.radians(angle)
        vertical_extent = (abs(width / 2 * math.sin(angle_rad)) +
                           abs(height / 2 * math.cos(angle_rad)))
        min_cy = bucket_top + vertical_extent + 10
        max_cy = max(min_cy + 1, safe_height)
        cy = random.uniform(min_cy, max_cy)
        bar.set_center(cx, cy).set_angle(angle)
        bar.set_color("black")

    # Random polygons.
    for _ in range(max(0, args.num_polys)):
        num_vertices = random.randint(3, 6)
        max_radius = min(50, safe_height - bucket_top - 30)
        radius = random.uniform(20, max_radius)
        cx = random.uniform(radius + 10, creator.scene.width - radius - 10)
        cy = random.uniform(bucket_top + radius + 20, safe_height)
        for _attempt in range(10):
            vertices = _random_convex_polygon(cx, cy, radius, num_vertices)
            poly_vectors = [scene_if.Vector(x, y) for x, y in vertices]
            if shapes_lib.is_valid_convex_polygon(poly_vectors):
                break
        else:
            continue
        polygon = creator.add_convex_polygon(vertices, dynamic=False)
        polygon.set_color("black")

    # Dynamic ball near the top.
    ball = creator.add("dynamic ball", scale=args.ball_radius)
    bucket_width = creator.scene.width / max(1, len(bucket_centers))
    center = random.choice(bucket_centers)
    offset = random.uniform(-0.3 * bucket_width, 0.3 * bucket_width)
    ball_x = _clamp(center + offset, 30, creator.scene.width - 30)
    ball_y = creator.scene.height - 20
    ball.set_center(ball_x, ball_y)
    ball.set_color("blue")

    creator.update_task(
        body1=ball,
        body2=creator.body_list[0],  # arbitrary static wall (bottom)
        relationships=[creator.SpatialRelationship.ABOVE],
    )
    creator.set_meta(creator.SolutionTier.GENERAL)

    return creator, bucket_centers, bucket_top


def _add_buckets(creator: creator_lib.TaskCreator, count: int) -> Tuple[List[float], float]:
    width = creator.scene.width
    bucket_width = width / count
    wall_thickness = 3
    bucket_height = 60
    base_height = 6

    last_right_x = None
    centers: List[float] = []

    for i in range(count):
        center_x = (i + 0.5) * bucket_width

        # Base (gray)
        base = creator.add_box(width=bucket_width,
                               height=base_height,
                               dynamic=False)
        base.set_center(center_x, base_height / 2)
        base.set_color("purple")

        if i > 0:
            left = creator.add_box(width=wall_thickness,
                                   height=bucket_height,
                                   dynamic=False)
            left_x = center_x - (bucket_width / 2) + wall_thickness / 2
            if last_right_x is not None:
                left_x = last_right_x  # remove horizontal gap
            left.set_center(left_x, bucket_height / 2 + base_height)
            left.set_color("purple")

        if i < count - 1:
            right = creator.add_box(width=wall_thickness,
                                    height=bucket_height,
                                    dynamic=False)
            right_x = center_x + (bucket_width / 2) - wall_thickness / 2
            right.set_center(right_x, bucket_height / 2 + base_height)
            right.set_color("purple")
            last_right_x = right_x
        centers.append(center_x)

    bucket_top = bucket_height + base_height
    return centers, bucket_top

def main(args: argparse.Namespace) -> None:
    base_output = args.output_dir
    base_output.mkdir(parents=True, exist_ok=True)

    for run in range(1, args.runs + 1):
        if args.runs > 1:
            output_dir = base_output / f"run_{run:03d}"
        else:
            output_dir = base_output
        output_dir.mkdir(parents=True, exist_ok=True)

        run_seed = (args.seed + run - 1) if args.seed is not None else None
        run_args = argparse.Namespace(**vars(args))
        run_args.seed = run_seed

        creator, bucket_centers, bucket_top = _build_random_scene(run_args)

        scene_frames = simulator.simulate_scene(creator.scene, args.steps)

        pixel_scale = max(args.pixel_scale, 1)
        fps = max(args.fps, 1)
        stride = max(args.frame_stride, 1)
        speed = max(args.playback_speed, 0.01)

        trajectories = renderer._collect_trajectories(scene_frames)
        label_specs = [(center, str(i + 1)) for i, center in enumerate(bucket_centers)]
        frames, num_frames = renderer._generate_frames(
            scene_frames,
            scale=pixel_scale,
            frame_stride=stride,
            trajectories=trajectories,
        )

        video_path, effective_fps = renderer._write_animation(
            frames,
            output_dir=output_dir,
            fps=fps,
            frame_stride=stride,
            playback_speed=speed,
            video_format=args.video_format,
            pixel_scale=pixel_scale,
            labels=label_specs,
        )

        desired_video = output_dir / ("random_scene.gif" if args.video_format == "gif" else "random_scene.mp4")
        if video_path != desired_video:
            video_path.replace(desired_video)
            video_path = desired_video

        default_start = output_dir / "simple_scene_start.png"
        default_final = output_dir / "simple_scene_final.png"
        start_snapshot = output_dir / "random_scene_start.png"
        final_snapshot = output_dir / "random_scene_final.png"
        if default_start.exists():
            default_start.replace(start_snapshot)
        else:
            start_snapshot = default_start
        if default_final.exists():
            default_final.replace(final_snapshot)
        else:
            final_snapshot = default_final

        body_summaries = renderer._summarize_scene_bodies(creator.scene)
        bucket_width = creator.scene.width / max(1, len(bucket_centers))
        bucket_left_edges = [center - bucket_width / 2 for center in bucket_centers]
        bucket_right_edges = [center + bucket_width / 2 for center in bucket_centers]
        ball_index = next(
            idx for idx, body in enumerate(body_summaries)
            if body["bodyType"] == "DYNAMIC"
        )
        final_x, final_y = trajectories[ball_index][-1]
        ball_radius = creator.scene.bodies[ball_index].shapes[0].circle.radius
        bucket_hit = None
        if final_y + ball_radius <= bucket_top:
            for idx, (left, right) in enumerate(zip(bucket_left_edges, bucket_right_edges), start=1):
                if left + ball_radius <= final_x <= right - ball_radius:
                    bucket_hit = idx
                    break

        metadata = {
            "seed": run_seed,
            "scene": {
                "width": creator.scene.width,
                "height": creator.scene.height,
                "num_bodies": len(creator.scene.bodies),
                "bodies": body_summaries,
                "buckets": [
                    {"index": i + 1, "center_x": center}
                    for i, center in enumerate(bucket_centers)
                ],
            },
            "simulation": {
                "steps_requested": args.steps,
                "frames_recorded": num_frames,
                "frame_stride": stride,
                "requested_fps": fps,
                "playback_speedup": speed,
                "effective_fps": effective_fps,
                "solved": None,
                "bucket_hit": bucket_hit,
                "final_ball_x": final_x,
                "final_ball_y": final_y,
            },
            "outputs": {
                "path": str(video_path),
                "format": args.video_format,
                "start_frame": str(start_snapshot),
                "final_frame": str(final_snapshot),
            },
            "render": {
                "pixel_scale": pixel_scale,
            },
            "trajectories": {
                str(idx): [{"x": x, "y": y} for (x, y) in positions]
                for idx, positions in trajectories.items()
            },
        }

        metadata_path = output_dir / "random_scene_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        print(f"Run {run}/{args.runs}")
        print("  Scene metadata written to", metadata_path)
        print("  Animation written to", video_path)
        print("  Start frame saved to", start_snapshot)
        print("  Final frame saved to", final_snapshot)


if __name__ == "__main__":
    main(parse_args())
