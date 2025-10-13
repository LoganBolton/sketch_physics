"""Generate semi-random PHYRE scenes with tracked trajectories."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
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
    parser.add_argument("--steps", type=int, default=1000,
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


def _build_random_scene(args: argparse.Namespace) -> Tuple[creator_lib.TaskCreator, List[float], float, int, Tuple[float, float]]:
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    creator = creator_lib.TaskCreator()
    creator.scene.width = 512
    creator.scene.height = 512

    # Remove the old boundary walls (they were created with old dimensions)
    creator.body_list = [body for body in creator.body_list if 'wall' not in body.object_type]
    creator.scene.bodies = [body._thrift_body for body in creator.body_list]

    bucket_centers, bucket_top = _add_buckets(creator, max(1, args.num_buckets))

    # Add thin side walls to prevent ball from bouncing out
    wall_thickness = 2  # Very thin, barely visible
    wall_height = creator.scene.height

    # Left wall
    left_wall = creator.add_box(width=wall_thickness, height=wall_height, dynamic=False)
    left_wall.set_center(wall_thickness / 2, wall_height / 2)
    left_wall.set_color("black")

    # Right wall
    right_wall = creator.add_box(width=wall_thickness, height=wall_height, dynamic=False)
    right_wall.set_center(creator.scene.width - wall_thickness / 2, wall_height / 2)
    right_wall.set_color("black")

    # Determine number of bars: cycle through 1, 2, 3 based on run number
    if hasattr(args, 'run_number'):
        num_bars = ((args.run_number - 1) % 3) + 1
    else:
        num_bars = random.choice([1, 2, 3])

    # Divide the space above buckets into 3 vertical sections
    # Reserve 100 pixels at the top for the ball (ball is at height - 20, so we need 120 total buffer)
    scene_height = creator.scene.height
    ball_buffer = 120  # 100 pixels between line and ball, plus 20 for ball position
    available_height = scene_height - bucket_top - ball_buffer
    section_height = available_height / 3

    # Define height ranges for each bar (from bottom to top)
    # Each section gets equal height with 20px padding between sections
    height_sections = [
        (bucket_top + 20, bucket_top + section_height - 10),           # Bottom section
        (bucket_top + section_height + 10, bucket_top + 2 * section_height - 10),  # Middle section
        (bucket_top + 2 * section_height + 10, bucket_top + 3 * section_height - 10),  # Top section
    ]

    # Select which sections to use based on number of bars
    if num_bars == 1:
        selected_sections = [height_sections[1]]  # Use middle section
    elif num_bars == 2:
        selected_sections = [height_sections[0], height_sections[2]]  # Use bottom and top
    else:  # num_bars == 3
        selected_sections = height_sections  # Use all three

    # Random static bars in designated sections
    # Add horizontal margins to shrink the available width for lines
    horizontal_margin = 60  # 60 pixels margin on each side
    available_width = creator.scene.width - 2 * horizontal_margin

    max_bar_width = available_width * 0.8
    min_bar_width = available_width * 0.5
    MAX_ANGLE = 20
    MIN_ANGLE = 5

    bars_created = 0
    previous_angle = None  # Track the previous bar's angle for sequential placement

    for section_idx, (min_cy, max_cy) in enumerate(selected_sections):
        width = random.uniform(min_bar_width, max_bar_width)
        height = 4
        bar = creator.add_box(width=width, height=height, dynamic=False)

        # Determine angle and position based on whether this is the first bar
        min_cx = horizontal_margin + width / 2
        max_cx = creator.scene.width - horizontal_margin - width / 2
        scene_third = available_width / 3

        if section_idx == 0:
            # First bar: random angle and position
            if random.random() < 0.5:
                angle = random.uniform(MIN_ANGLE, MAX_ANGLE)
            else:
                angle = random.uniform(-MAX_ANGLE, -MIN_ANGLE)
            cx = random.uniform(min_cx, max_cx)
        else:
            # Subsequent bars: angle opposite to previous, position based on deflection
            # If previous angle was positive (tilts right), ball deflects left
            # If previous angle was negative (tilts left), ball deflects right
            if previous_angle > 0:
                # Previous bar tilted right -> ball deflects left -> negative angle on left side
                angle = random.uniform(-MAX_ANGLE, -MIN_ANGLE)
                cx = random.uniform(min_cx, horizontal_margin + scene_third - width / 2)
            else:
                # Previous bar tilted left -> ball deflects right -> positive angle on right side
                angle = random.uniform(MIN_ANGLE, MAX_ANGLE)
                cx = random.uniform(horizontal_margin + 2 * scene_third + width / 2, max_cx)

        # Calculate vertical extent and ensure bar fits in section
        angle_rad = math.radians(angle)
        vertical_extent = (abs(width / 2 * math.sin(angle_rad)) +
                           abs(height / 2 * math.cos(angle_rad)))

        # Adjust bounds to account for rotation
        adjusted_min = min_cy + vertical_extent
        adjusted_max = max_cy - vertical_extent

        if adjusted_max > adjusted_min:
            cy = random.uniform(adjusted_min, adjusted_max)
        else:
            cy = (min_cy + max_cy) / 2

        bar.set_center(cx, cy).set_angle(angle)
        bar.set_color("black")
        bars_created += 1
        previous_angle = angle  # Store for next iteration

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

    # Dynamic ball near the top with normal distribution centered in the middle
    ball = creator.add("dynamic ball", scale=args.ball_radius)

    # Use normal distribution centered at the middle of the scene
    scene_center_x = creator.scene.width / 2
    # Standard deviation is ~1/6 of the width so ~99.7% of values fall within the scene
    std_dev = creator.scene.width / 6

    # Sample from normal distribution and clamp to valid range
    ball_x = np.random.normal(scene_center_x, std_dev)
    ball_x = _clamp(ball_x, 30, creator.scene.width - 30)

    ball_y = creator.scene.height - 20
    ball.set_center(ball_x, ball_y)
    ball.set_color("blue")

    creator.update_task(
        body1=ball,
        body2=creator.body_list[0],  # arbitrary static wall (bottom)
        relationships=[creator.SpatialRelationship.ABOVE],
    )
    creator.set_meta(creator.SolutionTier.GENERAL)

    return creator, bucket_centers, bucket_top, bars_created, (ball_x, ball_y)


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

        if i > 0:
            left = creator.add_box(width=wall_thickness,
                                   height=bucket_height,
                                   dynamic=False)
            left_x = center_x - (bucket_width / 2) + wall_thickness / 2
            if last_right_x is not None:
                left_x = last_right_x  # remove horizontal gap
            left.set_center(left_x, bucket_height / 2)
            left.set_color("purple")

        if i < count - 1:
            right = creator.add_box(width=wall_thickness,
                                    height=bucket_height,
                                    dynamic=False)
            right_x = center_x + (bucket_width / 2) - wall_thickness / 2
            right.set_center(right_x, bucket_height / 2)
            right.set_color("purple")
            last_right_x = right_x
        centers.append(center_x)

    bucket_top = bucket_height
    return centers, bucket_top


def _process_single_run(run: int, args: argparse.Namespace, base_output: pathlib.Path) -> None:
    """Process a single run - designed to be called in parallel."""
    if args.runs > 1:
        output_dir = base_output / f"run_{run:03d}"
    else:
        output_dir = base_output
    output_dir.mkdir(parents=True, exist_ok=True)

    run_seed = (args.seed + run - 1) if args.seed is not None else None
    run_args = argparse.Namespace(**vars(args))
    run_args.seed = run_seed
    run_args.run_number = run

    creator, bucket_centers, bucket_top, line_count, ball_start = _build_random_scene(run_args)

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

    simulation_info = {
        "steps_requested": args.steps,
        "frames_recorded": num_frames,
        "frame_stride": stride,
        "requested_fps": fps,
        "playback_speedup": speed,
        "effective_fps": effective_fps,
        "num_lines": line_count,
        "start_ball_x": ball_start[0],
        "start_ball_y": ball_start[1],
        "final_ball_x": final_x,
        "final_ball_y": final_y,
        "bucket_hit": bucket_hit,
        "solved": None,
    }

    metadata = {
        "simulation": simulation_info,
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
        "seed": run_seed,
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

    print(f"Run {run}/{args.runs} completed")
    print(f"  Scene metadata written to {metadata_path}")
    print(f"  Animation written to {video_path}")
    print(f"  Start frame saved to {start_snapshot}")
    print(f"  Final frame saved to {final_snapshot}")


def main(args: argparse.Namespace) -> None:
    base_output = args.output_dir
    base_output.mkdir(parents=True, exist_ok=True)

    # Limit the number of parallel processes to avoid memory issues
    # Use at most 6 processes or the number of CPU cores, whichever is smaller
    num_processes = min(6, multiprocessing.cpu_count())

    print(f"Starting parallel generation of {args.runs} runs using {num_processes} processes...")

    # Use multiprocessing with a limited pool size and process in chunks
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Process runs in chunks to avoid memory buildup
        chunk_size = num_processes
        for i in range(0, args.runs, chunk_size):
            chunk_runs = range(i + 1, min(i + chunk_size + 1, args.runs + 1))
            pool.starmap(
                _process_single_run,
                [(run, args, base_output) for run in chunk_runs]
            )
            print(f"Completed {min(i + chunk_size, args.runs)}/{args.runs} runs")

    print(f"\nAll {args.runs} runs completed!")


if __name__ == "__main__":
    main(parse_args())
