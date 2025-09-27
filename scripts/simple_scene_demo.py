"""Generate a tiny PHYRE scene, simulate it, and export metadata + animation."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, Iterable, List, Tuple

import imageio.v3 as imageio
import imageio.v2 as imageio_v2
import imageio_ffmpeg  # noqa: F401 ensures ffmpeg plugin is registered
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
GEN_PY = REPO_ROOT / "cmake_build/gen-py"
SRC_PY = REPO_ROOT / "src/python"

if SRC_PY.exists():
    sys.path.insert(0, str(SRC_PY))
if GEN_PY.exists():
    sys.path.insert(0, str(GEN_PY))

from phyre.creator import constants as creator_constants
from phyre.creator import creator as creator_lib
from phyre import simulator
from phyre.interface.scene import ttypes as scene_if


OUTPUT_DIR = pathlib.Path("output/simple_scene")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _body_shape_summary(body) -> Dict[str, object]:
    """Convert the thrift shape list into a JSON-friendly summary."""
    summary: List[Dict[str, object]] = []
    for shape in body.shapes:
        if shape.circle:
            summary.append({
                "type": "circle",
                "radius": shape.circle.radius,
            })
        elif shape.polygon:
            summary.append({
                "type": "polygon",
                "vertices": [[v.x, v.y] for v in shape.polygon.vertices],
            })
        else:
            summary.append({"type": "unknown"})
    return {"shapes": summary, "angle": body.angle}


def _summarize_scene_bodies(scene) -> List[Dict[str, object]]:
    bodies: List[Dict[str, object]] = []
    for idx, body in enumerate(scene.bodies):
        color_id = body.color if body.color is not None else 0
        try:
            color_name = creator_constants.color_to_name(color_id).lower()
        except KeyError:
            color_name = "unknown"
        shape_info = _body_shape_summary(body)
        body_type_name = scene_if.BodyType._VALUES_TO_NAMES.get(
            body.bodyType, str(body.bodyType)
        )
        shape_type_name = None
        if body.shapeType is not None:
            shape_type_name = scene_if.ShapeType._VALUES_TO_NAMES.get(
                body.shapeType, str(body.shapeType)
            )

        bodies.append({
            "index": idx,
            "bodyType": body_type_name,
            "color": {
                "id": color_id,
                "name": color_name,
            },
            "position": {
                "x": body.position.x,
                "y": body.position.y,
            },
            "dynamic": body_type_name == "DYNAMIC",
            "shapeType": shape_type_name,
            "angle": shape_info["angle"],
            "shapes": shape_info["shapes"],
        })
    return bodies


def _collect_trajectories(
    scenes: Iterable,
) -> Dict[int, List[Tuple[float, float]]]:
    """Capture per-body dynamic trajectories across all frames."""

    scenes = list(scenes)
    if not scenes:
        return {}

    dynamic_indices = [
        idx for idx, body in enumerate(scenes[0].bodies)
        if body.bodyType == scene_if.BodyType.DYNAMIC
    ]
    trajectories: Dict[int, List[Tuple[float, float]]] = {
        idx: [] for idx in dynamic_indices
    }
    for scene in scenes:
        for idx in dynamic_indices:
            body = scene.bodies[idx]
            trajectories[idx].append((body.position.x, body.position.y))
    return trajectories


def _draw_polyline(image: np.ndarray, points: List[Tuple[int, int]], color: Tuple[int, int, int]) -> None:
    """Draw a polyline on the image."""

    if len(points) < 2:
        if points:
            r, c = points[0]
            image[r, c] = color
        return
    h, w, _ = image.shape
    for (r0, c0), (r1, c1) in zip(points[:-1], points[1:]):
        steps = int(max(abs(r1 - r0), abs(c1 - c0)))
        if steps == 0:
            rr = np.array([r0])
            cc = np.array([c0])
        else:
            rr = np.linspace(r0, r1, steps + 1).round().astype(int)
            cc = np.linspace(c0, c1, steps + 1).round().astype(int)
        rr = np.clip(rr, 0, h - 1)
        cc = np.clip(cc, 0, w - 1)
        image[rr, cc] = color


def _generate_frames(
    scenes: Iterable,
    *,
    scale: int,
    frame_stride: int,
    trajectories: Dict[int, List[Tuple[float, float]]],
) -> Tuple[List[np.ndarray], int]:
    """Convert simulator scenes into RGB frames with trajectory overlays."""

    palette: Dict[int, Tuple[int, int, int]] = {
        creator_constants.ROLE_TO_COLOR_ID["BACKGROUND"]: (255, 255, 255),
        creator_constants.ROLE_TO_COLOR_ID["STATIC"]: (0, 0, 0),
        creator_constants.ROLE_TO_COLOR_ID["DYNAMIC"]: (128, 128, 128),
        creator_constants.ROLE_TO_COLOR_ID["DYNAMIC_OBJECT"]: (0, 170, 255),
        creator_constants.ROLE_TO_COLOR_ID["DYNAMIC_SUBJECT"]: (0, 120, 0),
        creator_constants.ROLE_TO_COLOR_ID["STATIC_OBJECT"]: (170, 0, 255),
        creator_constants.ROLE_TO_COLOR_ID["USER_BODY"]: (255, 128, 0),
    }
    palette.setdefault(0, (255, 255, 255))

    scenes = list(scenes)
    sampled_indices = range(0, len(scenes), frame_stride)
    frames: List[np.ndarray] = []
    height = scenes[0].height if scenes else 0
    width = scenes[0].width if scenes else 0

    for scene_idx in sampled_indices:
        scene = scenes[scene_idx]
        raster = simulator.scene_to_raster(scene)
        rgb = np.zeros((raster.shape[0], raster.shape[1], 3), dtype=np.uint8)
        for color_id, rgb_value in palette.items():
            mask = raster == color_id
            rgb[mask] = rgb_value

        rgb = np.flipud(rgb)

        if scale > 1:
            rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)

        if trajectories:
            for body_idx, positions in trajectories.items():
                upto = positions[:scene_idx + 1]
                pts: List[Tuple[int, int]] = []
                for x, y in upto:
                    col = int(round(x))
                    row = int(round(height - 1 - y))
                    if 0 <= col < width and 0 <= row < height:
                        if scale > 1:
                            pts.append((row * scale, col * scale))
                        else:
                            pts.append((row, col))
                if pts:
                    _draw_polyline(rgb, pts, (255, 0, 0))

        frames.append(rgb)
    return frames, len(frames)


def _write_animation(
    frames: List[np.ndarray],
    *,
    output_dir: pathlib.Path,
    fps: int,
    frame_stride: int,
    playback_speed: float,
    video_format: str,
) -> Tuple[pathlib.Path, float]:
    """Serialize frames to disk and return the output path and effective FPS."""

    effective_speed = playback_speed if playback_speed > 0 else 1.0
    if video_format == "mp4":
        effective_fps = fps * effective_speed / frame_stride
        video_path = output_dir / "simple_scene.mp4"
        try:
            with imageio_v2.get_writer(
                video_path,
                format="FFMPEG",
                fps=effective_fps,
                codec="libx264",
            ) as writer:
                for frame in frames:
                    writer.append_data(frame)
        except Exception as exc:
            raise RuntimeError(
                "Writing MP4 requires the imageio-ffmpeg plugin and an ffmpeg binary."
            ) from exc
        png_path = output_dir / "simple_scene_final.png"
        imageio.imwrite(png_path, frames[-1])
        return video_path, effective_fps

    duration = frame_stride / (fps * effective_speed)
    if duration < 0.01:
        print(
            "Warning: GIF frame duration < 10ms; many viewers clamp to 10ms. "
            "Use --video-format mp4 for true high-speed playback."
        )
    gif_path = output_dir / "simple_scene.gif"
    imageio.imwrite(gif_path, frames, duration=max(duration, 0.01), loop=0)
    gif_fps = 1.0 / max(duration, 0.01)
    # Also dump the final frame as a PNG snapshot.
    png_path = output_dir / "simple_scene_final.png"
    imageio.imwrite(png_path, frames[-1])
    return gif_path, gif_fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a simple PHYRE scene.")
    parser.add_argument("--pixel-scale", type=int, default=4,
                        help="Factor to upsample the rendered frames (default: 4)")
    parser.add_argument("--fps", type=int, default=60,
                        help="Playback frames per second (default: 60)")
    parser.add_argument("--frame-stride", type=int, default=2,
                        help="Keep every Nth physics frame (default: 2)")
    parser.add_argument("--playback-speed", type=float, default=20.0,
                        help="Additional playback speed multiplier (default: 20.0)")
    parser.add_argument("--steps", type=int, default=240,
                        help="Number of physics steps to simulate (default: 240)")
    parser.add_argument("--ball-start-y", type=float, default=240.0,
                        help="Ball centre Y coordinate before release (default: 240)")
    parser.add_argument("--output-dir", type=pathlib.Path, default=OUTPUT_DIR,
                        help="Directory to write outputs (default: output/simple_scene)")
    parser.add_argument("--video-format", choices=("gif", "mp4"), default="gif",
                        help="Output animation format (default: gif)")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    creator = creator_lib.TaskCreator()

    platform = creator.add("static bar", scale=0.6)
    platform.set_center(150, 120).set_angle(25)

    ball = creator.add("dynamic ball", scale=0.12)
    ball.set_center(90, args.ball_start_y)
    ball.set_color("blue")

    creator.update_task(
        body1=ball,
        body2=platform,
        relationships=[creator.SpatialRelationship.TOUCHING],
    )
    creator.set_meta(creator.SolutionTier.SINGLE_OBJECT)

    simulation = simulator.simulate_task(creator.task, steps=args.steps, stride=1)
    pixel_scale = max(args.pixel_scale, 1)
    fps = max(args.fps, 1)
    stride = max(args.frame_stride, 1)
    speed = max(args.playback_speed, 0.01)
    trajectories = _collect_trajectories(simulation.sceneList)
    frames, num_sampled = _generate_frames(
        simulation.sceneList,
        scale=pixel_scale,
        frame_stride=stride,
        trajectories=trajectories,
    )
    video_path, effective_fps = _write_animation(
        frames,
        output_dir=output_dir,
        fps=fps,
        frame_stride=stride,
        playback_speed=speed,
        video_format=args.video_format,
    )

    relationship_value = creator.task.relationships[0]
    relationship_name = creator.SpatialRelationship._VALUES_TO_NAMES.get(
        relationship_value, str(relationship_value)
    )

    metadata = {
        "tier": creator.task.tier,
        "relationship": relationship_name,
        "scene": {
            "width": creator.scene.width,
            "height": creator.scene.height,
            "num_bodies": len(creator.scene.bodies),
            "bodies": _summarize_scene_bodies(creator.scene),
        },
        "simulation": {
            "steps_requested": args.steps,
            "frames_recorded": num_sampled,
            "frame_stride": stride,
            "requested_fps": fps,
            "playback_speedup": speed,
            "effective_fps": effective_fps,
            "solved": simulation.isSolution,
        },
        "outputs": {
            "path": str(video_path),
            "format": args.video_format,
            "snapshot": str(output_dir / "simple_scene_final.png"),
        },
        "render": {
            "pixel_scale": pixel_scale,
        },
        "ball_start": {
            "x": 90,
            "y": args.ball_start_y,
        },
        "trajectories": {
            str(idx): [{"x": x, "y": y} for (x, y) in positions]
            for idx, positions in trajectories.items()
        },
    }

    metadata_path = output_dir / "simple_scene_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Scene metadata written to", metadata_path)
    print("Animation written to", video_path)
    print("Snapshot written to", output_dir / "simple_scene_final.png")


if __name__ == "__main__":
    main(parse_args())
