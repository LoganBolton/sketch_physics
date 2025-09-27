"""Generate a tiny PHYRE scene, simulate it, and export metadata + animation."""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Dict, Iterable, List, Tuple

import imageio.v3 as imageio
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
GEN_PY = REPO_ROOT / "cmake_build/gen-py"
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


def _frames_to_gif(
    scenes: Iterable,
    *,
    scale: int = 2,
    fps: int = 60,
    frame_stride: int = 2,
) -> Tuple[pathlib.Path, int]:
    """Render scenes to RGB frames and save an animated GIF."""

    def frame_to_rgb(scene) -> np.ndarray:
        raster = simulator.scene_to_raster(scene)
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
        rgb = np.zeros((raster.shape[0], raster.shape[1], 3), dtype=np.uint8)
        for color_id, rgb_value in palette.items():
            mask = raster == color_id
            rgb[mask] = rgb_value
        rgb = np.flipud(rgb)
        if scale > 1:
            rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
        return rgb

    sampled_scenes = list(scenes[::frame_stride]) if frame_stride > 1 else list(scenes)
    frames = [frame_to_rgb(scene) for scene in sampled_scenes]
    gif_path = OUTPUT_DIR / "simple_scene.gif"
    duration = frame_stride / max(fps, 1)
    imageio.imwrite(gif_path, frames, duration=duration, loop=0)
    return gif_path, len(sampled_scenes)


def main() -> None:
    creator = creator_lib.TaskCreator()

    platform = creator.add("static bar", scale=0.6)
    platform.set_center(150, 120).set_angle(25)

    ball = creator.add("dynamic ball", scale=0.12)
    ball.set_center(90, 210)
    ball.set_color("blue")

    creator.update_task(
        body1=ball,
        body2=platform,
        relationships=[creator.SpatialRelationship.TOUCHING],
    )
    creator.set_meta(creator.SolutionTier.SINGLE_OBJECT)

    simulation = simulator.simulate_task(creator.task, steps=240, stride=1)
    gif_path, num_frames = _frames_to_gif(simulation.sceneList, frame_stride=2)

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
            "steps_requested": 240,
            "frames_recorded": num_frames,
            "frame_stride": 2,
            "playback_fps": 60,
            "solved": simulation.isSolution,
        },
        "outputs": {
            "gif": str(gif_path),
        },
    }

    metadata_path = OUTPUT_DIR / "simple_scene_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Scene metadata written to", metadata_path)
    print("Animation written to", gif_path)


if __name__ == "__main__":
    main()
