# Rendering Commands

## GIF Output (default speedup warnings possible)
```
python scripts/simple_scene_demo.py \
    --pixel-scale 4 \
    --frame-stride 2 \
    --playback-speed 20 \
    --steps 240 \
    --output-dir output/simple_scene
```

## MP4 Output (true high-speed playback)
```
python scripts/simple_scene_demo.py \
    --video-format mp4 \
    --frame-stride 1 \
    --playback-speed 50 \
    --pixel-scale 2 \
    --steps 120 \
    --output-dir output/fast_mp4
```

Note: MP4 output requires `pip install imageio-ffmpeg` so the FFmpeg writer is available. Trajectory lines (red) are overlaid automatically for every dynamic body.
