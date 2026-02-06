import time
from pathlib import Path
from statistics import mean, median

import typer

from zamba.data.video import MegadetectorConfig, VideoLoaderConfig, load_video_frames
from zamba.object_detection.yolox.megadetector_lite_yolox import MegadetectorLiteYoloXConfig
from zamba.settings import VIDEO_SUFFIXES


def main(
    video_dir: Path = typer.Argument(..., help="Directory containing video files."),
    total_frames: int = typer.Option(16, help="Target number of frames."),
    model_input_height: int = typer.Option(240, help="Model input height."),
    model_input_width: int = typer.Option(426, help="Model input width."),
    fps: float = typer.Option(4.0, help="Frames per second for sampling."),
    crop_bottom_pixels: int = typer.Option(50, help="Pixels to crop from bottom."),
):
    """Compare video loading speed for MDLite vs MegaDetector."""
    video_dir = Path(video_dir)
    if not video_dir.is_dir():
        raise ValueError(f"{video_dir} is not a directory.")

    suffixes = {s.lower() if s.startswith(".") else f".{s.lower()}" for s in VIDEO_SUFFIXES}
    videos = [
        path
        for path in video_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    ]
    videos = sorted(videos)

    if not videos:
        raise ValueError(f"No video files found in {video_dir}")

    print(f"Found {len(videos)} videos\n")

    base_config = VideoLoaderConfig(
        total_frames=total_frames,
        model_input_height=model_input_height,
        model_input_width=model_input_width,
        crop_bottom_pixels=crop_bottom_pixels,
        fps=fps,
        ensure_total_frames=True,
    )

    mdlite_config = base_config.copy(
        update={
            "megadetector_lite_config": MegadetectorLiteYoloXConfig(
                confidence=0.25,
                fill_mode="score_sorted",
                frame_batch_size=24,
                image_height=640,
                image_width=640,
                n_frames=total_frames,
            )
        }
    )

    md_config = base_config.copy(
        update={
            "megadetector_config": MegadetectorConfig(n_frames=total_frames)
        }
    )

    print("Loading with MDLite...")
    mdlite_times = []
    for video in videos:
        start = time.perf_counter()
        _ = load_video_frames(video, config=mdlite_config)
        mdlite_times.append(time.perf_counter() - start)
        print(f"  {video.name}: {mdlite_times[-1]:.2f}s")

    print("\nLoading with MegaDetector...")
    md_times = []
    for video in videos:
        start = time.perf_counter()
        _ = load_video_frames(video, config=md_config)
        md_times.append(time.perf_counter() - start)
        print(f"  {video.name}: {md_times[-1]:.2f}s")

    print("\n" + "=" * 60)
    print(f"MDLite - mean: {mean(mdlite_times):.3f}s, median: {median(mdlite_times):.3f}s")
    print(f"MD     - mean: {mean(md_times):.3f}s, median: {median(md_times):.3f}s")
    print(f"Speedup: {mean(mdlite_times) / mean(md_times):.2f}x")


if __name__ == "__main__":
    typer.run(main)

