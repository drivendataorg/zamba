import click
import logging
from pathlib import Path
from subprocess import check_output, CalledProcessError, STDOUT
from tqdm import tqdm

from zamba.config import log_levels
from zamba.utils import unique_processed_paths


def ffmpeg_resample(
    input_file,
    output_file,
    fps,
    width,
    height,
):
    if output_file.exists():
        return output_file

    command = [
        'ffmpeg',
        '-y',
        '-i', str(input_file),
        '-r', str(fps),  # frame rate
        '-c:v', 'libx264',  # use H264 codec
        "-an",  # drop audio stream
        '-crf', '20',  # H264 constant rate factor (0 lossless - 51 worst possible quality)
        '-strict', '-6',
        '-vf', (
            f"""scale=w={width}:h={height}:force_original_aspect_ratio=1,"""
            f"""pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"""
            """histeq"""
        ),
        str(output_file),
    ]

    try:
        check_output(command, stderr=STDOUT)
        logging.debug("Created %s", str(output_file))

    except CalledProcessError as e:
        logging.debug(e.output)

    return output_file


def convert_videos(
    input_paths,
    output_directory,
    fps,
    width,
    height,
):
    """Converts videos to a standard resolution and frame rate and saves to disk

    Args:
        input_paths (list of str or Path): A list of paths to the input files
        output_directory (str or Path): Path to an output directory
        fps (float)
        width (int)
        height (int)

    Returns:
        list of Path: A list of output paths
    """
    output_directory.mkdir(exist_ok=True)
    output_paths = unique_processed_paths(input_paths, output_directory)

    for input_path, output_path in tqdm(
        zip(input_paths, output_paths),
        total=len(input_paths),
        desc="Resampling videos",
    ):
        ffmpeg_resample(input_path, output_path, fps, width, height)

    return output_paths


@click.command()
@click.option(
    "--input_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to a list of file names to be processed",
)
@click.option(
    "--output_directory",
    type=click.Path(),
    required=True,
    help="Directory where the output will be saved",
)
@click.option(
    "--fps",
    default=15,
    type=int,
    help="Output frames per second",
)
@click.option(
    "--width",
    default=448,
    type=int,
    help="Width of output video",
)
@click.option(
    "--height",
    default=252,
    type=int,
    help="Height of output video",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    type=bool,
    help="Output debug messages",
)
def run(
    input_path,
    output_directory,
    fps,
    width,
    height,
    verbose,
):
    logging.basicConfig(
        level=log_levels[verbose],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    input_path = Path(input_path)
    output_directory = Path(output_directory)

    if input_path.is_dir():
        input_files = sorted(list(input_path.glob("**/*")))
    else:
        with input_path.open("r") as f:
            input_files = [Path(line.strip("\n")) for line in f.readlines()]

    convert_videos(input_files, output_directory, fps=fps, width=width, height=height)


if __name__ == "__main__":
    run()
