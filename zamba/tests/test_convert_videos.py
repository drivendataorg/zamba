from pathlib import Path
import tempfile
from zamba.convert_videos import convert_videos
from zamba.utils import get_valid_videos


def test_convert_videos(data_dir):
    input_paths = [path for path in data_dir.glob("*") if not path.stem.startswith(".")]
    output_directory = tempfile.TemporaryDirectory()
    output_paths = convert_videos(
        input_paths, Path(output_directory.name), fps=15, width=448, height=252,
    )

    valid_videos, invalid_videos = get_valid_videos(output_paths)

    assert len(valid_videos) == len(output_paths)
    assert len(invalid_videos) == 0

    output_directory.cleanup()
