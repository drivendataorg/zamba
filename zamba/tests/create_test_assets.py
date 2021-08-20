import os
import ffmpeg
import typer

from loguru import logger
from zamba.settings import ROOT_DIRECTORY
from zamba.data.metadata import load_metadata

TEST_DATA_DIRECTORY = ROOT_DIRECTORY / "tests" / "assets" / "videos"


def main():
    metadata = load_metadata(subset="dev", include_local_path=True)
    random_video_paths = metadata.groupby("country").sample(1, random_state=89).local_path.tolist()

    trouble_video_paths = [
        "data/raw/savanna/Grumeti_Tanzania/G41_check2/09100029_Eland.MP4",
        "data/raw/noemie/Videos_All/Taï_TD4/Taï_cam17_691988_643179_20161119/PICT0026.AVI",
        "data/raw/noemie/Videos_All/Taï_T147/Taï_cam170_678016_649215_20161227/02120086.AVI",
        "data/raw/noemie/Videos_All/Taï_T124/Taï_cam61_687002_647212_20161106/f4636736.avi",
        "data/raw/goualougo_2013/chimp_MPI_FID_2013/MPI_FID_31_Abel/06-May-2013/FID_31_Abel_2013-5-6_0027.AVI",
    ]

    video_paths = random_video_paths + trouble_video_paths
    height = 20

    for input_path in video_paths:
        logger.info(f"Rescaling video {input_path} to height={height}")

        output_path = TEST_DATA_DIRECTORY / input_path
        output_path.parent.mkdir(exist_ok=True, parents=True)

        if output_path.exists():
            os.remove(output_path)

        (
            ffmpeg.input(str(ROOT_DIRECTORY / input_path))
            .filter("scale", -2, height)
            .output(str(output_path))
            .run(capture_stdout=True, capture_stderr=True)
        )


if __name__ == "__main__":
    typer.run(main)
