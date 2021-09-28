import os

import ffmpeg
from loguru import logger
import typer

from zamba.settings import ROOT_DIRECTORY
from zamba.data.metadata import load_metadata


ASSETS_DIR = ROOT_DIRECTORY / "tests" / "assets"


def main():
    metadata = load_metadata(subset="dev", include_local_path=True)
    random_video_paths = metadata.groupby("country").sample(1, random_state=89).local_path.tolist()

    # [
    #     'data/raw/dzanga_sangha/T5/First_visit_T5/First_visit_T5C2/11260046.MP4',
    #     'data/raw/lope/chimp_videos/Lop_E3/Lop_CAM12_0790537_9974042_20141016/EK000016.AVI',
    #     'data/raw/benjamin/20180516_112634_Cephalophus rufilatus_252212_1267525_Gauche.MP4',
    #     'data/raw/noemie/Videos_All/Taï_T167/Taï_cam121_684008_650182_20161017/12010136.AVI',
    #     'data/raw/savanna/Gorongosa_Mozambique/2016 Videos/005/Baboon/09100353.AVI',
    #     'data/raw/goualougo_2013/gorillas_2013/MPI_FID_15_Emeli/11-Sept-2013/FID_15_Emeli_2013-9-11_0028.AVI',
    #     'data/raw/savanna/Grumeti_Tanzania/M41_check1/08230028_Dikdik.AVI',
    # ]

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

        output_path = ASSETS_DIR / "videos" / input_path
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
