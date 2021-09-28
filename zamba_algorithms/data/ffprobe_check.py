from cloudpathlib import S3Path
from loguru import logger
import ffmpeg
import pandas as pd
from tqdm import tqdm

from zamba_algorithms.settings import DATA_DIRECTORY

logger.info("Loading unified metadata")
with (DATA_DIRECTORY / "processed" / "unified_metadata.csv").open("r") as f:
    df = pd.read_csv(f, low_memory=False, parse_dates=["datetime"])

df["local_path"] = df.filepath.apply(lambda x: S3Path(x).key)

failed_videos = []
logger.info("Try ffprobe on each video")
for video_path in tqdm(df.local_path.unique()):
    try:
        ffmpeg.probe(video_path)
    except:  # noqa: E722
        logger.info(f"Found failed video: {video_path}")
        failed_videos.append(video_path)

logger.info("Writing out failed videos")
failed_vids = pd.DataFrame(failed_videos)
failed_vids.columns = ["local_path"]
failed_vids.to_csv(DATA_DIRECTORY / "interim" / "failed_videos.csv", index=False)
