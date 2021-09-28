from cloudpathlib import S3Path
from loguru import logger
import pandas as pd
from tqdm import tqdm

from zamba_algorithms.data.video import load_video_frames, VideoLoaderConfig
from zamba_algorithms.settings import DATA_DIRECTORY


with (DATA_DIRECTORY / "interim" / "chimpandsee_labels_for_videos.csv").open("r") as f:
    df = pd.read_csv(f)

df["local_path"] = df.filepath.apply(lambda x: S3Path(x).key)

failed_videos = []
logger.info("Trying to load each video")
for video_path in tqdm(df.local_path.unique()):
    try:
        # size very small to not explode cache
        load_video_frames(
            filepath=video_path, config=VideoLoaderConfig(video_height=10, video_width=10)
        )
    except:  # noqa: E722
        logger.info(f"Found failed video: {video_path}")
        failed_videos.append(video_path)

logger.info("Writing out failed videos")
failed_vids = pd.DataFrame(failed_videos)
failed_vids.columns = ["local_path"]
failed_vids.to_csv(DATA_DIRECTORY / "interim" / "chimpandsee_failed_videos.csv", index=False)
