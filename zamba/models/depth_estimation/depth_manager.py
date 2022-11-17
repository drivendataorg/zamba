from PIL import Image
from loguru import logger
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.utils
import torch.utils.data
from tqdm import tqdm
from typing import Optional

from zamba.data.video import load_video_frames
from zamba.models.utils import RegionEnum, download_weights
from zamba.object_detection.yolox.megadetector_lite_yolox import MegadetectorLiteYoloX


MODELS = dict(
    depth=dict(
        private_weights_url="s3://drivendata-client-zamba/depth_estimation_winner_weights/second_place/tf_efficientnetv2_l_in21k_2_5_pl4/model_best.pt",
        weights="zamba_depth_30aaa90525.pt",
    )
)


def normalize(img):
    img = np.transpose(img, (2, 0, 1))
    img = img.astype("float32") / 255
    return img


class DepthDataset(torch.utils.data.Dataset):
    def __init__(self, filepaths):

        self.height = 270
        self.width = 480
        self.channels = 3
        self.window_size = 2
        self.num_frames = self.window_size * 2 + 1

        mdlite = MegadetectorLiteYoloX()
        cached_frames = dict()
        detection_dict = dict()

        logger.info(f"Running object detection on {len(filepaths)} videos.")
        for video_filepath in tqdm(filepaths):

            # get video array at 1 fps, use full size for detecting objects
            logger.debug(f"Loading video: {video_filepath}")
            try:
                arr = load_video_frames(video_filepath, fps=1)
            except:
                logger.warning(f"Video {video_filepath} could not be loaded. Skipping.")
                continue

            # add video to cached dict with length (number of seconds since fps=1)
            cached_frames[video_filepath] = dict(video_length=len(arr))

            # get detections in each frame
            logger.debug(f"Detecting video: {video_filepath}")
            detections_per_frame = mdlite.detect_video(video_arr=arr)

            # iterate over frames
            for frame_idx, (detections, scores) in enumerate(detections_per_frame):

                # if anything is detected in the frame, save out relevant frames
                if len(detections) > 0:
                    logger.debug(f"{len(detections)} detection(s) found at second {frame_idx}.")

                    # get frame indices around frame with detection
                    min_frame = frame_idx - self.window_size
                    max_frame = frame_idx + self.window_size

                    # add relevant resized frames to dict if not already added
                    # if index is before start or after end of video, use an array of zeros
                    for i in range(min_frame, max_frame + 1):
                        if f"frame_{i}" not in cached_frames[video_filepath].keys():
                            try:
                                selected_frame = np.array(
                                    # PIL expects size to be (w, h)
                                    Image.fromarray(arr[i]).resize((self.width, self.height))
                                )
                            except:
                                selected_frame = np.zeros((self.height, self.width))

                            cached_frames[video_filepath][f"frame_{i}"] = selected_frame

                    # iterate over detections in frame to create universal detection ID
                    for i, (detection, score) in enumerate(zip(detections, scores)):
                        universal_det_id = f"{i}_{frame_idx}_{video_filepath}"

                        # save out bounding box and score info in case we want to mask out portions
                        detection_dict[universal_det_id] = dict(
                            bbox=detection,
                            score=score,
                            frame=frame_idx,
                            video=video_filepath,
                        )

            del arr

        self.detection_dict = detection_dict
        self.detection_indices = detection_dict.keys()
        self.cached_frames = cached_frames

    def __len__(self):
        return len(self.detection_indices)

    def __getitem__(self, index):
        """Given the index of the target image, returns a tuple of the stacked image array, the image
        filename stem, and the time into the video for the target image.
        """

        # get detection info
        detection_idx = self.detection_indices[index]
        det_metadata = self.detection_dict[detection_idx]
        det_frame = det_metadata["frame"]
        det_video = det_metadata["video"]

        # set up input array of frames within window of detection
        # frames are stacked channel-wise
        input = np.zeros((self.height, self.width, self.channels * self.num_frames))
        n = 0
        for frame_idx in range(det_frame - self.window_size, det_frame + self.window_size + 1):
            input[:, :, n : n + self.channels] = self.cached_frames[det_video][
                f"frame_{frame_idx}"
            ]
            n += self.channels

        # TODO: original order was -1, -2, 0, 1, 2; TBD if we want to maintain that bug
        # TODO: mask out other detection from the same frame?

        # normalize and convert to tensor
        input = normalize(input)
        tensor = torch.from_numpy(input)

        # keep track of video name and time
        return tensor, det_video, det_frame


class DepthEstimationManager:
    def __init__(
        self,
        model_cache_dir: Optional[Path] = None,
        weight_download_region: RegionEnum = RegionEnum("us"),
        batch_size: int = 64,
        tta: int = 2,
        use_log: bool = False,
    ):
        """Create a depth estimation manager object

        Args:
            model_cache_dir (Path, optional): Path for downloading and saving model weights.
                Defaults to env var `MODEL_CACHE_DIR` or the OS app cache dir.
            weight_download_region (str): s3 region to download pretrained weights from.
                Options are "us" (United States), "eu" (Europe), or "asia" (Asia Pacific).
                Defaults to "us".
            batch_size (int, optional): Batch size to use for inference. Defaults to 64.
                Note: a batch is a set of frames, not videos, for the depth model.
            tta (int, optional): Number of flips to apply for test time augmentation.
            use_log (bool, optional): Whether to take the exponential of the predictions
                (see torch.special.expm1). Defaults to False.
        """
        self.batch_size = batch_size
        self.tta = tta
        self.use_log = use_log

        model = MODELS["depth"]

        self.model_weights = model_cache_dir / model["weights"]
        if not self.model_weights.exists():
            model_cache_dir.mkdir(parents=True, exist_ok=True)
            self.model_weights = download_weights(
                model["weights"], model_cache_dir, weight_download_region
            )

        # automatically use CPU if no cuda available
        if not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = "cuda"

    def predict(self, filepaths):
        """Generate predictions for a list of filepaths, each representing one target frame.
        Filepaths should be given relative to the data_dir."""

        # load model
        model = torch.jit.load(self.model_weights, map_location=self.device).eval()

        # load dataset
        test_dataset = DepthDataset(filepaths)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=None,
            collate_fn=None,
            num_workers=min(self.batch_size, 8),
            pin_memory=False,
            persistent_workers=True,
        )

        logger.info("Generating depth predictions for detected animals.")
        predictions = []
        with torch.no_grad():
            with tqdm(test_loader) as pbar:
                distance: torch.Tensor = torch.zeros(self.batch_size, device=self.device)
                for image, filepath, time in pbar:
                    bs = image.size(0)
                    image = image.to(self.device, non_blocking=True)

                    distance.zero_()
                    logits = model(image)
                    logits = logits.squeeze(1)
                    distance[:bs] += logits

                    if self.tta > 1:
                        logits = model(torch.flip(image, dims=[-1]))
                        logits = logits.squeeze(1)
                        distance[:bs] += logits

                    distance /= self.tta

                    if self.use_log:
                        distance.expm1_()

                    time = time.numpy()

                    for d, vid, t in zip(distance.cpu().numpy(), filepath, time):
                        predictions.append((vid, t, d))

        predictions = pd.DataFrame(
            predictions,
            columns=["filepath", "time", "distance"],
        )

        logger.info("Processing output.")
        # post process to add nans for frames where no animal was detected
        videos = list(test_dataset.cached_frames.keys())
        lengths = [np.arange(test_dataset.cached_frames[v]["video_length"]) for v in videos]

        # create one row per frame for duration of video
        df = pd.Series(index=videos, data=lengths).explode().to_frame().reset_index()
        df.columns = ["filepath", "time"]

        # merge in predictions
        output = df.merge(predictions, on=["filepath", "time"], how="outer")
        return output
