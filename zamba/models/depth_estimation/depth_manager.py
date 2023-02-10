from loguru import logger
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.utils
import torch.utils.data
from torchvision import transforms
from torchvision.transforms import Resize
from tqdm import tqdm

from zamba.data.video import load_video_frames
from zamba.models.utils import RegionEnum, download_weights
from zamba.object_detection.yolox.megadetector_lite_yolox import MegadetectorLiteYoloX
from zamba.pytorch.transforms import ConvertHWCtoCHW


MODELS = dict(
    depth=dict(
        private_weights_url="s3://drivendata-client-zamba/depth_estimation_winner_weights/second_place/tf_efficientnetv2_l_in21k_2_5_pl4/model_best.pt",
        weights="zamba_depth_30aaa90525.pt",
    )
)


def depth_transforms(size):
    return transforms.Compose(
        [
            # put channels first
            ConvertHWCtoCHW(),
            # resize to desired height and width
            Resize(size),
        ]
    )


class DepthDataset(torch.utils.data.Dataset):
    def __init__(self, filepaths):
        # these are hardcoded because they depend on the trained model weights used for inference
        self.height = 270
        self.width = 480
        self.channels = 3
        self.window_size = 2
        # first frames are swapped; this maintains the bug in the original code
        self.order = [-1, -2, 0, 1, 2]
        self.num_frames = self.window_size * 2 + 1
        self.fps = 1

        mdlite = MegadetectorLiteYoloX()
        cached_frames = dict()
        detection_dict = dict()

        transform = depth_transforms(size=(self.height, self.width))

        logger.info(f"Running object detection on {len(filepaths)} videos.")
        for video_filepath in tqdm(filepaths):
            # get video array at 1 fps, use full size for detecting objects
            logger.debug(f"Loading video: {video_filepath}")
            try:
                arr = load_video_frames(video_filepath, fps=self.fps)
            except:  # noqa: E722
                logger.warning(f"Video {video_filepath} could not be loaded. Skipping.")
                continue

            # add video entry to cached dict with length (number of seconds since fps=1)
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
                                selected_frame = arr[i]
                            except:  # noqa: E722
                                selected_frame = np.zeros(
                                    (self.height, self.width, self.channels), dtype=int
                                )

                            # transform puts channels first and resizes
                            cached_frames[video_filepath][f"frame_{i}"] = transform(
                                torch.tensor(selected_frame)
                            ).numpy()

                            del selected_frame

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
            del detections_per_frame

        self.detection_dict = detection_dict
        self.detection_indices = list(detection_dict.keys())
        self.cached_frames = cached_frames

    def __len__(self):
        return len(self.detection_indices)

    def __getitem__(self, index):
        """Given a detection index, returns a tuple containing the tensor of stacked frames,
        video filename, and time into the video for the target frame.
        """

        # get detection info
        detection_idx = self.detection_indices[index]
        det_metadata = self.detection_dict[detection_idx]
        det_frame = det_metadata["frame"]
        det_video = det_metadata["video"]

        # set up input array of frames within window of detection
        # frames are stacked channel-wise
        input = np.concatenate(
            [self.cached_frames[det_video][f"frame_{det_frame + i}"] for i in self.order]
        )

        # to tensor and normalize
        tensor = torch.from_numpy(input) / 255.0

        # keep track of video name and time as well
        return tensor, det_video, det_frame


class DepthEstimationManager:
    def __init__(
        self,
        model_cache_dir: Path,
        gpus: int,
        weight_download_region: RegionEnum = RegionEnum("us"),
        batch_size: int = 64,
        tta: int = 2,
        num_workers: int = 8,
    ):
        """Create a depth estimation manager object

        Args:
            model_cache_dir (Path): Path for downloading and saving model weights.
            gpus (int): Number of GPUs to use for inference.
            weight_download_region (str): s3 region to download pretrained weights from.
                Options are "us" (United States), "eu" (Europe), or "asia" (Asia Pacific).
                Defaults to "us".
            batch_size (int, optional): Batch size to use for inference. Defaults to 64.
                Note: a batch is a set of frames, not videos, for the depth model.
            tta (int, optional): Number of flips to apply for test time augmentation.
            num_workers (int): Number of subprocesses to use for data loading. The maximum value is
                the number of CPUs in the system. Defaults to 8.
        """
        self.batch_size = batch_size
        self.tta = tta
        self.num_workers = num_workers
        self.gpus = gpus

        model = MODELS["depth"]

        self.model_weights = model_cache_dir / model["weights"]
        if not self.model_weights.exists():
            model_cache_dir.mkdir(parents=True, exist_ok=True)
            self.model_weights = download_weights(
                model["weights"], model_cache_dir, weight_download_region
            )

        if self.gpus > 0:
            self.device = "cuda"
        else:
            self.device = "cpu"

    def predict(self, filepaths):
        """Generate predictions for a list of video filepaths."""

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
            num_workers=self.num_workers,
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

                    time = time.numpy()

                    for d, vid, t in zip(distance.cpu().numpy(), filepath, time):
                        predictions.append((vid, t, d))

        predictions = pd.DataFrame(
            predictions,
            columns=["filepath", "time", "distance"],
        ).round(
            {"distance": 1}
        )  # round to useful number of decimal places

        logger.info("Processing output.")
        # post process to add nans for frames where no animal was detected
        videos = list(test_dataset.cached_frames.keys())
        lengths = [np.arange(test_dataset.cached_frames[v]["video_length"]) for v in videos]

        # create one row per frame for duration of video
        output = pd.Series(index=videos, data=lengths).explode().to_frame().reset_index()
        output.columns = ["filepath", "time"]

        # merge in predictions
        if len(predictions) > 0:
            output = output.merge(predictions, on=["filepath", "time"], how="outer")
        else:
            # create empty distance column
            output["distance"] = np.nan
        return output
