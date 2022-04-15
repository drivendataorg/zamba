import os

from cloudpathlib import S3Path, S3Client
import cv2
from loguru import logger
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.utils
import torch.utils.data
import tqdm
from typing import Union


def imread(path):
    img = cv2.imread(str(path))

    return img


def normalize(img):
    img = np.transpose(img, (2, 0, 1))
    img = img.astype("float32") / 255
    img = torch.from_numpy(img)

    return img


def download_from_s3(
    filepath: str,
    destination_dir: Union[os.PathLike, str],
):
    s3p = S3Path(filepath, client=S3Client(local_cache_dir=destination_dir))

    return s3p.fspath


class DepthDataset(torch.utils.data.Dataset):
    def __init__(self, filepaths, window_size):
        self.filepaths = filepaths
        self.window_size = window_size
        self.order = [f"image_{i}" for i in range(1, window_size + 1)]
        self.order.append("image")
        self.order.extend([f"image{i}" for i in range(1, window_size + 1)])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        """Given the index of the target image, returns a tuple of the stacked image array, the image
        filename stem, and the time into the video for the target image.
        """
        target_image_path = Path(self.filepaths[index])

        img = imread(target_image_path)
        inputs = {"image": img}

        time = int(target_image_path.stem.split("_")[-1])
        stem = "_".join(target_image_path.stem.split("_")[:-1])

        if self.window_size > 0:
            suffix = target_image_path.suffix
            for i in range(1, self.window_size + 1):
                s = time - i
                if s >= 0:
                    new_stem = "_".join([stem, str(s)])
                    new_name = f"{new_stem}{suffix}"
                    new_path = target_image_path.with_name(new_name)
                    inputs[f"image_{i}"] = imread(new_path)
                # pad images if before beginning
                else:
                    inputs[f"image_{i}"] = np.zeros_like(img)

                s = time + i
                new_stem = "_".join([stem, str(s)])
                new_name = f"{new_stem}{suffix}"
                new_path = target_image_path.with_name(new_name)
                if new_path.is_file():
                    inputs[f"image{i}"] = imread(new_path)
                # pad images if past end
                else:
                    inputs[f"image{i}"] = np.zeros_like(img)

        # concatenate filepaths in the correct order
        img = np.concatenate([inputs[i] for i in self.order], axis=-1)
        img = normalize(img)

        return img, stem, time


class DepthEstimationManager:
    def __init__(
        self,
        img_dir: Path,
        model_s3_path: Path = S3Path(
            "s3://drivendata-client-zamba/depth_estimation_winner_weights/second_place/tf_efficientnetv2_l_in21k_2_5_pl4/model_best.pt"
        ),
        model_cache_dir: Path = Path(".zamba_cache"),
        batch_size: int = 64,
        window_size: int = 2,
        tta: int = 2,
        use_log: bool = False,
    ):
        """Create a depth estimation manager object

        Args:
            img_dir (Path): Where to find the files listed in filepaths, as well as the other images
                before and after each frame to create stacked image arrays
            model_s3_path (Path, optional): Where to download the depth estimation model weights from.
                Defaults to S3Path( "s3://drivendata-client-zamba/depth_estimation_winner_weights/second_place/tf_efficientnetv2_l_in21k_2_5_pl4/model_best.pt" ).
            model_cache_dir (Path, optional): Path for downloading and saving model weights. Defaults
                to Path(".zamba_cache").
            batch_size (int, optional): Batch size for running the depth model. Defaults to 64.
            window_size (int, optional): _description_. Defaults to 2.
            tta (int, optional): How many images to stack before and after the target frame. If set
                to two, each stacked image array will have five images total. Defaults to 2.
            use_log (bool, optional): Whether to take the exponential of the predictions (see
                torch.special.expm1). Defaults to False.
        """
        # import model weights
        model_weights_path = model_cache_dir / str(model_s3_path).replace("s3://", "")
        if not model_weights_path.exists():
            logger.info(f"Downloading model weights from {model_s3_path}")
            model_cache_dir.mkdir(parents=True, exist_ok=True)
            self.model_weights = download_from_s3(model_s3_path, model_cache_dir)
        else:
            self.model_weights = model_weights_path
        logger.info(f"Using model downloaded at {self.model_weights}")

        # automatically use CPU if no cuda available
        if not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = "cuda"

        self.img_dir = img_dir
        self.batch_size = batch_size
        self.window_size = window_size
        self.tta = tta
        self.use_log = use_log

    def predict(self, filepaths):
        """Generate predictions for a list of filepaths, each representing one target frame. Filepaths should be given relative to the img_dir"""
        torch.backends.cudnn.benchmark = True

        # load model
        model = torch.jit.load(self.model_weights, map_location=self.device).eval()

        # load dataset
        test_dataset = DepthDataset(filepaths, window_size=self.window_size)
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

        predictions = []
        with torch.no_grad():
            with tqdm.tqdm(test_loader) as pbar:
                distance: torch.Tensor = torch.zeros(self.batch_size, device=self.device)
                for image, filepath_stem, time in pbar:
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

                    for d, vid, t in zip(distance.cpu().numpy(), filepath_stem, time):
                        predictions.append((vid, t, d))

        predictions = pd.DataFrame(
            predictions,
            columns=["filepath_stem", "time", "distance"],
        )

        return predictions
