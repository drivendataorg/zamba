import copy
import os
from itertools import repeat
from pathlib import Path
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
from loguru import logger
from megadetector.detection import run_detector
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from zamba.images.bbox import (
    absolute_bbox,
    crop_to_bounding_box,
    get_cache_filename,
    load_image,
    BboxLayout,
)


class ImageClassificationDataset(Dataset):
    def __init__(self, data_dir: Path, annotations: pd.DataFrame, transform) -> None:
        self.annotations = annotations
        self.data_dir = data_dir

        self.transform = transform

    def _get_image_path(self, item) -> Path:
        if "cached_bbox" in item:
            return item["cached_bbox"]
        else:
            return self.data_dir / item["filepath"]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        item = self.annotations.iloc[index]
        label = item["label"]

        img_path = self._get_image_path(item)

        with img_path.open("rb") as fp:
            image = Image.open(fp)
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, int(label)


class ImageClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        annotations: pd.DataFrame,
        cache_dir: Path,
        crop_images: bool,
        batch_size: int = 16,
        num_workers: Optional[int] = None,
        train_transforms=None,
        test_transforms=None,
        detection_threshold: float = 0.2,
    ) -> None:
        super().__init__()
        if train_transforms is None:
            train_transforms = transforms.Compose([transforms.ToTensor()])
        if test_transforms is None:
            test_transforms = transforms.Compose([transforms.ToTensor()])

        self.data_dir = data_dir
        self.cache_dir = cache_dir

        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

        self.detection_threshold = detection_threshold

        if num_workers is None:
            num_workers = os.cpu_count()
        self.num_workers = num_workers

        self.annotations = annotations
        if crop_images:
            self.annotations = self.preprocess_annotations(annotations)

    def preprocess_annotations(self, annotations: pd.DataFrame) -> pd.DataFrame:
        """Preprocesses annotations by cropping bounding boxes or running the MegaDetector."""
        num_annotations = len(annotations)
        bbox_in_df = all(column in annotations.columns for column in ["x1", "x2", "y1", "y2"])

        if bbox_in_df:
            logger.info(f"Bboxes found in annotations. Cropping images to cache_dir: {self.cache_dir}")

            processed_annotations = process_map(
                crop_to_bounding_box,
                annotations.iterrows(),
                repeat(self.cache_dir),
                repeat(self.data_dir),
                total=len(annotations),
                desc="Cropping images",
            )

            annotations = pd.DataFrame(processed_annotations)
        else:
            processed_annotations = []
            detector = run_detector.load_detector("MDV5A")

            for _, row in tqdm(
                annotations.iterrows(),
                total=len(annotations),
                desc="Running MegaDetector for bounding boxes",
            ):
                filepath = self.data_dir / row["filepath"]
                image = load_image(filepath)
                result = detector.generate_detections_one_image(
                    image, row["filepath"], detection_threshold=self.detection_threshold
                )

                for detection in result["detections"]:
                    detection_row = copy.deepcopy(row)
                    detection_row["detection_conf"] = detection["conf"]
                    detection_row["detection_category"] = detection["category"]

                    bbox = absolute_bbox(image, detection["bbox"], bbox_layout=BboxLayout.XYWH)
                    cache_path = self.cache_dir / get_cache_filename(detection_row["filepath"], bbox)

                    if not cache_path.exists():
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        cropped_image = image.crop(bbox)
                        with open(cache_path, "wb") as f:
                            cropped_image.save(f)

                    detection_row.update({
                        "x1": bbox[0], "x2": bbox[2], "y1": bbox[1], "y2": bbox[3],
                        "cached_bbox": cache_path.resolve().absolute(),
                    })

                    processed_annotations.append(detection_row)

            annotations = pd.DataFrame(processed_annotations)

        logger.info(
            f"Objects before preprocessing: {num_annotations}, "
            f"Objects after preprocessing: {len(annotations)}"
        )

        return annotations

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            ImageClassificationDataset(
                self.data_dir,
                self.annotations[self.annotations["split"] == "train"],
                self.train_transforms,
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            ImageClassificationDataset(
                self.data_dir,
                self.annotations[self.annotations["split"] == "val"],
                self.test_transforms,
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            ImageClassificationDataset(
                self.data_dir,
                self.annotations[self.annotations["split"] == "test"],
                self.test_transforms,
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
