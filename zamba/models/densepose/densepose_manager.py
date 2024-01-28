import json
import logging
from pathlib import Path

import cv2

try:
    from densepose import add_densepose_config
    from densepose.data.utils import get_class_to_mesh_name_mapping
    from densepose.modeling.build import build_densepose_embedder
    from densepose.structures.cse import DensePoseEmbeddingPredictorOutput
    from densepose.vis.densepose_outputs_vertex import (
        DensePoseOutputsTextureVisualizer,
        DensePoseOutputsVertexVisualizer,
    )
    from densepose.vis.densepose_results_textures import get_texture_atlas
    from densepose.vis.extractor import (
        create_extractor,
    )
    from detectron2.config import get_cfg
    from detectron2.data.detection_utils import read_image
    from detectron2.engine.defaults import DefaultPredictor
    from detectron2.structures.instances import Instances

    DENSEPOSE_AVAILABLE = True
except ImportError:
    DENSEPOSE_AVAILABLE = False
    DensePoseOutputsTextureVisualizer = None  # dummies for static defs
    DensePoseOutputsVertexVisualizer = None
    get_texture_atlas = lambda x: None  # noqa: E731


import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from zamba.data.video import load_video_frames
from zamba.models.utils import RegionEnum, download_weights


MODELS = dict(
    animals=dict(
        config=str(
            Path(__file__).parent
            / "assets"
            / "densepose_rcnn_R_50_FPN_soft_animals_I0_finetune_16k.yaml"
        ),
        densepose_weights_url="https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_soft_animals_I0_finetune_16k/270727112/model_final_421d28.pkl",
        weights="zamba_densepose_model_final_421d28.pkl",
        viz_class=DensePoseOutputsVertexVisualizer,
        viz_class_kwargs=dict(),
    ),
    chimps=dict(
        config=str(
            Path(__file__).parent
            / "assets"
            / "densepose_rcnn_R_50_FPN_soft_chimps_finetune_4k.yaml"
        ),
        densepose_weights_url="https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_soft_chimps_finetune_4k/253146869/model_final_52f649.pkl",
        weights="zamba_densepose_model_final_52f649.pkl",
        viz_class=DensePoseOutputsTextureVisualizer,
        viz_class_kwargs=dict(
            texture_atlases_dict={
                "chimp_5029": get_texture_atlas(
                    str(Path(__file__).parent / "assets" / "chimp_texture_colors_flipped.tif")
                )
            }
        ),
        anatomy_color_mapping=str(Path(__file__).parent / "assets" / "chimp_5029_parts.csv"),
    ),
)


class DensePoseManager:
    def __init__(
        self,
        model=MODELS["chimps"],
        model_cache_dir: Path = Path(".zamba_cache"),
        download_region=RegionEnum("us"),
    ):
        """Create a DensePoseManager object.

        Parameters
        ----------
        model : dict, optional (default MODELS['chimps'])
            A dictionary with the densepose model defintion like those defined in MODELS.
        """
        if not DENSEPOSE_AVAILABLE:
            raise ImportError(
                "Densepose not installed. See: https://zamba.drivendata.org/docs/stable/models/densepose/#installation"
            )

        # setup configuration for densepose
        self.cfg = get_cfg()
        add_densepose_config(self.cfg)

        self.cfg.merge_from_file(model["config"])

        if not (model_cache_dir / model["weights"]).exists():
            model_cache_dir.mkdir(parents=True, exist_ok=True)
            self.cfg.MODEL.WEIGHTS = download_weights(
                model["weights"], model_cache_dir, download_region
            )

        # automatically use CPU if no cuda available
        if not torch.cuda.is_available():
            self.cfg.MODEL.DEVICE = "cpu"

        self.cfg.freeze()

        logging.getLogger("fvcore").setLevel("CRITICAL")  # silence noisy detectron2 logging
        # set up predictor with the configuration
        self.predictor = DefaultPredictor(self.cfg)

        # we have a specific texture atlas for chimps with relevant regions
        # labeled that we can use instead of the default segmentation
        self.visualizer = model["viz_class"](
            self.cfg,
            device=self.cfg.MODEL.DEVICE,
            **model.get("viz_class_kwargs", {}),
        )

        # set up utilities for use with visualizer
        self.vis_extractor = create_extractor(self.visualizer)
        self.vis_embedder = build_densepose_embedder(self.cfg)
        self.vis_class_to_mesh_name = get_class_to_mesh_name_mapping(self.cfg)
        self.vis_mesh_vertex_embeddings = {
            mesh_name: self.vis_embedder(mesh_name).to(self.cfg.MODEL.DEVICE)
            for mesh_name in self.vis_class_to_mesh_name.values()
            if self.vis_embedder.has_embeddings(mesh_name)
        }

        if "anatomy_color_mapping" in model:
            self.anatomy_color_mapping = pd.read_csv(model["anatomy_color_mapping"], index_col=0)
        else:
            self.anatomy_color_mapping = None

    def predict_image(self, image):
        """Run inference to get the densepose results for an image.

        Parameters
        ----------
        image :
            numpy array (unit8) of an image in BGR format or path to an image

        Returns
        -------
        tuple
            Returns the image array as passed or loaded and the the densepose Instances as results.
        """
        if isinstance(image, (str, Path)):
            image = read_image(image, format="BGR")

        return image, self.predict(image)

    def predict_video(self, video, video_loader_config=None, pbar=True):
        """Run inference to get the densepose results for a video.

        Parameters
        ----------
        video :
            numpy array (uint8) of a a video in BGR layout with time dimension first or path to a video
        video_loader_config : VideoLoaderConfig, optional
            A video loader config for loading videos (uses all defaults except pix_fmt="bgr24")
        pbar : bool, optional
            Whether to display a progress bar, by default True

        Returns
        -------
        tuple
            Tuple of (video_array, list of densepose results per frame)
        """
        if isinstance(video, (str, Path)):
            video = load_video_frames(video, config=video_loader_config)

        pbar = tqdm if pbar else lambda x, **kwargs: x

        return video, [
            self.predict_image(img)[1] for img in pbar(video, desc="Frames")
        ]  # just the predictions

    def predict(self, image_arr):
        """Main call to DensePose for inference. Runs inference on an image array.

        Parameters
        ----------
        image_arr : numpy array
            BGR image array

        Returns
        -------
        Instances
            Detection instances with boxes, scores, and densepose estimates.
        """
        with torch.no_grad():
            instances = self.predictor(image_arr)["instances"]

        return instances

    def serialize_video_output(self, instances, filename=None, write_embeddings=False):
        serialized = {
            "frames": [
                self.serialize_image_output(
                    frame_instances, filename=None, write_embeddings=write_embeddings
                )
                for frame_instances in instances
            ]
        }

        if filename is not None:
            with Path(filename).open("w") as f:
                json.dump(serialized, f, indent=2)

        return serialized

    def serialize_image_output(self, instances, filename=None, write_embeddings=False):
        """Convert the densepose output into Python-native objects that can
            be written and read with json.

        Parameters
        ----------
        instances : Instance
            The output from the densepose model
        filename : (str, Path), optional
            If not None, the filename to write the output to, by default None
        """
        if isinstance(instances, list):
            img_height, img_width = instances[0].image_size
        else:
            img_height, img_width = instances.image_size

        boxes = instances.get("pred_boxes").tensor
        scores = instances.get("scores").tolist()
        labels = instances.get("pred_classes").tolist()

        try:
            pose_result = instances.get("pred_densepose")
        except KeyError:
            pose_result = None

        # include embeddings + segmentation if they exist and they are requested
        write_embeddings = write_embeddings and (pose_result is not None)

        serialized = {
            "instances": [
                {
                    "img_height": img_height,
                    "img_width": img_width,
                    "box": boxes[i].cpu().tolist(),
                    "score": scores[i],
                    "label": {
                        "value": labels[i],
                        "mesh_name": self.vis_class_to_mesh_name[labels[i]],
                    },
                    "embedding": (
                        pose_result.embedding[[i], ...].cpu().tolist()
                        if write_embeddings
                        else None
                    ),
                    "segmentation": (
                        pose_result.coarse_segm[[i], ...].cpu().tolist()
                        if write_embeddings
                        else None
                    ),
                }
                for i in range(len(instances))
            ]
        }

        if filename is not None:
            with Path(filename).open("w") as f:
                json.dump(serialized, f, indent=2)

        return serialized

    def deserialize_output(self, instances_dict=None, filename=None):
        if filename is not None:
            with Path(filename).open("r") as f:
                instances_dict = json.load(f)

        # handle image case
        is_image = False
        if "frames" not in instances_dict:
            instances_dict = {"frames": [instances_dict]}
            is_image = True

        frames = []
        for frame in instances_dict["frames"]:
            heights, widths, boxes, scores, labels, embeddings, segmentations = zip(
                *[
                    (
                        i["img_height"],
                        i["img_width"],
                        i["box"],
                        i["score"],
                        i["label"]["value"],
                        i["embedding"] if i["embedding"] is not None else [np.nan],
                        i["segmentation"] if i["segmentation"] is not None else [np.nan],
                    )
                    for i in frame["instances"]
                ]
            )

            frames.append(
                Instances(
                    (heights[0], widths[0]),
                    pred_boxes=boxes,
                    scores=scores,
                    pred_classes=labels,
                    pred_densepose=DensePoseEmbeddingPredictorOutput(
                        embedding=torch.tensor(embeddings),
                        coarse_segm=torch.tensor(segmentations),
                    ),
                )
            )

        # if image or single frame, just return the instance
        if is_image:
            return frames[0]
        else:
            return frames

    def visualize_image(self, image_arr, outputs, output_path=None):
        """Visualize the pose information.

        Parameters
        ----------
        image_arr : numpy array (unit8) BGR
            The numpy array representing the image.
        outputs :
            The outputs from running DensePoseManager.predict*
        output_path : str or Path, optional
            If not None, write visualization to this path; by default None

        Returns
        -------
        numpy array (unit8) BGR
            DensePose outputs visualized on top of the image.
        """
        bw_image = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
        bw_image = np.tile(bw_image[:, :, np.newaxis], [1, 1, 3])
        data = self.vis_extractor(outputs)
        image_vis = self.visualizer.visualize(bw_image, data)

        if output_path is not None:
            cv2.imwrite(str(output_path), image_vis)

        return image_vis

    def anatomize_image(self, visualized_img_arr, outputs, output_path=None):
        """Convert the pose information into the percent of pixels in the detection
            bounding box that correspond to each part of the anatomy in an image.

        Parameters
        ----------
        visualized_img_arr : numpy array (unit8) BGR
            The numpy array the image after the texture has been visualized (by calling DensePoseManager.visualize_image).
        outputs :
            The outputs from running DensePoseManager.predict*

        Returns
        -------
        pandas.DataFrame
            DataFrame with percent of pixels of the bounding box that correspond to each anatomical part
        """
        if self.anatomy_color_mapping is None:
            raise ValueError(
                "No anatomy_color_mapping provided to track anatomy; did you mean to use a different MODEL?"
            )

        # no detections, return empty df for joining later (e.g., in anatomize_video)
        if not outputs:
            return pd.DataFrame([])

        _, _, N, bboxes_xywh, pred_classes = self.visualizer.extract_and_check_outputs_and_boxes(
            self.vis_extractor(outputs)
        )

        all_detections = []
        for n in range(N):
            x, y, w, h = bboxes_xywh[n].int().cpu().numpy()
            detection_area = visualized_img_arr[y : y + h, x : x + w]

            detection_stats = {
                name: (detection_area == np.array([[[color.B, color.G, color.R]]]))
                .all(axis=-1)
                .sum()
                / (h * w)  # calc percent of bounding box with this color
                for name, color in self.anatomy_color_mapping.iterrows()
            }

            detection_stats["x"] = x
            detection_stats["y"] = y
            detection_stats["h"] = h
            detection_stats["w"] = w

            all_detections.append(detection_stats)

        results = pd.DataFrame(all_detections)

        if output_path is not None:
            results.to_csv(output_path, index=False)

        return results

    def visualize_video(
        self, video_arr, outputs, output_path=None, frame_size=None, fps=30, pbar=True
    ):
        """Visualize the pose information on a video

        Parameters
        ----------
        video_arr : numpy array (unit8) BGR, time first
            The numpy array representing the video.
        outputs :
            The outputs from running DensePoseManager.predict*
        output_path : str or Path, optional
            If not None, write visualization to this path (should be .mp4); by default None
        frame_size : (innt, float), optional
            If frame_size is float, scale up or down by that float value; if frame_size is an integer,
            set width to that size and scale height appropriately.
        fps : int
            frames per second for output video if writing; defaults to 30
        pbar : bool
            display a progress bar

        Returns
        -------
        numpy array (unit8) BGR
            DensePose outputs visualized on top of the image.
        """
        pbar = tqdm if pbar else lambda x, **kwargs: x

        out_frames = np.array(
            [
                self.visualize_image(
                    image_arr,
                    output,
                )
                for image_arr, output in pbar(
                    zip(video_arr, outputs), total=video_arr.shape[0], desc="Visualize frames"
                )
            ]
        )

        if output_path is not None:
            # get new size for output video if scaling
            if frame_size is None:
                frame_size = video_arr.shape[2]  # default to same size

            # if float, scale as a multiple
            if isinstance(frame_size, float):
                frame_width = round(video_arr.shape[2] * frame_size)
                frame_height = round(video_arr.shape[1] * frame_size)

            # if int, use as width of the video and scale height proportionally
            elif isinstance(frame_size, int):
                frame_width = frame_size
                scale = frame_width / video_arr.shape[2]
                frame_height = round(video_arr.shape[1] * scale)

            # setup output for writing
            output_path = output_path.with_suffix(".mp4")
            out = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                max(1, int(fps)),
                (frame_width, frame_height),
            )

            for f in pbar(out_frames, desc="Write frames"):
                if (f.shape[0] != frame_height) or (f.shape[1] != frame_width):
                    f = cv2.resize(
                        f,
                        (frame_width, frame_height),
                        # https://stackoverflow.com/a/51042104/1692709
                        interpolation=(
                            cv2.INTER_LINEAR if f.shape[1] < frame_width else cv2.INTER_AREA
                        ),
                    )
                out.write(f)

            out.release()

        return out_frames

    def anatomize_video(self, visualized_video_arr, outputs, output_path=None, fps=30):
        """Convert the pose information into the percent of pixels in the detection
            bounding box that correspond to each part of the anatomy in a video.

        Parameters
        ----------
        visualized_video_arr : numpy array (unit8) BGR
            The numpy array the video after the texture has been visualized (by calling DensePoseManager.visualize_video).
        outputs :
            The outputs from running DensePoseManager.predict*

        Returns
        -------
        numpy array (unit8) BGR
            DensePose outputs visualized on top of the image.
        """
        all_detections = []

        for ix in range(visualized_video_arr.shape[0]):
            detection_df = self.anatomize_image(visualized_video_arr[ix, ...], outputs[ix])
            detection_df["frame"] = ix
            detection_df["seconds"] = ix / fps
            all_detections.append(detection_df)

        results = pd.concat(all_detections)

        if output_path is not None:
            results.to_csv(output_path, index=False)

        return results
