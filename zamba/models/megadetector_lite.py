""" Inference with ONNX megadetector lite
"""
from io import BytesIO
from pathlib import Path

from cloudpathlib import CloudPath
import cv2
import numpy as np
import onnx
import onnxruntime as rt
from tqdm import tqdm

from zamba_algorithms.settings import DATA_DIRECTORY


LOCAL_MD_LITE_MODEL = str(
    DATA_DIRECTORY
    / "artifacts"
    / "spikes"
    / "megadetector"
    / "yolov4"
    / "artifacts"
    / "yolov4_1_3_480_480_static.onnx"
)


def add_nms_nodes_to_onnx_model(
    model, model_destination, score_threshold=0.01, iou_threshold=0.3, max_box_per_class=10
):
    """Add layers to perform non-maximal suppression for darknet object detection model in ONNX.

    `model` should be a loaded ONNX-converted Darknet object dection model or path to one.
    We expect the shape of the output to be [batch_size, detections, 6], where the last
    dimension is [cx, cy, w, h, objectness_prob, class_prob]. Currently we only
    support a single class_prob.

    `model_destination` should be a writeable buffer.

    `score_threshold` is the threshold for (objectness_prob * class_prob)
    `iou_threshold` is the threshold IOU value for boxes to be considered overlapping
    `max_box_per_class` is the number of boxes for a given class that we will return
    """
    if isinstance(model, (str, Path)):
        model = onnx.load(str(model))

    # track the nodes and constants added to ONNX model
    added_constants = []
    added_nodes = []

    # for scores, dimension with number of candidates needs to be last before entering NonMaxSuppression
    reshape = onnx.helper.make_tensor("reshape_dims", onnx.TensorProto.INT64, [3], [1, 1, -1])
    added_constants += [reshape]

    reshape_node = onnx.helper.make_node(
        "Reshape",
        inputs=["confs", "reshape_dims"],
        outputs=["reshaped_scores"],
    )
    added_nodes += [reshape_node]

    # remove class dimension from boxes so [batch, detections, classes, boxes] -> [batch, detections, boxes]
    sq_node = onnx.helper.make_node(
        "Squeeze", inputs=["boxes"], outputs=["squeezed_boxes"], axes=[2]
    )
    added_nodes += [sq_node]

    # NMS node - to actually do the NMS
    score_threshold = onnx.helper.make_tensor(
        "score_threshold", onnx.TensorProto.FLOAT, [1], [score_threshold]
    )
    iou_threshold = onnx.helper.make_tensor(
        "iou_threshold", onnx.TensorProto.FLOAT, [1], [iou_threshold]
    )
    max_output_boxes_per_class = onnx.helper.make_tensor(
        "max_output_boxes_per_class", onnx.TensorProto.INT64, [1], [max_box_per_class]
    )  # number individual detections
    added_constants += [score_threshold, iou_threshold, max_output_boxes_per_class]

    nms_node = onnx.helper.make_node(
        "NonMaxSuppression",
        [
            "squeezed_boxes",
            "reshaped_scores",
            "max_output_boxes_per_class",
            "iou_threshold",
            "score_threshold",
        ],
        ["selected_indices"],
        center_point_box=1,
    )
    added_nodes += [nms_node]

    # All the outputs are [0, 0, ix] NMS requires 3 dimensions; select just the indices out
    split_box_indices = onnx.helper.make_node(
        "Split",
        inputs=["selected_indices"],  # name from existing onnx model
        outputs=["batch_dim", "spatial_dim", "col_of_indices"],
        axis=1,
        split=[1, 1, 1],
    )
    added_nodes += [split_box_indices]

    # Squeeze to a single dimension instead of 2
    squeeze_box_indices = onnx.helper.make_node(
        "Squeeze",
        inputs=["col_of_indices"],  # name from existing onnx model
        outputs=["actual_indices"],
    )
    added_nodes += [squeeze_box_indices]

    # Select boxes with indices from NMS
    gather_node = onnx.helper.make_node(
        "Gather",
        inputs=["squeezed_boxes", "actual_indices"],
        outputs=["selected_boxes"],
        axis=1,
    )
    added_nodes += [gather_node]

    # Select confidences with the indices from NMS
    gather_confs_node = onnx.helper.make_node(
        "Gather",
        inputs=["confs", "actual_indices"],
        outputs=["selected_confs"],
        axis=1,
    )
    added_nodes += [gather_confs_node]

    # add to initializers for the constants we added
    for const in added_constants:
        model.graph.initializer.append(const)

    # add to the list of graph nodes
    for n in added_nodes:
        model.graph.node.append(n)

    # remove old outputs
    for _ in range(len(model.graph.output)):
        model.graph.output.pop()

    # set output to selected_detections
    box_info = onnx.helper.make_tensor_value_info(
        "selected_boxes", onnx.TensorProto.FLOAT, shape=[1, -1, 4]
    )
    conf_info = onnx.helper.make_tensor_value_info(
        "selected_confs", onnx.TensorProto.FLOAT, shape=[1, -1, 1]
    )

    model.graph.output.append(box_info)
    model.graph.output.append(conf_info)

    # # check that it works and re-save
    onnx.checker.check_model(model)

    onnx.save_model(model, model_destination)
    return model_destination


class MegadetectorLite:
    def __init__(
        self,
        onnx_model=LOCAL_MD_LITE_MODEL,
        add_nms=True,
        nms_score_threshold=0.01,
    ):
        """MegadetectorLite object

        Args:
            onnx_model (Path, optional): Path to ONNX model. Defaults to LOCAL_MD_LITE_MODEL, which will be
                downloaded from that S3 path if it does not exist.
                S3 path: s3://drivendata-client-zamba/data/artifacts/spikes/megadetector/yolov4/artifacts/yolov4_1_3_480_480_static.onnx
            add_nms (bool, optional): Perform non-maximal suppression. Usually you will want this to avoid overlapping
                detections of the same animal. If you know you do not need this, you can turn it off by setting to False.
                Defaults to True.
            nms_score_threshold (float, optional): [0-1] Probability threshold for above. Not recommended to change unless
                there are an excessive number of extra bounding boxes and false positives. Adjust the threshold in
                detect_* or filter_frames instead. Defaults to 0.01.
        """
        onnx_model = Path(onnx_model)
        if not onnx_model.exists():
            # add parent directories if downloading locally
            onnx_model.parent.mkdir(parents=True, exist_ok=True)
            onnx_model = CloudPath(
                "s3://drivendata-client-zamba/data/artifacts/spikes/megadetector/yolov4/artifacts/yolov4_1_3_480_480_static.onnx"
            ).download_to(
                onnx_model.parent
            )  # DL + local path for loading in onnx

        if add_nms:
            tmp_model_buffer = BytesIO()

            add_nms_nodes_to_onnx_model(
                onnx_model,
                tmp_model_buffer,
                score_threshold=nms_score_threshold,
            )

            tmp_model_buffer.seek(0)
            onnx_model = tmp_model_buffer.read()

        if isinstance(onnx_model, (str, Path, bytes)):
            # enable optimization to supress CleanUnusedInitializers warnings
            sess_options = rt.SessionOptions()
            sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

            self.onnx_session = rt.InferenceSession(onnx_model, sess_options=sess_options)
        else:
            self.onnx_session = onnx_model

        # B, C, W, H; usually: (1, 3, 480, 480)
        self.input_shape = tuple(
            self.onnx_session.get_inputs()[0].shape
        )  # onnx provides list, but should be tuple

    def _check_input_shape(self, arr, raise_on_error=True):
        try:
            # last three dimensions should be C, W, H
            if not np.all(arr.shape[-3:] == self.input_shape[-3:]):
                raise ValueError(
                    f"Model expects input of shape: {self.input_shape}; passed {arr.shape}"
                )

            if arr.dtype != np.float32:
                raise ValueError(f"Array must be float 32, but passed {arr.dtype}")

            return True
        except ValueError:
            if raise_on_error:
                raise  # reraise
            else:
                return False

    def _preprocess(self, video_arr):
        shape = self.input_shape[2:]
        # rescale to specified shape. assume input is (time, height, width, channels)
        if video_arr.shape[1:3] != shape:
            video_arr = np.array(
                [
                    cv2.resize(image_arr, shape, interpolation=cv2.INTER_AREA)
                    for image_arr in video_arr
                ]
            )

        # convert to float32, scale 0-1 from 0-255, reorder axes so channels are before height/width
        onnx_frames = np.moveaxis(video_arr.astype(np.float32) / 255.0, 3, 1)

        return onnx_frames

    def detect_video(
        self,
        frames_arr,
        pbar=False,
        preprocess_inputs=True,
    ):
        if preprocess_inputs:
            frames_arr = self._preprocess(frames_arr)

        pbar = tqdm if pbar else lambda x: x

        return [
            self.detect_image(img_arr=frames_arr[f, ...]) for f in pbar(range(frames_arr.shape[0]))
        ]

    def detect_image(self, img_arr=None, preprocess_inputs=True):
        # add expected "batch" dimension for onnx model if necessary
        if img_arr.ndim == 3:
            img_arr = img_arr[np.newaxis, ...]

        if preprocess_inputs and not self._check_input_shape(img_arr, raise_on_error=False):
            img_arr = self._preprocess(img_arr)

        boxes, scores = self.onnx_session.run(
            None,
            {
                self.onnx_session.get_inputs()[0].name: img_arr,
            },
        )

        # remove batch dimension if present (it is always 1 in this pipeline)
        if boxes.ndim == 3:
            boxes = boxes[0, ...]
            scores = scores[0, ...]

        return (
            boxes,  # list of detections (x1, y1, x2, y2)
            scores,  # list of scores for each detection
        )

    @staticmethod
    def filter_frames(
        frames_arr,
        probabilities,
        score_threshold=0.25,
        n_frames=None,
        fill_mode="repeat",
        sort_by_time=True,
        random_state=55,
        min_over_threshold=None,
    ):
        """Filter video frames using megadetector lite. Which frames are returned depends on the
        fill_mode, how many frames are above the score_threshold, and how many frames have non-zero
        probability of detection. If more than n_frames are above the threshold, the top n_frames are returned.
        Otherwise add to those over threshold based on fill_mode:
            - 'repeat': if frames_over_threshold > min_over_threshold, randomly resample qualifying frames
                to get to n_frames
            - 'score_sorted': take up to n_frames in sort order (even if some have zero probability)
            - 'nonzero': take the top n_frames among all of the frames with nonzero probability
            - 'nonzero_repeat': take the top n_frames with nonzero probability, randomly resample
                qualifying frames to get to n_frames if there are not enough
            - 'weighted_euclidean': if frames_over_threshold > min_over_threshold, sample the remaining
                frames weighted by their euclidean distance in time to the frames over the threshold
            - 'weighted_prob': sample the remaining frames weighted by their predicted probability
        If none of these conditions are met, returns all of the frames above the threshold.

        Args:
            frames_arr (np.ndarray): Array of video frames to filter
            probabilities (list[tuples]): List of detection results for each frame. Each element is a tuple of
                the list of bounding boxes [array(x1, y1, x2, y2)] and the detection probabilities, both as float32
            score_threshold (float, optional): Score threshold to use for filtering. Defaults to 0.25.
            n_frames (int, optional): Max number of frames to return. If None returns all frames above the threshold.
                Defaults to None.
            fill_mode (str, optional): Mode for upsampling if the number of frames above the threshold is less than
                n_frames. Defaults to "repeat"
            sort_by_time (bool, optional): Whether to sort the selected frames by time (original order) before returning.
                If False, returns frames sorted by score (descending). Defaults to True.
            random_state (int, optional): Random state for random number generator. Defaults to 55.
            min_over_threshold (int or float, optional): Minimum number of frames to repeat or use in weighted distance
                sampling. If None, set to n_frames/5. If False, not applied (set to 0). Defaults to None.

        Returns:
            np.ndarray: An array of video frames of length n_frames or shorter
        """

        frame_scores = np.array(
            [(np.max(s) if (s.shape[0] != 0) else 0) for _, s in probabilities]
        )  # reduce to one score per frame
        frame_sort_order = np.argsort(frame_scores)[::-1]  # ascending is default

        frames_over_threshold = (frame_scores > score_threshold).sum()
        frames_over_zero = (frame_scores > 0).sum()

        rng = np.random.RandomState(random_state)

        if min_over_threshold is None and n_frames is not None:
            min_over_threshold = round(n_frames / 5)
        elif min_over_threshold is False:
            min_over_threshold = 0

        # when enough frames are over the threshold, select n_frames with highest prob
        if n_frames is None:
            selected = frame_sort_order[:frames_over_threshold]

        elif frames_over_threshold >= n_frames:
            selected = frame_sort_order[:n_frames]

        # repeat frames that are above threshold to get to n_frames
        elif (
            (fill_mode == "repeat")
            and (frames_over_threshold >= min_over_threshold)
            and (frames_over_threshold > 0)
        ):
            selected = frame_sort_order[:frames_over_threshold]
            repeats = rng.choice(selected, n_frames - selected.shape[0], replace=True)
            selected = np.concatenate((selected, repeats))

        # take all nonzero (above nms_score_threshold from instantiation) up to n_frames
        elif (fill_mode == "nonzero") and (frames_over_zero > 0):
            selected = frame_sort_order[: min(frames_over_zero, n_frames)]

        # take frames in sorted order up to n_frames, even if score is zero
        elif fill_mode == "score_sorted":
            selected = frame_sort_order[:n_frames]

        # repeat frames that are nonzero to get to n_frames
        elif (
            (fill_mode == "nonzero_repeat")
            and (frames_over_zero >= min_over_threshold)
            and (frames_over_zero > 0)
        ):
            selected = frame_sort_order[:frames_over_zero]
            repeats = rng.choice(selected, n_frames - selected.shape[0], replace=True)
            selected = np.concatenate((selected, repeats))

        # sample up to n_frames, prefer points closer to frames with detection
        elif (
            (fill_mode == "weighted_euclidean")
            and (frames_over_threshold >= min_over_threshold)
            and (frames_over_threshold > 0)
        ):
            selected = frame_sort_order[:frames_over_threshold]
            sample_from = frame_sort_order[frames_over_threshold:]
            # take one over euclidean distance to all points with detection
            weights = [
                1 / np.linalg.norm(selected - np.repeat([fr], frames_over_threshold))
                for fr in sample_from
            ]
            # normalize weights
            weights = weights / np.sum(weights)
            sampled = rng.choice(
                sample_from, n_frames - frames_over_threshold, replace=False, p=weights
            )

            selected = np.concatenate((selected, sampled))

        # sample up to n_frames, weight by predicted probability - only if some frames have nonzero prob
        elif (fill_mode == "weighted_prob") and (frames_over_zero > 0):
            selected = frame_sort_order[:frames_over_threshold]
            sample_from = frame_sort_order[frames_over_threshold:]
            weights = frame_scores[sample_from] / np.sum(frame_scores[sample_from])
            sampled = rng.choice(
                sample_from, n_frames - frames_over_threshold, replace=False, p=weights
            )

            selected = np.concatenate((selected, sampled))

        # if no other criteria are met, just return the frames with probability above the threshold
        else:
            selected = frame_sort_order[:frames_over_threshold]

        # sort the selected images back into their original order
        if sort_by_time:
            selected = sorted(selected)

        return frames_arr[selected]
