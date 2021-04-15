from pathlib import Path

import cv2
import numpy as np
import skimage.io
import tensorflow as tf
from tensorflow.keras.utils import get_file
from tqdm import tqdm

import zamba
from zamba.utils import load_video

tf.logging.set_verbosity(tf.logging.ERROR)


class MegaDetector:
    """ Instantiate and detect on images or videos using AI for Earth's MegaDetector.

        Read more documentation at https://github.com/microsoft/CameraTraps/blob/master/megadetector.md and download
        the weights from "https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb"

        Attributes:
            model (tensorflow.framework.ops.Graph)
            sess (tensorflow.client.session.Session)
            confidence_threshold (float): Only keep bounding boxes with scores above this threshold
    """
    MODEL_URL = "https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb"
    FEATURE_NAMES = ["n_key_frames", "h", "w", "n_detections", "total_area"]

    def __init__(
        self,
        model_path=None,
        confidence_threshold=0.85,
    ):
        """
            Args:
                model_path (str, optional): Path to mega detector weights (downloaded from
                    "https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb"). If
                    omitted, the weights will be downloaded automatically
                confidence_threshold (float): Only keep bounding boxes with scores above this threshold
        """
        if model_path is None:
            model_path = self._get_model()

        self.model = self.load_model(str(model_path))
        self.sess = tf.Session(graph=self.model)
        self.confidence_threshold = confidence_threshold

    def __del__(self):
        if hasattr(self, "sess"):
            self.sess.close()

    def detect_image(self, image):
        """
        """
        if isinstance(image, (str, Path)):
            image = skimage.io.imread(image)

        image_expanded = np.expand_dims(image, axis=0)
        image_tensor = self.model.get_tensor_by_name("image_tensor:0")
        box = self.model.get_tensor_by_name("detection_boxes:0")
        score = self.model.get_tensor_by_name("detection_scores:0")
        clss = self.model.get_tensor_by_name("detection_classes:0")
        num_detections = self.model.get_tensor_by_name("num_detections:0")

        # Actual detection
        (box, score, clss, num_detections) = self.sess.run(
            [box, score, clss, num_detections],
            feed_dict={image_tensor: image_expanded},
        )

        boxes = box[score > self.confidence_threshold]
        scores = score[score > self.confidence_threshold]

        return boxes, scores

    def detect_video(
        self,
        video,
        key_frames_only=True,
    ):
        """Detects animals in a video

        Args:
            video (str, Path, or np.ndarray): Path to a video or an array with frames as the first dimension
            video_name (str, optional): If video is provided as an array and write_boxes_npz is true, must provide a
                name for the file containing the output bounding boxes
            key_frames_only (bool): Only load the key frames of the video

        Returns:
            A list of bounding boxes for each frame and a list of scores for each frame
        """
        if isinstance(video, (str, Path)):
            video = load_video(video, key_frames_only=key_frames_only)

        boxes = []
        scores = []

        for image in tqdm(video, desc=f"Frames", total=video.shape[0]):
            image_boxes, image_scores = self.detect_image(image)
            boxes.append(image_boxes)
            scores.append(image_scores)

        return boxes, scores

    def compute_features(
        self,
        videos,
        **kwargs,
    ):
        """Computes number of detections and total detected area for a list of videos

        Args:
            videos (list of Path or np.ndarray): A list of video paths or arrays to be processed
            kwargs (dict): Additional parameters passed to `detect_video`

        Returns:
            np.ndarray: An array with shape (num videos, 2) where the first column contains the number of detections
                for each video and the second column contains the total area of detections for each video
        """
        if not kwargs.get("key_frames_only", True):
            raise ValueError("Features only supported for `key_frame_only=True`")

        features = []
        for i, video in enumerate(videos):
            if isinstance(video, (str, Path)):
                video = load_video(video, key_frames_only=True)

            n_key_frames, height, width = video.shape[:3]
            boxes, scores = self.detect_video(video, **kwargs)
            n_detections, area = MegaDetector.compute_n_detections_and_areas(boxes, height, width)

            features.append([
                n_key_frames, height, width, n_detections, area
            ])

        features = np.array(features, dtype=np.int32)

        return features

    def detect_image_directory(
        self, directory, extensions=None, recursive=True, key_frames_only=True,
    ):
        # setup recursive or not glob string
        glob_str = "**/*" if recursive else "*"

        # limit extensions if they are passed
        if isinstance(extensions, str):
            extensions = [extensions]

        if extensions is not None:
            extensions = [e.strip(".") for e in extensions]
            glob_searches = [glob_str + f".{e}" for e in extensions]
        else:
            glob_searches = [glob_str]

        # for every kind of file, build up the list of videos
        img_paths = []
        for formatted_glob in glob_searches:
            img_paths += list(Path(directory).glob(formatted_glob))

        # detect on the images
        output_boxes = {}
        output_scores = {}
        for i in img_paths:
            boxes, scores = self.detect_image(i, key_frames_only=key_frames_only)
            output_boxes[str(i)], output_scores[str(i)] = boxes, scores

        return output_boxes, output_scores

    def detect_video_directory(
        self, directory, extensions=None, recursive=True, key_frames_only=True,
    ):
        # setup recursive or not glob string
        glob_str = "**/*" if recursive else "*"

        # limit extensions if they are passed
        if isinstance(extensions, str):
            extensions = [extensions]

        if extensions is not None:
            extensions = [e.strip(".") for e in extensions]
            glob_searches = [glob_str + f".{e}" for e in extensions]
        else:
            glob_searches = [glob_str]

        # for every kind of file, build up the list of videos
        vid_paths = []
        for formatted_glob in glob_searches:
            vid_paths += list(Path(directory).glob(formatted_glob))

        # detect on the videos
        output_boxes = {}
        output_scores = {}
        for v in tqdm(vid_paths, desc="Processing videos"):
            try:
                boxes, scores = self.detect_video(v, key_frames_only=key_frames_only)
                output_boxes[str(v)], output_scores[str(v)] = boxes, scores
            except Exception as e:
                self.logger.debug("Error processing %s\n%s", str(v), e)
                output_boxes[str(v)], output_scores[str(v)] = [np.array([])], [np.array([])]

        return output_boxes, output_scores

    @staticmethod
    def compute_n_detections_and_areas(boxes, h, w):
        """Computes the number of detections and total detected area from object detection bounding boxes

        Args:
            boxes (list of np.ndarray): A list of bounding boxes
            h (int): Height of the video in pixels
            w (int): Width of the video in pixels

        Returns:
            The number of detections and the total detected area
        """
        total_area = 0
        n_detections = 0

        for box in boxes:
            if len(box) > 0:
                y1 = int(box[0, 0] * h)
                x1 = int(box[0, 1] * w)
                y2 = int(box[0, 2] * h)
                x2 = int(box[0, 3] * w)

                total_area += (y2 - y1) * (x2 - x1)
                n_detections += 1

        return n_detections, total_area

    @staticmethod
    def visualize_video(video, boxes, key_frames_only=True, sleep=0.2):
        import zamba.visualization as viz

        if isinstance(video, (str, Path)):
            video = load_video(video, key_frames_only=key_frames_only)

        def draw_box(frame, box):
            try:
                if len(box):
                    y1 = int(box[0, 0] * frame.shape[0])
                    x1 = int(box[0, 1] * frame.shape[1])
                    y2 = int(box[0, 2] * frame.shape[0])
                    x2 = int(box[0, 3] * frame.shape[1])
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            except Exception as e:
                print(e)

            finally:
                return frame

        def callback(frame, ix):
            f = draw_box(frame, boxes[ix])
            print(f"frame {ix + 1} / {video.shape[0]}")
            return f

        viz.display_video(video, frame_cb=callback, sleep=sleep)

    def load_model(self, checkpoint):
        """
        Load a detection model (i.e., create a graph) from a .pb file
        """
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(checkpoint, "rb") as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")

        return detection_graph

    def _get_model(self, model_url=None):
        if model_url is None:
            model_url = self.MODEL_URL

        file_name = model_url.rsplit("/", 1)[-1]

        cache_subdir = "megadetector"
        model_path = zamba.config.cache_dir / cache_subdir / file_name

        if not model_path.exists():
            model_path = get_file(
                fname=file_name,
                origin=model_url,
                cache_dir=zamba.config.cache_dir,
                cache_subdir=cache_subdir,
                extract=True,
            )

        return model_path
