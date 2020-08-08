import json
from math import ceil
from subprocess import check_output

import cv2
import av
import numpy as np


def validate_video(path, n_frames=1):
    """Quickly checks whether a video file is valid by reading the first and last frames

    Args:
        path (str): Path to a video file
        n_frames (int): Number of frames to load

    Returns:
        bool: True if video is valid, False if video is invalid
    """
    try:
        load_video(path, n_frames=n_frames)
        is_valid = True

    except Exception:
        is_valid = False

    return is_valid


def get_valid_videos(paths):
    """Splits videos into valid and invalid

    Args:
        paths (list of str): A list of paths to videos

    Returns:
        A list of valid video paths and a list of invalid video paths
    """
    valid_videos, invalid_videos = [], []

    for path in paths:
        if validate_video(path):
            valid_videos.append(path)
        else:
            invalid_videos.append(path)

    return valid_videos, invalid_videos


def get_video_info(video_path, verbose=False):
    command = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        str(video_path),
    ]

    output_string = check_output(command).decode()
    info = json.loads(output_string)

    if verbose:
        print(output_string)

    video_streams = [s for s in info["streams"] if s["codec_type"] == "video"]

    if len(video_streams) > 1:
        raise ValueError(f"Multiple video streams detected for video {video_path}.")

    stream_data = video_streams[0]

    # frame rate is expressed as a fraction so we divide and round
    numr, denom = map(int, stream_data["r_frame_rate"].split("/"))
    fps = ceil(numr / denom)
    length_s = ceil(float(stream_data["duration"]))
    return {
        "fps": fps,
        "height": stream_data["height"],
        "width": stream_data["width"],
        "length_s": length_s,
        "nb_frames": length_s * fps,
    }


def frame_apply(video, method, *args, inplace=False, **kwargs):
    """Applies a function to each frame of a video

    Args:
        video (np.ndarray): An array whose first dimension is the frames of a video. The remaining dimensions can be
            anything that is comparible with the function being applied
        method (callable): The function to be called on each frame.
        inplace (bool): If true, directly update the input video. If false, make a copy. In this case `method` must
            return frames of the same size and channel depth as the original video

    Returns:
        An array containing the new video

    """
    if not inplace:
        out = np.zeros_like(video)
    else:
        out = video

    for i in range(video.shape[0]):
        out[i, ...] = method(video[i, ...], *args, **kwargs)

    return out


def convert_colorspace(array, conversion):
    """Converts an image of video from one colorspace to another.

    Args:
        array (np.ndarray): An image with shape (width, height, channels) or video with shape (frames, width, height,
            channels)
        conversion (str or int): 'hsv2rgb' 'rgb2hsv' or an integer representing an opencv color conversion, e.g.,
            cv2.COLOR_RGB2HSV

    Returns:
        An array of the input image or video converted to a new colorspace.
    """
    if conversion.lower() == "hsv2rgb":
        conversion = cv2.COLOR_HSV2RGB
    elif conversion.lower() == "rgb2hsv":
        conversion = cv2.COLOR_RGB2HSV
    elif not isinstance(conversion, int):
        raise ValueError(
            "`conversion` must be 'hsv2rgb', 'rgb2hsv' or an integer representing an opencv color conversion"
        )

    shape = array.shape
    output_array = cv2.cvtColor(
        array.reshape(
            -1, 1, 3
        ),  # trick opencv into thinking this is an image with dimensions (h, w, 3)
        conversion,
    ).reshape(shape)

    return output_array


def load_video(
    path,
    h=None,
    w=None,
    n_frames=None,
    key_frames_only=False,
    open_options=None,
    grayscale=False,
):
    """Loads a video using pyav

    Args:
        path (str): Path to the video
        h (int, optional): When to provide?
        w (int, optional): When to provide?
        n_frames (int, optional): Number of frames to load, if None load the entire video
        key_frames_only (bool): Only load the video key frames
        open_options (dict, optional): Additional arguments passed to `av.open`

    Returns:
        A numpy array with shape (number of frames, height, width, 3) containing the video
    """
    if open_options is None:
        open_options = dict()

    with av.open(str(path), options=open_options) as container:
        container.streams.video[0].thread_type = 'AUTO'  # Go faster!

        if not all([h, w, n_frames]):
            if h is None:
                h = container.streams.video[0].height

            if w is None:
                w = container.streams.video[0].width

            if n_frames is None:
                n_frames = container.streams.video[0].frames

        if key_frames_only:
            container.streams.video[0].codec_context.skip_frame = 'NONKEY'

        shape = (n_frames, h, w, 3) if not grayscale else (n_frames, h, w)
        video_array = np.zeros(shape, dtype=np.uint8)

        i = 0
        for frame in container.decode(video=0):
            if grayscale:
                video_array[i] = np.array(frame.to_image().convert('L'))
            else:
                video_array[i] = frame.to_rgb().to_ndarray()

            i += 1
            if i >= n_frames:
                break

    return video_array[:i, ...]


def unique_processed_path(
    input_path, directory, file_extension=None, existing_paths=None,
):
    """Generates a path that can serve as a destination path after processing an input file.

    One solution is to generate a random destination path, but that can make it difficult to easily tell how raw and
    processed files are related. Instead, try to use the input file's name as the name of the output file, and if there
    is a file name collision, add a suffix until the names do not collide.

    Args:
        input_path (pathlib.Path): The name of the raw file. The output name will have a similar name to this.
        directory (pathlib.Path): Directory of the destination file path
        file_extension (str, optional): If none give, the extension of the input path will be used
        existing_paths (list of pathlib.Path, optional): The destination path will not collide with the file paths in
            this list

    Returns:
        pathlib.Path: An output path that does not yet exist and is not in the `existing_paths`


    """
    def is_valid(path):
        return (not path.exists()) and (path not in existing_paths)

    if file_extension is None:
        file_extension = input_path.suffix

    if existing_paths is None:
        existing_paths = []

    output_path = directory / input_path.with_suffix(file_extension).name
    stem = output_path.stem
    unique_suffix = 0

    while not is_valid(output_path):
        output_path = output_path.with_name(f"{stem}_{unique_suffix}{file_extension}")
        unique_suffix += 1

    return output_path


def unique_processed_paths(
    input_paths, directory, file_extension=None, existing_paths=None,
):
    if existing_paths is None:
        existing_paths = []

    output_paths = []
    for input_path in input_paths:
        output_path = unique_processed_path(
            input_path,
            directory,
            file_extension=file_extension,
            existing_paths=existing_paths + output_paths,
        )

        output_paths.append(output_path)

    return output_paths
