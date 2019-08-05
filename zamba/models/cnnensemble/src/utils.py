import time
from contextlib import contextmanager
import numpy as np
import skvideo.io
import skimage.transform
from zamba.models.cnnensemble.src import config


def validate_video(path, n_frames=1):
    """Quickly checks whether a video file is valid by reading the first and last frames

    Args:
        path (str): Path to a video file
        n_frames (int): Number of frames to load

    Returns:
        bool: True if video is valid, False if video is invalid
    """
    try:
        video = skvideo.io.vreader(path)

        for i in range(n_frames):
            _ = next(video)

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


def preprocessed_input_to_img_resnet(x):
    # Zero-center by mean pixel
    x = x.copy()
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # 'BGR' -> RGB
    img = x.copy()
    img[:, :, 0] = x[:, :, 2]
    img[:, :, 1] = x[:, :, 1]
    img[:, :, 2] = x[:, :, 0]
    return img / 255.0


@contextmanager
def timeit_context(name):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    print('[{}] finished in {} ms'.format(name, int(elapsedTime * 1000)))


def chunks(l, n, add_empty=False):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l) // n * n + n - 1, n):
        if len(l[i:i + n]):
            yield l[i:i + n]
    if add_empty:
        yield []


def lock_layers_until(model, first_trainable_layer, verbose=False):
    found_first_layer = False
    for layer in model.layers:
        if layer.name == first_trainable_layer:
            found_first_layer = True

        if verbose and found_first_layer and not layer.trainable:
            print('Make layer trainable:', layer.name)
            layer.trainable = True

        layer.trainable = found_first_layer


def print_stats(title, array):
    print('{} shape:{} dtype:{} min:{} max:{} mean:{} median:{}'.format(
        title,
        array.shape,
        array.dtype,
        np.min(array),
        np.max(array),
        np.mean(array),
        np.median(array)
    ))


def load_video_clip_frames(video_fn, frames_numbers, output_size):
    """
    Load video clip frames.
    Load frames or requested frames_numbers and resize if necessary to output_size

    :param video_fn: path to video clip
    :param frames_numbers: list of frame numbers to load
    :param output_size: (rows, cols) tuple, size of loaded image
    :return: ndarray of shape (len(frames_numbers), rows, cols, 3)
    """
    X = np.zeros(shape=(len(frames_numbers),) + output_size + (3,), dtype=np.float32)

    v = skvideo.io.vread(str(video_fn))
    valid_frames = 0

    for i, frame_num in enumerate(frames_numbers):
        try:
            frame = v[frame_num]
            if frame.shape[:2] != output_size:
                frame = skimage.transform.resize(frame,
                                                 output_shape=output_size,
                                                 order=1,
                                                 mode='constant',
                                                 preserve_range=True).astype(np.float32)
            else:
                frame = frame.astype(np.float32)
            X[i] = frame
            valid_frames += 1
        except IndexError:
            if valid_frames > 0:
                X[i] = X[i % valid_frames]
            else:
                X[i] = 0.0
    return X


if __name__ == '__main__':
    pass
