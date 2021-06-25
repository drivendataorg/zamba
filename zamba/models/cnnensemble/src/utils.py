from contextlib import contextmanager
from pathlib import Path
import time

import numpy as np
import skvideo.io
import skimage.transform


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


def load_video_clip_frames(video_fn, frames_numbers, output_size, image_extensions=[".jpg", ".png"]):
    """
    Load video clip frames.
    Load frames or requested frames_numbers and resize if necessary to output_size

    :param video_fn: path to video clip
    :param frames_numbers: list of frame numbers to load
    :param output_size: (rows, cols) tuple, size of loaded image
    :return: ndarray of shape (len(frames_numbers), rows, cols, 3)
    """
    # if path is to an image, load as image and repeat to the expected number of frames
    if Path(video_fn).suffix.lower() in image_extensions:
        return load_images_as_frames(video_fn, frames_numbers, output_size)

    X = np.zeros(shape=(len(frames_numbers),) + output_size + (3,), dtype=np.float32)

    videogen = skvideo.io.vreader(str(video_fn))

    valid_frames = 0
    for frame_ix, frame in enumerate(videogen):
        if frame_ix in frames_numbers:
            if frame.shape[:2] != output_size:
                frame = skimage.transform.resize(frame,
                                                 output_shape=output_size,
                                                 order=1,
                                                 mode='constant',
                                                 preserve_range=True).astype(np.float32)
            else:
                frame = frame.astype(np.float32)

            X[valid_frames] = frame
            valid_frames += 1

        if frame_ix == max(frames_numbers):
            break

    return X


def load_images_as_frames(video_fn, frames_numbers, output_size):
    """
    Load an image in the video frame format expected by the model.
    Load image and repeat len(frames_numbers) times; resize if necessary to output_size

    The resulting array can be used by the video models directly.

    :param video_fn: path to video clip
    :param frames_numbers: list of frame numbers to load
    :param output_size: (rows, cols) tuple, size of loaded image
    :return: ndarray of shape (len(frames_numbers), rows, cols, 3)
    """

    frame = skimage.io.imread(str(video_fn))

    frame = skimage.transform.resize(
        frame,
        output_shape=output_size,
        order=1,
        mode='constant',
        preserve_range=True
    ).astype(np.float32)

    return np.repeat(frame[np.newaxis, ...], len(frames_numbers), axis=0)


if __name__ == '__main__':
    pass
