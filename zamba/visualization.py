from pathlib import Path
import time

import cv2
from io import BytesIO
from IPython.display import display, Image, clear_output
import numpy as np
import PIL.Image


def display_array(array, fmt="png"):
    """Displays a numpy array as an image in a Jupyter notebook"""
    array = np.uint8(array)
    byte = BytesIO()

    if len(array.shape) == 2:
        mode = 'L'
    else:
        mode = 'RGB'

    PIL.Image.fromarray(array, mode=mode).save(byte, fmt)
    display(Image(data=byte.getvalue()))


def display_video(vid, frame_cb=None, sleep=None, frame_rate=None):
    ''' Renders videos as frame-by-frame images in notebook; frame_cb
        is optional callback to do additional work with the frame and the index.

        vid can be a string or path, in which case it is read from the file
        vid can also be a numpy array, in which case it is looped over
    '''
    if isinstance(vid, (str, Path)):
        vid = cv2.VideoCapture(vid)

    def _get_next_frame(vid, ix):
        if isinstance(vid, cv2.VideoCapture):
            ret, frame = vid.read()

            if not ret:
                vid.release()
                return ret, frame

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return ret, frame
        else:
            if ix >= vid.shape[0]:
                return False, None
            else:
                return True, vid[ix, ...]

    i = 0
    try:
        while True:
            ret, frame = _get_next_frame(vid, i)

            if not ret:
                break

            if frame_cb:
                frame = frame_cb(frame, i)

            display_array(frame, fmt='jpeg')
            clear_output(wait=True)

            if frame_rate:
                time.sleep(1 / frame_rate)

            if sleep:
                time.sleep(sleep)

            i += 1

    except KeyboardInterrupt:
        if isinstance(vid, cv2.VideoCapture):
            vid.release()
