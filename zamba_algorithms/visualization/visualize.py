from ipywidgets import interactive
from PIL import Image
from IPython import display

from zamba_algorithms.data.video import load_video_frames


def video_widget(
    video_array=None,
    filepath=None,
    load_function=load_video_frames,
    frame_callback=None,
):
    """Widget to interact with a video in Jupyter notebook"""
    if video_array is None:
        video_array = load_function(filepath)

    # default height ~128
    default_scale = video_array.shape[3] // 128

    def scrub(frame=0, rescale=default_scale):
        f = video_array[frame, ::rescale, ::rescale, :]

        if frame_callback is not None:
            f = frame_callback(f)

        display.clear_output(wait=True)
        display.display(Image.fromarray(f))

    widget = interactive(scrub, frame=(0, (video_array.shape[0] - 1)), rescale=(1, 8))

    # set widget height to max height at downsample=1 to avoid jumpy output
    widget.children[-1].layout.height = f"{video_array.shape[1]}px"

    return widget
