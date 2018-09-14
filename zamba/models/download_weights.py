from collections import deque
from os import remove, getenv
from shutil import rmtree
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from tensorflow.python.keras.utils import get_file

load_dotenv(find_dotenv())


def download_weights(download_region='us'):
    """Checks for directories containing the ensemble weights, downloads them if neccessary.

    The CnnEnsemble uses many GB of pretrained weights for prediction. When using the package for the first
    time, these weights (too heavy for pypi) need to be downloaded. We use the built in keras utility get_file to
    handle the download and extraction. This function leaves some hidden files behind after extraction,
    which we manually remove.

    After download, the zamba/zamba/models/cnnensemble/ directory will have three new directories:

    1. input - contains training fold splits from the original data, and stores formatting information for output
    2. output - contains models weights for all models used in the ensemble
    3. data_fast - contains cached training image preprocessing results
    """

    download_regions = ['us', 'eu', 'asia']
    if download_region not in download_regions:
        raise ValueError(f"download_region must be one of:\t{download_regions}")
    else:
        region_urls = {'us': 'https://s3.amazonaws.com/drivendata-public-assets/',
                       'eu': 'https://s3.eu-central-1.amazonaws.com/drivendata-public-assets-eu/',
                       'asia': 'https://s3-ap-southeast-1.amazonaws.com/drivendata-public-assets-asia/'}
        region_url = region_urls[download_region]

    # file names, paths
    fnames = ["input.tar.gz", "output.tar.gz", "data_fast.zip"]

    cache_dir = Path(__file__).parent if getenv("CACHE_DIR") is None else getenv("CACHE_DIR")
    cache_subdir = Path("cnnensemble")

    paths_needed = [cache_dir / cache_subdir / "input",
                    cache_dir / cache_subdir / "output",
                    cache_dir / cache_subdir / "data_fast"]

    # download and extract if needed
    for path_needed, fname in zip(paths_needed, fnames):
        if not path_needed.exists():
            origin = region_url + fname
            get_file(fname=fname, origin=origin, cache_dir=cache_dir, cache_subdir=cache_subdir,
                     extract=True)

            # remove the compressed file
            remove(cache_dir / cache_subdir / fname)

    # remove hidden files or dirs if present
    cnnpath = cache_dir / cache_subdir
    hidden_dirs = [pth for pth in cnnpath.glob("**/*") if pth.parts[-1].startswith("._") and pth.is_dir()]
    if hidden_dirs:
        deque(map(rmtree, hidden_dirs))
    hidden_files = [pth for pth in cnnpath.glob("**/*") if pth.parts[-1].startswith("._") and pth.is_file()]
    if hidden_files:
        deque(map(remove, hidden_files))

    # now get the weights for pre-trained Keras models
    pretrained_model_weight_paths = {
        "resnet50_weights_tf_dim_ordering_tf_kernels.h5":
            f"https://github.com/fchollet/deep-learning-models/releases"
            f"/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5",
        "inception_v3_weights_tf_dim_ordering_tf_kernels.h5":
            f"https://github.com/fchollet/deep-learning-models/releases/download/v0.5"
            f"/inception_v3_weights_tf_dim_ordering_tf_kernels.h5",
        "xception_weights_tf_dim_ordering_tf_kernels.h5":
            f"https://github.com/fchollet/deep-learning-models/releases/download/v0.4"
            f"/xception_weights_tf_dim_ordering_tf_kernels.h5",
        "nasnet_mobile.h5":
            f"https://github.com/fchollet/deep-learning-models/releases/download/v0.8/NASNet-mobile.h5",
        "inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5":
            f"https://github.com/fchollet/deep-learning-models/releases/download/v0.7"
            f"/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5",
        "nasnet_mobile_no_top.h5":
            f"https://github.com/fchollet/deep-learning-models/releases/download/v0.8/NASNet-mobile-no-top.h5",
        "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5":
            f"https://github.com/fchollet/deep-learning-models/releases/download/v0.5"
            f"/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5":
            f"https://github.com/fchollet/deep-learning-models/releases/download/v0.7"
            f"/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5":
            f"https://github.com/fchollet/deep-learning-models/releases/download/v0.5"
            f"/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
    }

    for fname, origin in pretrained_model_weight_paths.items():
        get_file(fname=fname, origin=origin, cache_subdir='models')
