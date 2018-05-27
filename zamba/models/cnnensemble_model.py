from collections import deque
from os import remove
from pathlib import Path
from shutil import rmtree
import pandas as pd

from tensorflow.python.keras.utils import get_file
from tqdm import tqdm

from .model import Model
from .cnnensemble.src.single_frame_cnn import generate_prediction_test
from .cnnensemble.src import second_stage
from .cnnensemble.src import config


class CnnEnsemble(Model):
    def __init__(self, model_path, profile=config.DEFAULT_PROFILE, tempdir=None, download_region='us', verbose=False):
        # use the model object's defaults
        super().__init__(model_path, tempdir=tempdir)
        self.profile = profile

        self._download_weights_if_needed(download_region)
        self.verbose = verbose

    def load_data(self, data_path):
        """ Loads data and returns it in a format that can be used
            by this model. This is not recursive, and loads files that
            do not start with ".".

        Args:
            data_path: A path to the input data

        Returns:
            The data.
        """
        p = Path(data_path)
        file_names = [x for x in p.iterdir() if x.is_file() and not x.name.startswith(".")]
        return file_names

    def predict(self, file_names):
        """Predict class probabilities for each input, X

        Args:
            file_names: input data, list of video clip paths

        Returns:

        """

        l1_results = {}
        prof = config.PROFILES[self.profile]
        all_skipped = set()
        for l1_model in tqdm(prof, desc=f'Predicting on {len(prof)} L1 models'):
            l1_results[l1_model], skipped = generate_prediction_test(model_name=l1_model,
                                                                     weights=(Path(__file__).parent / 'cnnensemble' / 'output' /
                                                                     'checkpoints' / config.MODEL_WEIGHTS[l1_model]),
                                                                     file_names=file_names,
                                                                     verbose=self.verbose,
                                                                     save_results=False)

            all_skipped |= set([f.name for f in skipped])

        l2_results = second_stage.predict(l1_results, profile=self.profile)

        preds = pd.DataFrame(
            {
                'filename': [file_name.name for file_name in file_names if file_name.name not in all_skipped],
                **{cls: l2_results[:, i] for i, cls in enumerate(config.CLASSES)}
            },
            columns=['filename'] + config.CLASSES
        )

        return preds.set_index('filename')

    def fit(self, X, y):
        """Use the same architecture, but train the weights from scratch using
        the provided X and y.

        Args:
            X: training data
            y: training labels

        Returns:

        """
        pass

    def finetune(self, X, y):
        """Finetune the network for a different task by keeping the
        trained weights, replacing the top layer with one that outputs
        the new classes, and re-training for a few epochs to have the
        model output the new classes instead.

        Args:
            X:
            y:

        Returns:

        """
        pass

    def save_model(self):
        """Save the model weights, checkpoints, to model_path.
        """

    def _download_weights_if_needed(self, download_region):
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

        cache_dir = Path(__file__).parent
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
