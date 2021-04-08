from collections import deque, OrderedDict
from os import remove, getenv
from pathlib import Path
from shutil import rmtree
from dotenv import load_dotenv, find_dotenv
import numpy as np
import pandas as pd
import logging
import tempfile
from tqdm import tqdm

from tensorflow.keras.utils import get_file

import zamba
from zamba.convert_videos import convert_videos
from zamba.models.blanknonblank import BlankNonBlank
from zamba.models.cnnensemble.src.single_frame_cnn import generate_prediction_test
from zamba.models.cnnensemble.src import second_stage
from zamba.models.cnnensemble.src import config
from zamba.models.cnnensemble.src import prepare_train_data
from zamba.models.cnnensemble.src import single_frame_cnn
from zamba.models.mega_detector import MegaDetector
from zamba.models.model import Model

load_dotenv(find_dotenv())


class CnnEnsemble(Model):
    def __init__(self,
                 model_path,
                 profile=config.DEFAULT_PROFILE,
                 tempdir=None,
                 download_region='us',
                 download_weights=True,
                 site_aware=False,
                 labels_path=None,
                 raw_video_dir=None,
                 resample=True,
                 seperate_blank_model=True,
                 ):
        # use the model object's defaults
        super().__init__(model_path=Path(model_path))
        self.profile = profile
        self.site_aware = site_aware
        self.raw_video_dir = raw_video_dir
        self.labels_path = labels_path
        self.logger = logging.getLogger(f"{__file__}")
        self.resample = resample
        self.seperate_blank_model = seperate_blank_model


        if download_weights:
            self._download_weights_if_needed(download_region)

    def preprocess_videos(self, input_paths, resample=True):
        """Preprocesses videos into a format that can be used by this model.

        Input is a directory containing videos. The directory will be recursively searched for all files that do not
        start with ".". Also has the option to resample raw videos to standard resolution and frame rate.

        Args:
            data_path (pathlib.Path): Path to a directory containing the input files
            resample (bool): If true, resample the videos to a standard resolution and frame rate

        Returns:
            OrderedDict: A dict with `key: value` as `original path: processed path` for each video.
        """
        if resample:
            self.logger.debug("Converting videos to standard resolution and frame rate.")
            output_directory = Path(
                tempfile.mkdtemp(prefix="resampled_", dir=self.tempdir)
            )
            output_paths = convert_videos(
                input_paths, output_directory, fps=15, width=448, height=252,
            )
        else:
            output_paths = input_paths

        return OrderedDict(zip(input_paths, output_paths))

    def predict(self, file_names):
        """Predict class probabilities for each input

        Args:
            file_names: input data, list of video paths

        Returns:
            pd.DataFrame: A table of class probabilities, where index is the file name and columns are the different
                classes
        """
        processed_paths = self.preprocess_videos(file_names, resample=self.resample)

        # exclude videos where output path doesn't exist (e.g. videos that can't be resampled)
        valid_videos = [v for v in processed_paths.values() if v.exists()]
        invalid_videos = [v for v in processed_paths.values() if v not in valid_videos]

        l1_results = {}
        invalid_mask = np.zeros(len(valid_videos), dtype=bool)

        prof = config.PROFILES[self.profile]
        for l1_model in tqdm(prof, desc=f'Predicting on {len(prof)} L1 models'):
            l1_model_preds = generate_prediction_test(
                model_name=l1_model,
                weights=config.MODEL_DIR / 'output' / config.MODEL_WEIGHTS[l1_model],
                file_names=valid_videos,
                save_results=False
            )

            l1_results[l1_model] = l1_model_preds

            # if video is invalid for this model, track that
            invalid_mask |= np.isnan(l1_model_preds).all(axis=(1, 2))

        # remove invalid videos from l1 results:
        l1_results = {
            name: results[~invalid_mask, ...]
            for name, results in l1_results.items()
        }

        #  get names of invalid videos
        l1_invalid_videos = np.array(valid_videos)[invalid_mask].tolist()

        # filter out new invalid videos from valid
        valid_videos = [v for v in valid_videos if v not in l1_invalid_videos]

        # add new invalid videos to invalid
        invalid_videos += [v for v in l1_invalid_videos if v not in invalid_videos]

        l2_results = second_stage.predict(l1_results, profile=self.profile)

        cnn_features = pd.DataFrame(
            {
                c: l2_results[:, i]
                for i, c in enumerate(config.CLASSES)
            },
            index=[str(file_name) for file_name in valid_videos],
            columns=config.CLASSES,
        )

        if self.seperate_blank_model:
            cnn_features["blank"] = 0

            blank = self.compute_blank_probability(valid_videos, cnn_features)

        # add nans for invalid videos
        preds = pd.concat(
            [
                cnn_features,
                pd.DataFrame(
                    np.nan,
                    index=[str(file_name) for file_name in invalid_videos],
                    columns=config.CLASSES,
                )
            ],
            axis=0,
        )

        if self.seperate_blank_model:
            preds["blank"] = blank

        preds.rename(
            index={
                str(processed_path): str(original_path)
                for original_path, processed_path in processed_paths.items()
            },
            inplace=True,
        )

        return preds

    def _train_initial_l1_filter_model(self):
        for fold in config.TRAIN_FOLDS:
            self.logger.info(f"Train initial model, fold {fold}")

            force_fold = getenv('FOLD')
            if force_fold is not None and int(force_fold) != fold:
                continue

            single_frame_cnn.train(
                fold=fold,
                model_name=config.BLANK_FRAMES_MODEL,
                nb_epochs=config.NB_EPOCHS[config.BLANK_FRAMES_MODEL],
                use_non_blank_frames=False)

    def _find_non_non_blank_frames(self):
        model_name = config.BLANK_FRAMES_MODEL
        for fold in config.TRAIN_FOLDS:
            self.logger.info(f'find non blank frames for fold {fold}')

            force_fold = getenv('FOLD')
            if force_fold is not None and int(force_fold) != fold:
                continue

            # TODO: slow part, would benefit from parallel execution at least per fold
            single_frame_cnn.generate_prediction(
                   model_name=model_name,
                   weights=str(config.MODEL_DIR / f'output/{model_name}_s_fold_{fold}.h5'),
                   fold=fold)
            single_frame_cnn.find_non_blank_frames(model_name=model_name, fold=fold)

    def compute_blank_probability(self, video_paths, cnn_features):
        """Predicts whether each video is blank

        Args:
            video_paths (list of str)
            cnn_features (np.ndarray): Array of shape (num videos, num cnn categories) containing the output of the CNN
                ensemble model

        Returns:
            pd.Series: A series containing the blank probability for each video
        """
        mega = MegaDetector()
        mega_features = mega.compute_features(video_paths)

        mega_features = pd.DataFrame(
            mega_features,
            index=[str(file_name) for file_name in video_paths],
            columns=MegaDetector.FEATURE_NAMES,
        )

        features = pd.concat([mega_features, cnn_features], axis=1)

        bnb = BlankNonBlank()
        X = bnb.prepare_features(features)
        blank = bnb.predict_proba(X)[:, 1]

        blank = pd.Series(
            blank,
            index=[str(path) for path in video_paths],
        )

        return blank

    def _train_l1_models(self):
        for model_name in config.PROFILES['full']:
            for fold in config.TRAIN_FOLDS + [0]:  # 0 fold for training on the full dataset
                force_fold = getenv('FOLD')
                if force_fold is not None and int(force_fold) != fold:
                    continue

                model_weights_path = single_frame_cnn.model_weights_path(model_name, fold)
                if model_weights_path.exists():
                    self.logger.info(f'L1 model {model_name} for fold {fold} already trained, use existing weights '
                                     f'{model_weights_path}')
                else:
                    self.logger.info(f'train L1 model {model_name} for fold {fold}')
                    print(f'train L1 model {model_name} for fold {fold}')
                    single_frame_cnn.train(
                        model_name=model_name,
                        nb_epochs=config.NB_EPOCHS[model_name],
                        fold=fold,
                        use_non_blank_frames=False,
                        checkpoints_period=2
                    )

    def _predict_l1_oof(self):
        for model_name in config.PROFILES['full']:
            for fold in config.TRAIN_FOLDS:
                force_fold = getenv('FOLD')
                if force_fold is not None and int(force_fold) != fold:
                    continue

                model_weights_path = single_frame_cnn.model_weights_path(model_name, fold)
                self.logger.info(f'predict oof L1 model {model_name} for fold {fold}')
                single_frame_cnn.generate_prediction(
                    model_name=model_name,
                    fold=fold,
                    weights=str(model_weights_path)
                )
        single_frame_cnn.save_all_combined_train_results()

    def _train_l2_models(self):
        for profile in config.PROFILES.keys():
            l1_model_names = config.PROFILES[profile]

            mlp_model = second_stage.SecondLevelModelMLP(l1_model_names=l1_model_names,
                                                         combined_model_name='preset_' + profile)

            self.logger.info(f'train L2 MLP model for profile {profile}')
            mlp_model.train()

            xgboost_model = second_stage.SecondLevelModelXGBoost(l1_model_names=l1_model_names,
                                                                 combined_model_name='preset_' + profile)

            self.logger.info(f'train L2 xgboost model for profile {profile}')
            xgboost_model.train()

    def fit(self):
        """Use the same architecture, but train the weights from scratch using
        the dataset from configuration file.
        Args:
        Returns:

        """
        if self.site_aware:
            prepare_train_data.generate_folds_site_aware(self.labels_path)
        else:
            prepare_train_data.generate_folds_random(self.labels_path)

        prepare_train_data.generate_train_images(self.raw_video_dir)

        self._train_initial_l1_filter_model()
        self._find_non_non_blank_frames()
        self._train_l1_models()
        self._predict_l1_oof()
        self._train_l2_models()

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
        fnames = ["input.tar.gz", "zamba.zip", "data_fast.zip"]

        cache_dir = zamba.config.cache_dir
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
