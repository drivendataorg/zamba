from collections import deque
from os import remove, getenv
from pathlib import Path
from shutil import rmtree
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import logging
from tqdm import tqdm

from tensorflow.python.keras.utils import get_file

from .model import Model
from .cnnensemble.src.single_frame_cnn import generate_prediction_test
from .cnnensemble.src import second_stage
from .cnnensemble.src import config
from .cnnensemble.src import prepare_train_data
from .cnnensemble.src import single_frame_cnn

load_dotenv(find_dotenv())


class CnnEnsemble(Model):
    def __init__(self,
                 model_path,
                 profile=config.DEFAULT_PROFILE,
                 tempdir=None,
                 download_region='us',
                 download_weights=True,
                 verbose=False
                 ):
        # use the model object's defaults
        super().__init__(model_path, tempdir=tempdir)
        self.profile = profile
        self.verbose = verbose
        self.logger = logging.getLogger(f"{__file__}")

        if download_weights:
            self._download_weights_if_needed(download_region)

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
        file_names = [x for x in p.glob('**/*') if x.is_file() and not x.name.startswith(".")]
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
            l1_results[l1_model], skipped = generate_prediction_test(
                model_name=l1_model,
                weights=config.MODEL_DIR / 'output' / config.MODEL_WEIGHTS[l1_model],
                file_names=file_names,
                verbose=self.verbose,
                save_results=False
            )

            all_skipped |= set([str(f) for f in skipped])

        l2_results = second_stage.predict(l1_results, profile=self.profile)

        preds = pd.DataFrame(
            {
                'filename': [str(file_name) for file_name in file_names if str(file_name) not in all_skipped],
                **{cls: l2_results[:, i] for i, cls in enumerate(config.CLASSES)}
            },
            columns=['filename'] + config.CLASSES
        )

        return preds.set_index('filename')

    def _train_initial_l1_filter_model(self):
        for fold in config.TRAIN_FOLDS:
            self.logger.info(f"Train initial model, fold {fold}")
            single_frame_cnn.train(
                fold=fold,
                model_name=config.BLANK_FRAMES_MODEL,
                nb_epochs=config.NB_EPOCHS[config.BLANK_FRAMES_MODEL],
                use_non_blank_frames=False)

    def _find_non_non_blank_frames(self):
        model_name = config.BLANK_FRAMES_MODEL
        for fold in config.TRAIN_FOLDS:
            self.logger.info(f'find non blank frames for fold {fold}')
            # TODO: slow part, would benefit from parallel execution at least per fold
            single_frame_cnn.generate_prediction(
                   model_name=model_name,
                   weights=str(config.MODEL_DIR / f'output/{model_name}_s_fold_{fold}.h5'),
                   fold=fold)
            single_frame_cnn.find_non_blank_frames(model_name=model_name, fold=fold)

    def _train_l1_models(self):
        for model_name in config.PROFILES['full']:
            for fold in config.TRAIN_FOLDS + [0]:  # 0 fold for training on the full dataset
                model_weights_path = single_frame_cnn.model_weights_path(model_name, fold)
                if model_weights_path.exists():
                    self.logger.info(f'L1 model {model_name} for fold {fold} already trained, use existing weights '
                                     f'{model_weights_path}')
                else:
                    self.logger.info(f'train L1 model {model_name} for fold {fold}')
                    single_frame_cnn.train(
                        model_name=model_name,
                        nb_epochs=config.NB_EPOCHS[model_name],
                        fold=fold,
                        use_non_blank_frames=False
                    )

    def _predict_l1_oof(self):
        for model_name in config.PROFILES['full']:
            for fold in config.TRAIN_FOLDS:
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
        the provided X and y.

        Args:
            X: training data
            y: training labels

        Returns:

        """
        prepare_train_data.generate_folds(config)
        prepare_train_data.generate_train_images()
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
        fnames = ["input.tar.gz", "output.tar.gz", "data_fast.zip"]

        cache_dir = Path(__file__).parent if getenv("ZAMBA_CACHE_DIR") is None else getenv("ZAMBA_CACHE_DIR")
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
