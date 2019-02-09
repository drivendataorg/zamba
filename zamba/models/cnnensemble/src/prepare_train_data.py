import copy

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import random
import os
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm

from zamba.models.cnnensemble.src import config
from zamba.models.cnnensemble.src import utils
from zamba.models.cnnensemble.src import folds_split

import logging
logger = logging.getLogger(__file__)


def generate_folds_site_aware():
    training_set_labels_ds_full = pd.read_csv(config.TRAINING_SET_LABELS)
    group_names_df = pd.read_csv(config.MODEL_DIR / 'input' / 'obfuscation_map_with_api_data.csv', low_memory=False)
    df = folds_split.prepare_group_aware_split_to_folds(training_set_labels_ds_full, group_names_df)
    _save_folds(df)


def generate_folds_random():
    training_set_labels_ds_full = pd.read_csv(config.TRAINING_SET_LABELS)
    training_set_labels_ds_full['fold'] = 0

    data = training_set_labels_ds_full.as_matrix(columns=config.CLASSES + ['filename', 'fold'])

    category = np.argmax(data[:, :-2], axis=1)

    skf = StratifiedKFold(n_splits=len(config.TRAIN_FOLDS), shuffle=True, random_state=42)

    split_idx = list(skf.split(data, category))
    for fold, fold_split in enumerate(split_idx):
        items = fold_split[1]
        data[items, -1] = fold + 1
        logger.debug(f"Fold {fold}")
        logger.debug(f"Fold classes {np.sum(data[items, :-2], axis=0)}")

    training_set_labels_ds_full['fold'] = data[:, -1]
    _save_folds(training_set_labels_ds_full)


def _save_folds(training_set_labels_ds_full):
    if config.TRAIN_ON_SMALL_SUBSET:
        # pipeline testing mode, train on the the small subset of samples
        filenames = []
        for fold in [1, 2, 3, 4]:
            cur_fold_data = training_set_labels_ds_full[training_set_labels_ds_full.fold == fold]
            for cls in config.CLASSES:
                cls_fnames = list(cur_fold_data[cur_fold_data[cls] == 1].filename)
                random.seed(42)
                random.shuffle(cls_fnames)
                samples_to_keep = max(config.MIN_SAMPLES_TO_KEEP, int(len(cls_fnames) * config.TRAIN_SAMPLES_SUBSET))
                selected = cls_fnames[:samples_to_keep]
                filenames += list(selected)

        fold_data_small = training_set_labels_ds_full[training_set_labels_ds_full.filename.isin(set(filenames))]
        fold_data_small.to_csv(config.FOLDS_PATH,
                               columns=['filename', 'fold'],
                               index=False)
    else:
        training_set_labels_ds_full.to_csv(config.FOLDS_PATH,
                                           columns=['filename', 'fold'],
                                           index=False)


def _prepare_frame_data(video_id):
    # skip already processed files
    processed_images = 0
    dest_dir = config.TRAIN_IMG_DIR / video_id[:-4]
    for i, frame in enumerate(config.PREDICT_FRAMES):
        fn = str(dest_dir / f'{i+2:04}.jpg')
        if os.path.exists(fn) and os.path.getsize(fn) > 0:
            processed_images += 1
        else:
            break
    else:
        return  # all files already processed

    frames = utils.load_video_clip_frames(
        video_fn=config.RAW_VIDEO_DIR / video_id,
        frames_numbers=config.PREDICT_FRAMES[processed_images:],
        output_size=(config.INPUT_ROWS, config.INPUT_COLS)
    )
    os.makedirs(str(dest_dir), exist_ok=True)
    for i, frame in enumerate(config.PREDICT_FRAMES[processed_images:]):
        img = Image.fromarray(np.clip(frames[i], 0, 255).astype(np.uint8))
        img.save(str(dest_dir / f'{i+processed_images+2:04}.jpg'), quality=85)


def benchmark_load_video():
    # it's around 4x faster to load converted jpeg files
    video_id = '04GBFOZS5F.mp4'
    import scipy.misc
    from PIL import Image

    with utils.timeit_context('load video files'):
        frames = utils.load_video_clip_frames(
            video_fn=config.RAW_VIDEO_DIR / video_id,
            frames_numbers=config.PREDICT_FRAMES,
            output_size=(config.INPUT_ROWS, config.INPUT_COLS)
        )

    dest_dir = '/opt/data_fast/tmp/' + video_id[:-4]
    os.makedirs(str(dest_dir), exist_ok=True)

    for i, frame in enumerate(config.PREDICT_FRAMES):
        img = Image.fromarray(np.clip(frames[i], 0, 255).astype(np.uint8))
        img.save(str(dest_dir + f'/{i+2:04}.jpg'), quality=85)

    with utils.timeit_context('load images'):
        for i, frame in enumerate(config.PREDICT_FRAMES):
            fn = str(dest_dir + f'/{i+2:04}.jpg')
            X = scipy.misc.imread(fn).astype(np.float32)


def generate_train_images():
    """
    Generate jpeg frames from video clips, used for L1 models training.

    Reads videos from config.RAW_VIDEO_DIR for each entry in folds.csv
    and extract selected frames as jpeg images, saved to config.TRAIN_IMG_DIR
    """
    fold_data = pd.read_csv(config.FOLDS_PATH)
    pool = Pool(config.N_CORES)

    for _ in tqdm(pool.imap_unordered(_prepare_frame_data, fold_data.filename),
                  total=len(fold_data),
                  desc="Decode train images"):
        pass

    pool.close()
    pool.join()
