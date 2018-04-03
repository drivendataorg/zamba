import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import config

import shutil
import os


def generate_folds():
    training_set_labels_ds_full = pd.read_csv(config.TRAINING_SET_LABELS)

    CLASSES = ['bird', 'blank', 'cattle', 'chimpanzee', 'elephant', 'forest buffalo', 'gorilla', 'hippopotamus', 'human',
               'hyena', 'large ungulate', 'leopard', 'lion', 'other (non-primate)', 'other (primate)', 'pangolin',
               'porcupine', 'reptile', 'rodent', 'small antelope', 'small cat', 'wild dog', 'duiker', 'hog']

    training_set_labels_ds_full['fold'] = 0

    data = training_set_labels_ds_full.as_matrix(columns=CLASSES + ['filename', 'fold'])

    cls = np.argmax(data[:, :-2], axis=1)

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    split_idx = list(skf.split(data, cls))

    for fold, fold_split in enumerate(split_idx):
        items = fold_split[1]
        data[items, -1] = fold + 1
        print(fold)
        print(np.sum(data[items, :-2], axis=0))

    training_set_labels_ds_full['fold'] = data[:, -1]
    training_set_labels_ds_full.to_csv(Path(__file__).parent.parent / 'input/folds.csv', columns=['filename', 'fold'], index=False)


def move_non_train():
    training_set_labels_ds_full = pd.read_csv(config.TRAINING_SET_LABELS)
    train_files = set(training_set_labels_ds_full.filename)

    src_dir = config.MODEL_DIR / 'input/raw/'
    dst_dir = config.MODEL_DIR / 'input/raw_test/'
    skip_files = 0
    for fn in os.listdir(Path(__file__).parent.parent / 'input/raw/'):
        if fn not in train_files:
            skip_files += 1
            shutil.move(os.path.join(src_dir, fn), os.path.join(dst_dir, fn))
            if skip_files + 1 % 100 == 0:
                print(skip_files)
    print(skip_files)


def move_unused():
    test_set = pd.read_csv(config.SUBMISSION_FORMAT)
    test_files = set(test_set.filename)

    src_dir = config.MODEL_DIR / 'input/raw_test/'
    dst_dir = config.MODEL_DIR / 'input/raw_unused/'
    skip_files = 0
    for fn in os.listdir(Path(__file__).parent.parent / 'input/raw_test/'):
        if fn not in test_files:
            skip_files += 1
            shutil.move(os.path.join(src_dir, fn), os.path.join(dst_dir, fn))
            if skip_files + 1 % 100 == 0:
                print(skip_files)
    print(skip_files)

move_unused()
