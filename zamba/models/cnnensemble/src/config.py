from os import getenv
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

MODEL_DIR = Path(__file__).parent.parent if getenv("ZAMBA_CACHE_DIR") is None else Path(getenv("ZAMBA_CACHE_DIR")) / "cnnensemble"
RAW_VIDEO_DIR = MODEL_DIR / "input" / "raw"
TRAIN_IMG_DIR = MODEL_DIR / "data_fast" / "pri_matrix" / "train_img"
TEST_VIDEO_DIR = MODEL_DIR / "input" / "raw_test"
UNUSED_VIDEO_DIR = MODEL_DIR / "input" / "raw_unused"
UNUSED_IMG_DIR = MODEL_DIR / "input" / "raw_unused_img"
SMALL_VIDEO_DIR = MODEL_DIR / "input" / "small"
TRAINING_SET_LABELS = MODEL_DIR / "input" / "Pri-matrix_Factorization_-_Training_Set_Labels.csv"
SUBMISSION_FORMAT = MODEL_DIR / "input" / "Pri-matrix_Factorization_-_Submission_Format.csv"
FOLDS_PATH = MODEL_DIR / "input" / "folds.csv"
#
TRAINING_GROUPS = MODEL_DIR / 'input' / 'obfuscation_map_with_api_data.csv'

CLASSES = ['bird', 'blank', 'cattle', 'chimpanzee', 'elephant', 'forest buffalo', 'gorilla', 'hippopotamus', 'human',
           'hyena', 'large ungulate', 'leopard', 'lion', 'other (non-primate)', 'other (primate)', 'pangolin',
           'porcupine', 'reptile', 'rodent', 'small antelope', 'small cat', 'wild dog', 'duiker', 'hog']

# number of cores used for multiprocessing pools, jobs etc
N_CORES = 8

# downsample bins of sorted predictions per class for frames for the second level model input
L2_SORTED_BINS_DOWNSAMPLE = 4

# for pipeline testing only, train on the small subset of training samples
TRAIN_ON_SMALL_SUBSET = True
TRAIN_SAMPLES_SUBSET = 0.025
# for subset of training samples, keep at least 100 samples
MIN_SAMPLES_TO_KEEP = 100

VIDEO_FPS = 24
PREDICT_FRAMES = [2, 8, 12, 18] + [i * VIDEO_FPS // 2 + 24 for i in range(14 * 2)]
TRAIN_FRAMES = PREDICT_FRAMES[:16]

INPUT_ROWS = 404
INPUT_COLS = 720
INPUT_CHANNELS = 3

# L2 models are trained on OOF L1 model predictions, folds are listed here
TRAIN_FOLDS = [1, 2, 3, 4]

BLANK_FRAMES_MODEL = 'resnet50'

NB_EPOCHS = {
    'nasnet_mobile': 12,
    'inception_v2_resnet': 12,
    'inception_v3': 12,
    'xception_avg': 12,
    'resnet50': 12
}

# MODEL_WEIGHTS = {
#     'inception_v2_resnet': 'inception_v2_resnet_ch10_fold_0/checkpoint-010-0.0324-0.0300.hdf5',
#     'inception_v3': 'inception_v3_fold_0/checkpoint-009-0.0345-0.0331.hdf5',
#     'nasnet_mobile': 'nasnet_mobile_fold_0/checkpoint-012-0.0329-0.0315.hdf5',
#     'xception_avg': 'xception_avg_ch10_fold_0/checkpoint-009-0.0318-0.0399.hdf5'
# }

MODEL_WEIGHTS = {
    'inception_v2_resnet': 'inception_v2_resnet_s_fold_0.h5',
    'inception_v3': 'inception_v3_s_fold_0.h5',
    'nasnet_mobile': 'nasnet_mobile_s_fold_0.h5',
    'xception_avg': 'xception_avg_s_fold_0.h5'
}

PROFILES = {
    'fast': ['nasnet_mobile'],
    'full': ['nasnet_mobile', 'inception_v3', 'xception_avg', 'inception_v2_resnet']
}

DEFAULT_PROFILE = 'full'
