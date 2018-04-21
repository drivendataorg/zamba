from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent
RAW_VIDEO_DIR = MODEL_DIR / "input" / "raw"
TRAIN_IMG_DIR = MODEL_DIR / "data_fast" / "pri_matrix" / "train_img"
TEST_VIDEO_DIR = MODEL_DIR / "input" / "raw_test"
UNUSED_VIDEO_DIR = MODEL_DIR / "input" / "raw_unused"
UNUSED_IMG_DIR = MODEL_DIR / "input" / "raw_unused_img"
SMALL_VIDEO_DIR = MODEL_DIR / "input" / "small"
TRAINING_SET_LABELS = MODEL_DIR / "input" / "Pri-matrix_Factorization_-_Training_Set_Labels.csv"
SUBMISSION_FORMAT = MODEL_DIR / "input" / "Pri-matrix_Factorization_-_Submission_Format.csv"

CLASSES = ['bird', 'blank', 'cattle', 'chimpanzee', 'elephant', 'forest buffalo', 'gorilla', 'hippopotamus', 'human',
           'hyena', 'large ungulate', 'leopard', 'lion', 'other (non-primate)', 'other (primate)', 'pangolin',
           'porcupine', 'reptile', 'rodent', 'small antelope', 'small cat', 'wild dog', 'duiker', 'hog']

# number of cores used for multiprocessing pools, jobs etc
N_CORES = 8

# downsample bins of sorted predictions per class for frames for the second level model input
L2_SORTED_BINS_DOWNSAMPLE = 4

# L2 models are trained on OOF L1 model predictions, folds are listed here
TRAIN_FOLDS = [1, 2, 3, 4]

MODEL_WEIGHTS = {
    'inception_v2_resnet': 'inception_v2_resnet_ch10_fold_0/checkpoint-010-0.0324-0.0300.hdf5',
    'inception_v3': 'inception_v3_fold_0/checkpoint-009-0.0345-0.0331.hdf5',
    'nasnet_mobile': 'nasnet_mobile_fold_0/checkpoint-012-0.0329-0.0315.hdf5',
    'xception_avg': 'xception_avg_ch10_fold_0/checkpoint-009-0.0318-0.0399.hdf5'
}

PROFILES = {
    'fast': ['nasnet_mobile'],
    'full': ['nasnet_mobile', 'inception_v3', 'xception_avg', 'inception_v2_resnet']
}

DEFAULT_PROFILE = 'full'
