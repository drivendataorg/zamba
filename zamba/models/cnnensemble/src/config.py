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

ALL_MODELS_WITH_TRAIN_FOLDS = [
    [('resnet50_avg', 1), ('resnet50', 2), ('resnet50_avg', 3), ('resnet50_avg', 4)],
    [('xception_avg', fold) for fold in [1, 2, 3, 4]],
    [('xception_avg_ch10', fold) for fold in [1, 2, 3, 4]],
    [('inception_v3', fold) for fold in [1, 2, 3, 4]],
    [('inception_v2_resnet', fold) for fold in [1, 2, 3, 4]],
    [('inception_v2_resnet_ch10', fold) for fold in [1, 2, 3, 4]],
    [('resnet152', fold) for fold in [1, 2, 3, 4]],
    [('inception_v2_resnet_extra', fold) for fold in [1, 2, 3, 4]],
]

ALL_MODELS = [
    [('resnet50_avg', 1)],
    [('xception_avg', 1)],
    [('xception_avg_ch10', 1)],
    [('inception_v3', 1)],
    [('inception_v2_resnet', 1)],
    [('inception_v2_resnet_ch10', 1)],
    [('resnet152', 1)],
    [('inception_v2_resnet_extra', 1)],
]
