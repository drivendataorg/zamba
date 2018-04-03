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

ALL_MODELS = [
    [('resnet50_avg', 1), ('resnet50', 2), ('resnet50_avg', 3), ('resnet50_avg', 4)],
    [('xception_avg', fold) for fold in [1, 2, 3, 4]],
    [('xception_avg_ch10', fold) for fold in [1, 2, 3, 4]],
    [('inception_v3', fold) for fold in [1, 2, 3, 4]],
    [('inception_v2_resnet', fold) for fold in [1, 2, 3, 4]],
    [('inception_v2_resnet_ch10', fold) for fold in [1, 2, 3, 4]],
    [('resnet152', fold) for fold in [1, 2, 3, 4]],
    [('inception_v2_resnet_extra', fold) for fold in [1, 2, 3, 4]],
]

MODEL_WEIGHTS = {
    'resnet50_avg': 4,
    'resnet50': 4,
    'xception_avg': 6,
    'xception_avg_ch10': 8,
    'inception_v3': 8,
    'inception_v2_resnet': 6,
    'inception_v2_resnet_ch10': 10,
    'resnet152': 4,
    'inception_v2_resnet_extra': 20
}
