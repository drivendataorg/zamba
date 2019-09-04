import zamba

MODEL_DIR = zamba.config.cache_dir / "cnnensemble"
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
N_CORES = 16

# downsample bins of sorted predictions per class for frames for the second level model input
L2_SORTED_BINS_DOWNSAMPLE = 4

# for pipeline testing only, train on the small subset of training samples
TRAIN_ON_SMALL_SUBSET = False
# keep only 2.5% of samples for quick pipeline testing
TRAIN_SAMPLES_SUBSET = 0.025
# for subset of training samples, keep at least 100 samples
MIN_SAMPLES_TO_KEEP = 100


# expected FPS of video clips, used only for frame number selection below
VIDEO_FPS = 24
# list of video frames used for prediction.
# overall use 2 frames per second, with two exceptions:
# more samples during the first second as animal is more likely to be visible after the camera triggered
# exclude the first few frames as it's often blank or has incorrect exposure for many cameras
PREDICT_FRAMES = [2, 8, 12, 18] + [i * VIDEO_FPS // 2 + 24 for i in range(14 * 2)]
# for training use only the first 16 selected frames, approx 8 seconds of video
TRAIN_FRAMES = PREDICT_FRAMES[:16]

# the most common video resolution, all frames are resized to this resolution
INPUT_ROWS = 404
INPUT_COLS = 720
INPUT_CHANNELS = 3

# L2 models are trained on OOF L1 model predictions, folds are listed here
TRAIN_FOLDS = [1, 2, 3, 4]

# initial model trained on all
BLANK_FRAMES_MODEL = 'resnet50'

# number of epochs to train each L1 model
NB_EPOCHS = {
    'nasnet_mobile': 12,
    'inception_v2_resnet': 10,
    'inception_v3': 12,
    'xception_avg': 10,
    'resnet50': 9
}

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
