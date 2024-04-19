import os


VIDEO_SUFFIXES = os.environ.get("VIDEO_SUFFIXES")
if VIDEO_SUFFIXES is not None:
    VIDEO_SUFFIXES = VIDEO_SUFFIXES.split(",")
else:
    VIDEO_SUFFIXES = [".avi", ".mp4", ".asf"]

# random seed to use for splitting data without site info into train / val / holdout sets
SPLIT_SEED = os.environ.get("SPLIT_SEED", 4007)


# experimental support for predicting on images
IMAGE_SUFFIXES = [
    ext.strip() for ext in os.environ.get("IMAGE_SUFFIXES", ".jpg,.jpeg,.png,.webp").split(",")
]
PREDICT_ON_IMAGES = os.environ.get("PREDICT_ON_IMAGES", "False").lower() == "true"
