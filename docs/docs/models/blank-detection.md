# Blank detection

<a id='blank-nonblank'></a>

## `blank_nonblank` model

If you only want to separate blank videos (where no animal is present) from videos that contain an animal, use the `blank_nonblank` model. Unlike the [video species classification](video-classification.md) models, it does not predict a species; it only outputs the probability that a video is `blank`.

### Architecture

The `blank_nonblank` model uses the same [architecture](video-classification.md#time-distributed) as the `time_distributed` model, but there is only one output class as this is a binary classification problem.

### Default configuration

The full default configuration is available on [Github](https://github.com/drivendataorg/zamba/blob/master/zamba/models/official_models/blank_nonblank/config.yaml).

The `blank_nonblank` model uses the same [default configuration](video-classification.md#time-distributed-config) as the `time_distributed` model. For the frame selection, an efficient object detection model called [MegadetectorLite](megadetector.md) is run on all frames to determine which are the most likely to contain an animal. Then the classification model is run on only the 16 frames with the highest predicted probability of detection.

### Training data

The `blank_nonblank` model was trained on all the data used for the [`time_distributed`](video-classification.md#time-distributed-training-data) and [`european`](video-classification.md#european-training-data) models.

!!! tip "Blank detection with the species models"
    The [`time_distributed`](video-classification.md#time-distributed) and [`slowfast`](video-classification.md#slowfast) species models also include a `blank` class, so they can identify blank videos while classifying species. See the [`time_distributed` performance](video-classification.md#time-distributed-performance) section for an analysis of the blank/non-blank tradeoff.
