# Depth estimation

## Background

Our depth estimation model is useful for predicting the distance an animal is from the camera, which is an input into models used to estimate animal abundance.

The depth model comes from one of the winners of the [Deep Chimpact: Depth Estimation for Wildlife Conservation](https://www.drivendata.org/competitions/82/competition-wildlife-video-depth-estimation/) machine learning challenge hosted by DrivenData. The goal of this challenge was to use machine learning and advances in monocular (single-lens) depth estimation techniques to automatically estimate the distance between a camera trap and an animal contained in its video footage. The challenge drew on a unique labeled dataset from research teams from the Max Planck Institute for Evolutionary Anthropology (MPI-EVA) and the Wild Chimpanzee Foundation (WCF).

The species in the training dataset included bushbucks, chimpanzees, duikers, elephants, leopards, and monkeys. Videos were from Taï National Park in Côte d'Ivoire and Moyen-Bafing National Park in the Republic of Guinea.

The Zamba package supports running the depth estimation model on videos. Under the hood, it does the following:

- Resamples the video to one frame per second
- Runs the [MegadetectorLite](../models/species-detection.md#megadetectorlite) model on each frame to find frames with animals in them
- Estimates depth for each detected animal in the frame
- Outputs a csv with predictions

## Output format

The output of the depth estimation model is a csv with the following columns:

- `filepath`: video name
- `time`: seconds from the start of the video
- `distance`: distance between detected animal and the camera

There will be multiple rows per timestamp if there are multiple animals detected in the frame. Due to current limitations of the algorithm, the distance for all animals in the frame will be the same. If there is no animal in the frame, the distance will be null.

For example, the first few rows of the `depth_predictions.csv` might look like this:

```
filepath,time,distance
video_1.avi,0,7.4
video_1.avi,0,7.4
video_1.avi,1,
video_1.avi,2,
video_1.avi,3,
```

## Installation

The depth estimation is included by default. If you've already [installed zamba](/docs/install/), there's nothing more you need to do.

## Running depth estimation

Here's how to run the depth estimation model.

=== "CLI"
    ```bash
    # output a csv with depth predictions for each frame in the videos in PATH_TO_VIDEOS
    zamba depth --data-dir PATH_TO_VIDEOS
    ```
=== "Python"
    ```python
    from zamba.models.depth_estimation import DepthEstimationConfig
    depth_conf = DepthEstimationConfig(data_dir="PATH_TO_VIDEOS")
    depth_conf.run_model()
    ```

### Debugging

Unlike in the species classification models, selected frames are stored in memory rather than cached to disk. If you run out of memory, try predicting on a smaller number of videos. If you hit a GPU memory error, try reducing the [number of workers](../../debugging/#reducing-num_workers) or the [batch size](../../debugging/#reducing-the-batch-size).

## Getting help

To see all of the available options, run `zamba depth --help`.

```console
$ zamba depth --help

 Usage: zamba depth [OPTIONS]

 Estimate animal distance at each second in the video.

╭─ Options ─────────────────────────────────────────────────────────────────────────────────╮
│ --filepaths                       PATH          Path to csv containing `filepath` column  │
│                                                 with videos.                              │
│                                                 [default: None]                           │
│ --data-dir                        PATH          Path to folder containing videos.         │
│                                                 [default: None]                           │
│ --save-to                         PATH          An optional directory or csv path for     │
│                                                 saving the output. Defaults to            │
│                                                 `depth_predictions.csv` in the working    │
│                                                 directory.                                │
│                                                 [default: None]                           │
│ --overwrite               -o                    Overwrite output csv if it exists.        │
│ --batch-size                      INTEGER       Batch size to use for inference.          │
│                                                 [default: None]                           │
│ --num-workers                     INTEGER       Number of subprocesses to use for data    │
│                                                 loading.                                  │
│                                                 [default: None]                           │
│ --gpus                            INTEGER       Number of GPUs to use for inference. If   │
│                                                 not specifiied, will use all GPUs found   │
│                                                 on machine.                               │
│                                                 [default: None]                           │
│ --model-cache-dir                 PATH          Path to directory for downloading model   │
│                                                 weights. Alternatively, specify with      │
│                                                 environment variable `MODEL_CACHE_DIR`.   │
│                                                 If not specified, user's cache directory  │
│                                                 is used.                                  │
│                                                 [default: None]                           │
│ --weight-download-region          [us|eu|asia]  Server region for downloading weights.    │
│                                                 [default: None]                           │
│ --yes                     -y                    Skip confirmation of configuration and    │
│                                                 proceed right to prediction.              │
│ --help                                          Show this message and exit.               │
╰───────────────────────────────────────────────────────────────────────────────────────────╯
```