# Guide to common optional parameters

There are a LOT of ways to customize model training or inference. Here, we take that elephant-sized list of options and condense it to a manageable monkey-sized list of common considerations. To read about all possible customizations, see [All Configuration Options](configurations.md).

Many of the options below cannot be passed directly to the command line. Instead, some must be passed as part of a YAML configuration file. For example:

```console
$ zamba train --config path_to_your_config_file.yaml
```

For using a YAML file with the Python package and other details, see the [YAML Configuration File](yaml-config.md) page.

## Downloading model weights

`zamba` needs to download the "weights" files for the neural networks that it uses to make predictions. On first run it will download ~200-500 MB of files with these weights depending which model you choose. Model weights are stored on servers in three locations, and downloading weights from the server closest to you will run the fastest. By default, weights will be downloaded from the US. To specify a different region:

=== "CLI"
    ```console
    zamba predict --data-dir example_vids/ --weight_download_region asia
    ```
=== "Python"
    In Python this can be specified in [`PredictConfig`](configurations.md#prediction-arguments) or [`TrainConfig`](configurations.md#training-arguments):
    ```python
    predict_config = PredictConfig(
        data_dir="example_vids/",
        weight_download_region='asia',
    )
    ```

The options for `weight_download_region` are `us`, `eu`, and `asia`. Once a model's weights are downloaded, `zamba` will use the local version and will not need to perform this download again.

## Video size

When `zamba` loads videos prior to either inference or training, it resizes all of the video frames before feeding them into a model. Higher resolution videos will lead to superior accuracy in prediction, but will use more memory and take longer to train and/or predict. The default video loading configuration for all pretrained models resizes images to 240x426 pixels.

Say that you have a large number of videos, and you are more concerned with detecting blank v. non-blank videos than with identifying different species. In this case, you may not need a very high resolution and iterating through all of your videos with a high resolution would take a very long time. For example, to resize all images to 150x150 pixels instead of the default 240x426:

=== "YAML file"
    ```yaml
    video_loader_config:
        model_input_height: 150
        model_input_width: 150
        total_frames: 16 # total_frames must always be specified
    ```
=== "Python"
    In Python, video resizing can be specified when `VideoLoaderConfig` is instantiated:

    ```python hl_lines="7 8 9"
    from zamba.data.video import VideoLoaderConfig
    from zamba.models.config import PredictConfig
    from zamba.models.model_manager import predict_model

    predict_config = PredictConfig(data_dir="example_vids/")

    video_loader_config = VideoLoaderConfig(
        model_input_height=150, model_input_width=150, total_frames=16
    ) # total_frames must always be specified

    predict_model(
        predict_config=predict_config, video_loader_config=video_loader_config
    )
    ```

## Frame selection

Each video is simply a series of frames, or images. Most of the videos on which `zamba` was trained had 30 frames per second. That means even just a 15-second video would contain 450 frames.

All models only use a subset of the frames in a video, because using every frame would be far too computationally intensive, and many frames are not different enough from each other to look at independently. There are a number of different ways to select frames. For a full list of options, see the section on [Video loading arguments](configurations.md#video-loading-arguments). A few common approaches are explained below.

### Early bias

Some camera traps begin recording a video when movement is detected. If this is the case, you may be more likely to see an animal towards when the video starts. Setting `early_bias` to True selects 16 frames towards the beginning of a video.

=== "YAML File"
    ```yaml
    video_loader_config:
        early_bias: True
        # ... other parameters
    ```
=== "Python"
    In Python, `early_bias` is specified when `VideoLoaderConfig` is instantiated:

    ```python
    video_loader_config = VideoLoaderConfig(early_bias=True, ...)
    ```

This method was used by the winning solution of the [Pri-matrix Factorization](https://www.drivendata.org/competitions/49/deep-learning-camera-trap-animals/) machine learning competition, which was the basis for `zamba` v1.

This is a simple heuristic approach that is computationally cheap, and works decently for camera traps that are motion-triggered and short in total duration.

### Evenly distributed frames

A simple option is to sample frames that are evenly distributed throughout a video. For example, to select 32 evenly distributed frames:

=== "YAML file"
    ```yaml
    video_loader_config:
        total_frames: 32
        evenly_sample_total_frames: True
        ensure_total_frames: True
        # ... other parameters
    ```
=== "Python"
    In Python, these arguments can be specified when `VideoLoaderConfig` is instantiated:
    ```python
    video_loader_config = VideoLoaderConfig(
        total_frames=32,
        evenly_sample_total_frames=True,
        ensure_total_frames=True,
        ...
    )
    ```

### MegadetectorLite

You can use a pretrained object detection model called [MegadetectorLite](models/species-detection.md#megadetectorlite) to select only the frames that are mostly likely to contain an animal. This is the default strategy for all pretrained models. The parameter `megadetector_lite_config` is used to specify any arguments that should be passed to the MegadetectorLite model. If `megadetector_lite_config` is None, the MegadetectorLite model will not be used.

For example, to take the 16 frames with the highest probability of detection:

=== "YAML file"
    ```yaml
    video_loader_config:
        megadetector_lite_config:
            n_frames: 16
            fill_mode: "score_sorted"
        # ... other parameters
    ```
=== "Python"
    In Python, these can be specified in the `megadetector_lite_config` argument passed to `VideoLoaderConfig`:
    ```python hl_lines="6 7 8 9 10"
    video_loader_config = VideoLoaderConfig(
        model_input_height=240,
        model_input_width=426,
        crop_bottom_pixels=50,
        ensure_total_frames=True,
        megadetector_lite_config={
            "confidence": 0.25,
            "fill_mode": "score_sorted",
            "n_frames": 16,
        },
        total_frames=16,
    )

    train_config = TrainConfig(data_dir="example_vids/", labels="example_labels.csv",)

    train_model(video_loader_config=video_loader_config, train_config=train_config)
    ```

If you are using the [MegadetectorLite](models/species-detection.md#megadetectorlite) for frame selection, there are two ways that you can specify frame resizing:

- `frame_selection_width` and `frame_selection_height` resize images *before* they are input to the frame selection method (in this case, before being fed into MegadetectorLite). If both are `None`, the **full size images will be used during frame selection**. Using full size images for selection is recommended for better detection of smaller species, but will slow down training and inference.
- `model_input_height` and `model_input_width` resize images *after* frame selection. These specify the image size that is passed to the actual model for classification.

You can specify both of the above at once, just one, or neither. The example code feeds full-size images to MegadetectorLite, and then resizes images before running them through the neural network.

To see all of the options that can be passed to the MegadetectorLite, see the [`MegadetectorLiteYoloXConfig` class](../api-reference/object-detection-megadetector_lite_yolox/#zamba.object_detection.yolox.megadetector_lite_yolox.MegadetectorLiteYoloXConfig).

## Speed up training

Training will run faster if you increase `num_workers` and/or increase `batch_size`. `num_workers` is the number of subprocesses to use for data loading. The minimum is 0, meaning the data will be loaded in the main process, and the maximum is one less than the number of CPUs in your system. By default `num_workers` is set to 3 and `batch_size` is set to 2. Increasing either of these will use more GPU memory, and could raise an error if the memory required is more than your machine has available.

You may need to try a few configuration of `num_workers`, `batch_size` and the image sizes above to settle on a configuration that works on your particular hardware.

Both can be specified in either [`predict_config`](configurations.md#prediction-arguments) or [`train_config`](configurations.md#training-arguments). For example, to increase `num_workers` to 5 and `batch_size` to 4 for inference:

=== "YAML file"
    ```yaml
    predict_config:
        data_dir: example_vids/
        num_workers: 5
        batch_size: 4
        # ... other parameters
    ```
=== "Python"
    ```python
    predict_config = PredictConfig(
        data_dir="example_vids/",
        num_workers=5,
        batch_size=4,
        # ... other parameters
    )
    ```


And that's just the tip of the iceberg! See [All Configuration Options](configurations.md) page for more possibilities.
