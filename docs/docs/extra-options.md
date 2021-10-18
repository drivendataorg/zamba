# Guide to Common Optional Parameters

There are a LOT of ways to customize model training or inference. Here, we take that elephant-sized list of options and condense it to a manageable monkey-sized list of common considerations. To read about all possible customizations, see the [All Optional Arguments](configurations.md) page.

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
        data_directory="example_vids/",
        weight_download_region='asia',
    )
    ```

The options for `weight_download_region` are `us`, `eu`, and `asia`. Once a model's weights are downloaded, the tool will use the local version and will not need to perform this download again.

## Video size

When `zamba` loads videos prior to either inference or training, it resizes all of the video frames before feeding them into a model. Higher resolution videos will lead to more detailed accuracy in prediction, but will use more memory and take longer to either predict on or train from. The default video loading configuration for all three pretrained models resizes images to 224x224 pixels. 

Say that you have a large number of videos, and you are more concerned with detecting blank v. non-blank videos than with identifying different species. In this case, you may not need a very high resolution and iterating through all of your videos with a high resolution would take a very long time. To resize all images to 50x50 pixels instead of the default 224x224: 

=== "YAML file"
    ```yaml
    video_loader_config:
        model_input_height: 50
        model_input_width: 50
        total_frames: 16 # total_frames must always be specified
    ```
=== "Python"
    In Python, video resizing can be specified when `VideoLoaderConfig` is instantiated:

    ```python hl_lines="6 7 8"
    from zamba.models.model_manager import predict_model
    from zamba.models.config import PredictConfig
    from zamba.data.video import VideoLoaderConfig

    predict_config = PredictConfig(data_directory="example_vids/")
    video_loader_config = VideoLoaderConfig(
        model_input_height=50, model_input_width=50, total_frames=16
    ) # total_frames must always be specified
    predict_model(
        predict_config=predict_config, video_loader_config=video_loader_config
    )
    ```

## Frame selection

Each video is simply a series of frames, or images. Most of the videos on which `zamba` was trained had 30 frames per second. That means even just a 15-second video would contain 450 frames.

The model only trains or generates prediction based on a subset of the frames in a video, because using every frame would be far too computationally intensive. There are a number of different ways to select frames. For a full list of options, see the section on [Video loading arguments](configurations.md#video-loading-arguments). A few common approaches are explained below.

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

### MegadetectorLiteYoloX

You can use a pretrained object detection model called [MegadetectorLiteYoloX](models.md#megadetectorliteyolox) to select only the frames that are mostly likely to contain an animal. This is the default strategy for all three pretrained models. The parameter `megadetector_lite_config` is used to specify any arguments that should be passed to the megadetector model. If `megadetector_lite_config` is None, the MegadetectorLiteYoloX model will not be used. 

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
        model_input_height=224,
        model_input_width=224,
        crop_bottom_pixels=50,
        ensure_total_frames=True,
        megadetector_lite_config={
            "confidence": 0.25,
            "fill_mode": "score_sorted",
            "n_frames": 16,
        },
        total_frames=16,
    )

    train_config = TrainConfig(data_directory="example_vids/", labels="example_labels.csv",)

    train_model(video_loader_config=video_loader_config, train_config=train_config)
    ```

Using `model_input_height` and `model_input_width` resizes images *after* any frame selection is done. If you are using the MegaDetector, the frames that are input into MegadetectorLiteYoloX will still be full size. Using `frame_selection_height` and `frame_selection_width` resizes images *before* they are input to MegadetectorLiteYoloX. Inputting full size images is recommended, especially if your species of interest are on the smaller side, but resizing before using MegadetectorLiteYoloX will speed up training. The above feeds full-size images to MegadetectorLiteYoloX, and then resizes images before running them through the neural network.

To see all of the options that can be passed to `MegadetectorLiteYoloX`, see the `MegadetectorLiteYoloXConfig` class. <!-- TODO: add link to github code><!-->

## Speed up training

Training will run faster if you increase `num_workers` or increase `batch_size`. `num_workers` is the number of subprocesses to use for data loading. The minimum is 0, meaning the data will be loaded in the main process, and the maximum is one less than the number of CPUs in your system. By default `num_workers` is set to 3 and `batch_size` is set to 8. Increasing either of these will use more GPU memory, and could raise an error if the memory required is more than your machine has available.

Both can be specified in either [`predict_config`](configurations.md#prediction-arguments) or [`train_config`](configurations.md#training-arguments). For example, to increase `num_workers` to 5 and `batch_size` to 10 for inference:

=== "YAML file"
    ```yaml
    predict_config:
        data_directory: example_vids/
        num_workers: 5
        batch_size: 10
        # ... other parameters
    ```
=== "Python"
    ```python
    predict_config = PredictConfig(
        data_directory="example_vids/",
        num_workers=5,
        batch_size=10,
        # ... other parameters
    )
    ```


And that's just the tip of the iceberg! See the [All Optional Arguments](configurations.md) page for more possibilities.
