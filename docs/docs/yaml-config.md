# Using YAML configuration files

In both the command line and the Python module, options for video loading, training, and prediction can be set by passing a YAML file instead of passing arguments directly. YAML files (`.yml` or `.yaml`) are commonly used to serialize data in an easily readable way.

## Basic structure

The basic structure of a YAML model configuration depends on whether you're working with videos or images:

### Video Workflows

For video workflows, the basic YAML structure includes:

```yaml
$ cat video_config.yaml
video_loader_config:
  model_input_height: 240
  model_input_width: 426
  total_frames: 16
  # other video loading parameters

train_config:
  model_name: time_distributed
  data_dir: example_vids/
  labels: example_labels.csv
  # other training parameters, eg. batch_size

predict_config:
  model_name: time_distributed
  data_dir: example_vids/
  # other prediction parameters, eg. batch_size
```

For example, the configuration below will predict labels for the videos in `example_vids` using the `time_distributed` model. When videos are loaded, each will be resized to 240x426 pixels and 16 frames will be selected:

```yaml
video_loader_config:
  model_input_height: 240
  model_input_width: 426
  total_frames: 16

predict_config:
  model_name: time_distributed
  data_dir: example_vids/
```

### Image Workflows

For image classification workflows, the structure is:

```yaml
$ cat image_config.yaml
train_config:
  model_name: lila.science
  data_dir: example_imgs/
  labels: example_labels.csv
  image_size: 224
  # other training parameters, eg. batch_size

predict_config:
  model_name: lila.science
  data_dir: example_imgs/
  image_size: 224
  # other prediction parameters, eg. crop_images
```

For example, this configuration will predict labels for the images in `example_imgs` using the `lila.science` model:

```yaml
predict_config:
  model_name: lila.science
  data_dir: example_imgs/
  image_size: 224
  crop_images: true
```

## Required arguments

### For Video Workflows

Either `predict_config` or `train_config` is required, based on whether you will be running inference or training a model. See [All Configuration Options](configurations.md) for a full list of what can be specified under each class. To run inference, `data_dir`and/or `filepaths` must be specified. To train a model, `labels` must be specified.

In `video_loader_config`, you must specify at least `model_input_height`, `model_input_width`, and `total_frames`. While this is the minimum required, we strongly recommend being intentional in your choice of frame selection method. `total_frames` by itself will just take the first `n` frames. For a full list of frame selection methods, see the section on [Video loading arguments](configurations.md#video-loading-arguments).

* For `time_distributed`, `european`, or `blank_nonblank`, `total_frames` must be 16
* For `slowfast`, `total_frames` must be 32

### For Image Workflows

Either `predict_config` or `train_config` is required in the image pipeline. To run inference, `data_dir` and/or `filepaths` must be specified. To train a model, both `data_dir` and `labels` must be specified.

For image workflows, `image_size` is a key parameter that defines the input dimensions for the model.

## Command line interface

A YAML configuration file can be passed to the command line interface with the `--config` argument. For example, say the example configuration above is saved as `video_config.yaml`. To run prediction:

For video workflows:

```console
$ zamba predict --config video_config.yaml
$ zamba train --config video_config.yaml
```

For image workflows:

```console
$ zamba image predict --config image_config.yaml
$ zamba image train --config image_config.yaml
```

Only some of the possible parameters can be passed directly as arguments to the command line. Those not listed in the help output (e.g., `zamba predict --help` or `zamba image train --help`) must be passed in a YAML file (see the [Quickstart guide](quickstart.md#getting-help) for details).

## Python package

### Video Workflows

The main API for video workflows is the [`ModelManager` class](api-reference/models-model_manager.md#zamba.models.model_manager.ModelManager) that can be accessed with:

```python
from zamba.models.model_manager import ModelManager
```

The `ModelManager` class is used by `zamba`'s command line interface to handle preprocessing the filenames, loading the videos, training the model, performing inference, and saving predictions. Therefore any functionality available to the command line interface is accessible via the `ModelManager` class.

To instantiate the `ModelManager` based on a configuration file saved at `video_config.yaml`:
```python
>>> manager = ModelManager.from_yaml('video_config.yaml')
```

We can now run inference or model training without specifying any additional parameters, because they are already associated with our instance of the `ModelManager` class. To run inference or training:
```python
manager.predict() # inference
manager.train() # training
```

In our user tutorials, we refer to `train_model` and `predict_model` functions. The `ModelManager` class calls these same functions behind the scenes when `.predict()` or `.train()` is run.

### Image Workflows

For image classification workflows, the main API is the [`ZambaImagesManager` class](api-reference/images-manager.md#zamba.images.manager.ZambaImagesManager). A full workflow might look something like this:

```python
from zamba.images.manager import ZambaImagesManager
from zamba.images.config import ImageClassifictionPredictConfig
manager = ZambaImagesManager()
config = ImageClassifictionPredictConfig.parse_file('image_config.yaml')
manager.predict(config)
```

## Default configurations

In the command line, the default configuration for each model is passed in using a specified YAML file that ships with `zamba`. You can see the default configuration YAML files on [Github](https://github.com/drivendataorg/zamba/tree/master/zamba/models/official_models) in the `config.yaml` file within each model's folder.

For example, the default configuration for the [`time_distributed` model](models/species-detection.md#time-distributed) is:

```yaml
train_config:
  scheduler_config:
    scheduler: MultiStepLR
    scheduler_params:
      gamma: 0.5
      milestones:
      - 3
      verbose: true
  model_name: time_distributed
  backbone_finetune_config:
    backbone_initial_ratio_lr: 0.01
    multiplier: 1
    pre_train_bn: true
    train_bn: false
    unfreeze_backbone_at_epoch: 3
    verbose: true
  early_stopping_config:
    patience: 5
video_loader_config:
  model_input_height: 240
  model_input_width: 426
  crop_bottom_pixels: 50
  fps: 4
  total_frames: 16
  ensure_total_frames: true
  megadetector_lite_config:
    confidence: 0.25
    fill_mode: score_sorted
    n_frames: 16
predict_config:
  model_name: time_distributed
public_checkpoint: time_distributed_9e710aa8c92d25190a64b3b04b9122bdcb456982.ckpt
```

## Templates

To make modifying the existing defaults easier, we've set up the official models as templates in the [`templates` folder](https://github.com/drivendataorg/zamba/tree/master/templates).

Just fill in your data directory and labels, make any desired tweaks to the model config, and then kick off some [training](train-tutorial.md). Happy modeling!
