# User Tutorial: Training a Model on Labaled Videos

This section walks through how to train a model using `zamba`. If you are new to `zamba` and just want to classify some videos as soon as possible, see the [Quickstart](quickstart.md) guide.

This tutorial goes over the steps for using `zamba` if:

* You already have `zamba` installed (for details see the [Installation](install.md) page)
* You have labeled videos that you want to use to train or finetune a model

`zamba` can run two types of model training:

* Fine-tuning a model with labels that are a subset of the possible [zamba labels](models.md#species-classes)
* Retraining a model to predict an entirely new set of labels

The process is the same for both cases.

## Basic usage: command line interface

Say that we want to finetune the `time_distributed` model based on the videos in `example_vids` and the labels in `example_labels.csv`. 

Minimum example for training in the command line:

```console
$ zamba train --data-dir example_vids/ --labels example_labels.csv
```

### Required arguments

To run `zamba train` in the command line, you must specify both `--data-directory` and `--labels`.

* **`--data-dir PATH`:** Path to the folder containing your labeled videos.
* **`--labels PATH`:** Path to a CSV containing the video labels to use as ground truth during training. There must be columns for both filepath and label. 

## Basic usage: Python package

Say that we want to finetune the `time_distributed` model based on the videos in `example_vids` and the labels in `example_labels.csv`. 

Minimum example for training using the Python package:

```python
from zamba.models.model_manager import train_model
from zamba.models.config import TrainConfig
from zamba_algorithms.data.video import VideoLoaderConfig

train_config = TrainConfig(data_directory='example_vids/',
                           labels='example_labels.csv')
video_loader_config = VideoLoaderConfig(video_height=224, 
                                        video_width=224, 
                                        total_frames=16)

train_model(train_config=train_config, 
            video_loader_config=video_loader_config)
```

To specify various parameters when running `train_model`, the first step is to instantiate [`TrainConfig`](configurations.md#training-arguments) and [`VideoLoaderConfig`](configurations.md#video-loading-arguments) with any specifications for model training and video loading respectively. The only two arguments that can be specified in `train_model` are `train_config` and `video_loader_config`.

### Required arguments

To run `train_model` in Python, you must specify both `data_directory` and `labels` when `TrainConfig` is instantiated.

* **`data_directory (DirectoryPath)`:** Path to the folder containing your videos.

* **`labels (FilePath or pd.DataFrame)`:** Either the path to a CSV file with labels for training, or a dataframe of the training labels. There must be columns for `filename` and `label`.

In the command line, video loading configurations are loaded by default based on the model being used. This is not the case in Python. There are additional requirements for `VideoLoaderConfig` based on the model you are using.

* **`video_height (int)`, `video_width (int)`:** Dimensions for resizing videos as they are loaded. 
    - `time_distributed` or `european`: The suggested dimensions are 224x224, but any integers are acceptable
    - `slowfast`: Both must be greater than or equal to 200
* **`total_frames (int)`:** The number of frames to select from each video and use during training. 
    * `time_distributed` or `european`: Must be 16
    * `slowfast`: Must be 32

The full recommended `VideoLoaderConfig` for the `time_distributed` or `european` model is:
```python
from zamba_algorithms.data.video import VideoLoaderConfig
from zamba.models.megadetector_lite_yolox import MegadetectorLiteYoloXConfig

megadetector_config = MegadetectorLiteYoloXConfig(confidence=0.25,
                                                  fill_mode="score_sorted",
                                                  n_frames=16)
video_loader_config = VideoLoaderConfig(video_height=224,
                                        video_width=224,
                                        crop_bottom_pixels=50,
                                        ensure_total_frames=True,
                                        megadetector_list_config=megadetector_config,
                                        total_frames=16)
```

You can see the full default configuration for each model in `models/config`<!-- TODO: add link to source and update if needed><!-->. For detailed explanations of all possible configuration arguments, see [All Optional Arguments](configurations.md).

## Default behavior

By default, the model will be saved to a folder in the current working directory called `zamba_<model_name>`. For example, a model finetuned from the provided `time_distributed` model (the default) will be saved in `zamba_time_distributed`. 

```console
$ zamba train --data-dir example_vids/ --labels example_labels.csv
$ ls zamba_time_distributed
configuration.yaml 
hparams.yaml
events.out.tfevents.1632250686.ip-172-31-15-179.14229.0
```

`zamba_time_distributed` contains three files:

* `configuration.yaml`: The full model configuration used to generate the given model, including `video_loader_config` and `train_config`. To continue training using the same configuration, or to train another model using the same configuration, you can pass in `configurations.yaml` (see [Specifying Model Configurations with a YAML File](yaml-config.md)).
* `hparams.yaml`: Model hyperparameters. For example, the YAML file below tells us that the model was trained with a learning rate (`lr`) of 0.001:
    ```yaml
    $ cat zamba_time_distributed/hparams.yaml

    lr: 0.001
    model_class: TimeDistributedEfficientNetMultiLayerHead
    num_frames: 16
    scheduler: MultiStepLR
    scheduler_params:
    gamma: 0.5
    milestones:
    - 3
    verbose: true
    species:
    - species_blank
    - species_chimpanzee_bonobo
    - species_elephant
    - species_leopard
    ```
* `events.out.tfevents.1632250686.ip-172-31-15-179.14229.0`: Model checkpoint. The model checkpoint also includes both the model configuration in `configuration.yaml` and the model hyperparameters in `hparams.yaml`. You can continue training from this checkpoint by passing it to `zamba train` with the `--checkpoint` flag:
    ```console
    $ zamba train --checkpoint zamba_time_distributed/events.out.tfevents.1632250686.ip-172-31-15-179.14229.0 --data-dir example_vids/ --labels example_labels.csv
    ```

## Step-by-step tutorial

### 1. Specify the path to your videos 

Save all of your videos in one folder.

* Your videos should all be saved in formats that are suppored by FFMPEG, [which are listed here](https://www.ffmpeg.org/general.html#Supported-File-Formats_002c-Codecs-or-Features).
* Your video folder must contain only valid video files, since zamba will try to load all of the files in the directory.
* Your videos must all be in the top level of the video folder - `zamba` does not extract videos from nested directories.

Add the path to your video folder with `--data-dir`. For example, if your videos are in a folder called `example_vids`, add `--data-dir example_vids/` to your command.

In Python, the data directory is specified when `TrainConfig` is instantiated:

```python
from zamba.models.config import TrainConfig

# note this will not run yet because labels are not specified
train_config = TrainConfig(data_directory='example_vids/')
```

### 2. Specify your labels

Your labels should be saved in a `.csv` file with columns for filepath and label. For example:

```console
$ cat example_labels.csv
filepath,label
example_vids/eleph.MP4,elephant
example_vids/leopard.MP4,leopard
example_vids/blank.MP4,blank
example_vids/chimp.MP4,chimpanzee_bonobo
```

Add the path to your labels with `--labels`.  For example, if your videos are in a folder called `example_vids` and your labels are saved in `example_labels.csv`:

```console
$ zamba train --data-dir example_vids/ --labels example_labels.csv
```

In Python, the labels are passed in when `TrainConfig` is instantiated. The Python package allows you to pass in labels as either a file path or a pandas dataframe:
```python
labels_dataframe = pd.read_csv('example_labels.csv').set_index('filepath')
train_config = TrainConfig(data_directory='example_vids/', 
                           labels=labels_dataframe)
```

#### Labels `zamba` has seen before

Your labels may be included in the list of [`zamba` class labels](models.md#species-classes) that the provided models are trained to predict. If so, the relevant model that ships with `zamba` will essentially be used as a checkpoint, and model training will resume from that checkpoint.

#### Completely new labels

You can also train a model to predict completely new labels - the world is your oyster! (We'd love to see a model trained to predict oysters.) If this is the case, the model architecture will replacing the final [neural network](https://www.youtube.com/watch?v=aircAruvnKk&t=995s) layer with a new head that predicts *your* labels instead of those that ship with `zamba`. [Backpropogation](https://www.youtube.com/watch?v=Ilg3gGewQ5U) will continue from that point with the new head.

### 3. Choose a model for training

If your videos contain species common to central or west Africa, use the [`time_distributed` model](models.md#time-distributed). If they contain species common to western Europe, use the [`european` model](models.md#european). We do not recommend using the [`slowfast` model](models.md#slowfast) for training because it is much more computationally intensive and slower to run.

dd the model name to your command with `--model`. The `time_distributed` model will be used if no model is specified. For example, if you want to continue training the `european` model based on the videos in `example_euro_vids` and the labels in `example_euro_labels.csv`:

```console
$ zamba train --data-dir example_euro_vids/ --labels example_euro_labels.csv --model european
```

In Python, model is specified when `TrainConfig` is instantiated:

```python
train_config = TrainConfig(data_directory='example_euro_vids/',
                           labels='example_euro_labels.csv',
                           model_name='european')
```

### 4. Specify any additional parameters

And there's so much more! You can also do things like specify your region for faster model download (`--weight-download-region`), start training from a saved model checkpoint (`--checkpoint`), or run only one batch for faster debugging (`--dry-run`). We'll go through a few common options to consider. If you using the command line interface, all of the parameters in this section must be passed as part of a [YAML configuration file](yaml-config.md) rather than directly to the command line.

#### Video size

`zamba` can resize all videos before using them to train a model. Higher resolution videos will lead to more detailed accuracy in prediction, but will use more memory and take longer to train from.

The default for all pretrained models is 224x224 pixels. Say that you have a large number of videos, and you are more considered with detecting blank v. non-blank videos than with identifying different species. You could train a model more quickly and with less detail be resizing images to 50x50 pixels. Your [YAML configuration file](yaml-config.md) would include:
```yaml
video_loader_config:
  video_height: 50
  video_width: 50
```

In Python, video resizing can be specified when `VideoLoaderConfig` is instantiated:

```python
video_loader_config = VideoLoaderConfig(video_height=50, video_width=50)
```

#### Frame selection

The model only trains or generates prediction based on a subset of the frames in a video. There are a number of different ways to select frames (see the section on [Video loading arguments](configurations.md#video-loading-arguments) for details). A few possible methods:

* If animals are more likely to be seen early in the video because that is closer to when the camera trap was triggered, you may want to set `early_bias` to True. This selects 16 frames towards the beginning of the video.
* A simple option is to sample frames that are evenly distributed throughout a video. For example, to select 32 evenly distributed frames, add the following to a [YAML configuration file](yaml-config.md):
```yaml
video_loader_config:
    total_frames: 32
    evenly_sample_total_frames: True
    ensure_total_frames: True
```
In Python, these arguments can be specified when `VideoLoaderConfig` is instantiated:
```python
video_loader_config = VideoLoaderConfig(total_frames=32,
                                        evenly_sample_total_frames=True,
                                        ensure_total_frames=True)
```
* You can use a pretrained object detection model called [MegadetectorLiteYoloX](models.md#megadetectorliteyolox) to select only the frames that are mostly likely to contain an animal - this is the default method. The parameter `megadetector_lite_config` is used to specify any arguments that should be passed to the megadetector model. For example, to take the 16 frames with the highest probability of detection based on the megadetector, add the following to a [YAML configuration file](yaml-config.md):
```yaml
video_loader_config:
    megadetector_lite_config:
        n_frames: 16
        fill_mode: "score_sorted"
```

In Python, these can be specified in the `megadetector_lite_config` argument passed to `VideoLoaderConfig`:
```python
video_loader_config = VideoLoaderConfig(
    video_height=224,
    video_width=224,
    crop_bottom_pixels=50,
    ensure_total_frames=True,
    megadetector_lite_config={"confidence": 0.25, "fill_mode": "score_sorted", "n_frames": 16},
    total_frames=16,
)

train_config = TrainConfig(
    data_directory="example_vids/",
    labels="example_labels.csv",
    },
)

train_model(video_loader_config=video_loader_config, 
            train_config=train_config)
```

And that's just the tip of the iceberg! See the [All Optional Arguments](configurations.md) page for more possibilities.
