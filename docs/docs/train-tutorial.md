# User tutorial: Training a model on labeled videos

This section walks through how to train a model using `zamba`. If you are new to `zamba` and just want to classify some videos as soon as possible, see the [Quickstart](quickstart.md) guide.

This tutorial goes over the steps for using `zamba` if:

* You already have `zamba` installed (for details see the [Installation](install.md) page)
* You have labeled videos that you want to use to train or finetune a model

`zamba` can run two types of model training:

* Finetuning a model with labels that are a subset of the possible [zamba labels](models/species-detection.md#species-classes)
* Finetuning a model to predict an entirely new set of labels

The process is the same for both cases.

## Basic usage: command line interface

By default, the [`time_distributed`](models/species-detection.md#time-distributed) species classification model is used. Say that we want to finetune that model based on the videos in `example_vids` and the labels in `example_labels.csv`.

```console
$ cat example_labels.csv

filepath,label
blank.MP4,blank
chimp.MP4,chimpanzee_bonobo
eleph.MP4,elephant
leopard.MP4,leopard
```

Training at the command line would look like:

```console
$ zamba train --data-dir example_vids/ --labels example_labels.csv
```

### Required arguments

To run `zamba train` in the command line, you must specify `labels`.

* **`--labels PATH`:** Path to a CSV containing the video labels to use as ground truth during training. There must be columns for both `filepath` and `label`. Optionally, there can also be columns for `split` (which can have one of the three values for each row: `train`, `val`, or `holdout`) or `site` (which can contain any string identifying the location of the camera, used to allocate videos to splits if not already specified).

If the video filepaths in the labels csv are not absolute, be sure to provide the `data-dir` to which the filepaths are relative.

* **`--data-dir PATH`:** Path to the folder containing your labeled videos.


## Basic usage: Python package

To do the same thing as above using the library code, this would look like:

```python
from zamba.models.model_manager import train_model
from zamba.models.config import TrainConfig

train_config = TrainConfig(
    data_dir="example_vids/", labels="example_labels.csv"
)
train_model(train_config=train_config)
```

The only two arguments that can be passed to `train_model` are `train_config` and (optionally) `video_loader_config`. The first step is to instantiate [`TrainConfig`](configurations.md#training-arguments). Optionally, you can also specify video loading arguments by instantiating and passing in [`VideoLoaderConfig`](configurations.md#video-loading-arguments).

You'll want to go over the documentation to familiarize yourself with the options in both of these configurations since what you choose can have a large impact on the results of your model. We've tried to include in the documentation sane defaults and recommendations for how to set these parameters. For detailed explanations of all possible configuration arguments, see [All Configuration Options](configurations.md).

## Model output classes

The classes your trained model will predict are determined by which model you choose and whether the species in your labels are a subset of that model's [default labels](models/species-detection.md#species-classes). This table outlines the default behavior for a set of common scenarios.

| Classes in labels csv | Model | What we infer | Classes trained model predicts |
| --- | --- | --- | --- |
| cat, blank | `blank_nonblank` | binary model where one is "blank" |  blank |
| zebra, grizzly, blank | `time_distributed` | multiclass but not a subset of the zamba labels  | zebra, grizzly, blank |
| elephant, antelope_duiker, blank | `time_distributed` | multiclass and a subset of the zamba labels  | all African forest zamba species |

## Step-by-step tutorial

### 1. Specify the path to your videos

Save all of your videos in a folder.

* They can be in nested directories within the folder.
* Your videos should all be saved in formats that are suppored by FFmpeg, [which are listed here](https://www.ffmpeg.org/general.html#Supported-File-Formats_002c-Codecs-or-Features). Any videos that fail a set of FFmpeg checks will be skipped during inference or training. By default, `zamba` will look for files with the following suffixes: `.avi`, `.mp4`, `.asf`. To use other video suffixes that are supported by FFmpeg, set your `VIDEO_SUFFIXES` environment variable.

Add the path to your video folder with `--data-dir`. For example, if your videos are in a folder called `example_vids`, add `--data-dir example_vids/` to your command.

=== "CLI"
    ```console
    $ zamba train --data-dir example_vids
    ```

=== "Python"
    ```python
    from zamba.models.config import TrainConfig
    from zamba.models.model_manager import train_model

    train_config = TrainConfig(data_dir='example_vids/')
    train_model(train_config=train_config)
    ```
Note that the above will not run yet because labels are not specified.

The more training data you have, the better the resulting model will be. We recommend having a minimum of 100 videos **per species**. Having an imbalanced dataset - for example, where most of the videos are blank - is okay as long as there are enough examples of each individual species.

### 2. Specify your labels

Your labels should be saved in a `.csv` file with columns for filepath and label. For example:

```console
$ cat example_labels.csv
filepath,label
eleph.MP4,elephant
leopard.MP4,leopard
blank.MP4,blank
chimp.MP4,chimpanzee_bonobo
```

Add the path to your labels with `--labels`.  For example, if your videos are in a folder called `example_vids` and your labels are saved in `example_labels.csv`:

=== "CLI"
    ```console
    $ zamba train --data-dir example_vids/ --labels example_labels.csv
    ```
=== "Python"
    In Python, the labels are passed in when `TrainConfig` is instantiated. The Python package allows you to pass in labels as either a file path or a pandas dataframe:
    ```python
    labels_dataframe = pd.read_csv('example_labels.csv', index_col='filepath')
    train_config = TrainConfig(
        data_dir='example_vids/', labels=labels_dataframe
    )
    train_model(train_config=train_config)
    ```

#### Labels `zamba` has seen before

Your labels may be included in the list of [`zamba` class labels](models/species-detection.md#species-classes) that the provided models are trained to predict. If so, the relevant model that ships with `zamba` will essentially be used as a checkpoint, and model training will resume from that checkpoint.

By default, the model you train will continue to output all of the Zamba class labels, not just the ones in your dataset. For different behavior, see [`use_default_model_labels`](configurations.md#use_default_model_labels-bool-optional).

#### Completely new labels

You can also train a model to predict completely new labels - the world is your oyster! (We'd love to see a model trained to predict oysters.) If this is the case, the model architecture will replace the final [neural network](https://www.youtube.com/watch?v=aircAruvnKk&t=995s) layer with a new head that predicts *your* labels instead of those that ship with `zamba`.

You can then make your model available to others by adding it to the [Model Zoo on our wiki](https://github.com/drivendataorg/zamba/wiki).

### 3. Choose a model for training

Any of the models that ship with `zamba` can be trained. If you're training on entirely new species or new ecologies, we recommend starting with the [`time_distributed` model](models/species-detection.md#time-distributed) as this model is less computationally intensive than the [`slowfast` model](models/species-detection.md#slowfast).

However, if you're tuning a model to a subset of species (e.g. a `european_beaver` or `blank` model), use the model that was trained on data that is most similar to your new data.

Add the model name to your command with `--model`. The `time_distributed` model will be used if no model is specified. For example, if you want to continue training the `european` model based on the videos in `example_euro_vids` and the labels in `example_euro_labels.csv`:

=== "CLI"
    ```console
    $ zamba train --data-dir example_euro_vids/ --labels example_euro_labels.csv --model european
    ```
=== "Python"
    ```python
    train_config = TrainConfig(
        data_dir="example_euro_vids/",
        labels="example_euro_labels.csv",
        model_name="european",
    )
    train_model(train_config=train_config)
    ```

### 4. Specify any additional parameters

And there's so much more! You can also do things like specify your region for faster model download (`--weight-download-region`), start training from a saved model checkpoint (`--checkpoint`), or specify a different path where your model should be saved (`--save-dir`). To read about a few common considerations, see the [Guide to Common Optional Parameters](extra-options.md) page.

### 5. Test your configuration with a dry run

Before kicking off the full model training, we recommend testing your code with a "dry run". This will run one training and validation batch for one epoch to quickly detect any bugs. See the [Debugging](debugging.md) page for details.

## Files that get written out during training

 You can specify where the outputs should be saved with `--save-dir`. If no save directory is specified, `zamba` will write out incremental `version_n` folders to your current working directory. For example, a model finetuned from the provided `time_distributed` model (the default) will be saved in `version_0`.

`version_0` contains:

* `train_configuration.yaml`: The full model configuration used to generate the given model, including `video_loader_config` and `train_config`. To continue training using the same configuration, or to train another model using the same configuration, you can pass in `train_configurations.yaml` (see [Specifying Model Configurations with a YAML File](yaml-config.md)) along with the `labels` filepath.
* `hparams.yaml`: Model hyperparameters. These are included in the checkpoint file as well.
* `time_distributed.ckpt`: Model checkpoint. You can continue training from this checkpoint by passing it to `zamba train` with the `--checkpoint` flag:
    ```console
    $ zamba train --checkpoint version_0/time_distributed.ckpt --data-dir example_vids/ --labels example_labels.csv
    ```
* `events.out.tfevents.1632250686.ip-172-31-15-179.14229.0`: [TensorBoard](https://www.tensorflow.org/tensorboard/get_started) logs. You can view these with tensorboard:
    ```console
    $ tensorboard --logdir version_0/
    ```
* `val_metrics.json`: The model's performance on the validation subset
* `test_metrics.json`: The model's performance on the test (holdout) subset
* `splits.csv`: Which files were used for training, validation, and as a holdout set. If split is specified in the labels file passed to training, `splits.csv` will not be saved out.

