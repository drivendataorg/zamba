# User Tutorial: Training a Model on Labaled Videos

This section walks through how to train a model using `zamba`. If you are new to `zamba` and just want to classify some videos as soon as possible, see the [Quickstart](quickstart.md) guide.

This tutorial goes over the steps for using `zamba` if:

* You already have `zamba` installed (for details see the [Installation](install.md) page)
* You have labeled videos that you want to use to train or finetune a model

`zamba` can run two types of model training:

* Fine-tuning a model with labels that are a subset of the possible [zamba labels](models.md#species-classes)
* Fine-tuning a model to predict an entirely new set of labels

The process is the same for both cases.

## Basic usage: command line interface

Say that we want to finetune the `time_distributed` model based on the videos in `example_vids` and the labels in `example_labels.csv`. 

Minimum example for training in the command line:

```console
$ zamba train --data-dir example_vids/ --labels example_labels.csv
```

### Required arguments

To run `zamba train` in the command line, you must specify both `--data-directory` and `--labels`.

* **`--data-dir PATH`:** Path to the folder containing your labeled videos. `zamba` will generate predictions for videos in the top level directory and in any nested folders.
* **`--labels PATH`:** Path to a CSV containing the video labels to use as ground truth during training. There must be columns for both filepath and label. Filepaths should be either absolute paths or relative to `data-dir`. Optionally, there can also be columns for `split` (`train`, `val`, or `holdout`) and `site`. If your labels file does not have a column for `split`, you can alternately use the `split_proportions` argument.

```console
$ cat example_labels.csv

filepath,label
blank.MP4,blank
chimp.MP4,chimpanzee_bonobo
eleph.MP4,elephant
leopard.MP4,leopard
```

## Basic usage: Python package

Say that we want to finetune the `time_distributed` model based on the videos in `example_vids` and the labels in `example_labels.csv`. 

Minimum example for training using the Python package:

```python
from zamba.models.model_manager import train_model
from zamba.models.config import TrainConfig

train_config = TrainConfig(
    data_directory="example_vids/", labels="example_labels.csv"
)
train_model(train_config=train_config)
```

The only two arguments that can be passed to `train_model` are `train_config` and (optionally) `video_loader_config`. The first step is to instantiate [`TrainConfig`](configurations.md#training-arguments). Optionally, you can also specify video loading arguments by instantiating and passing in [`VideoLoaderConfig`](configurations.md#video-loading-arguments). 

### Required arguments

To run `train_model` in Python, you must specify both `data_directory` and `labels` when `TrainConfig` is instantiated.

* **`data_directory (DirectoryPath)`:** Path to the folder containing your videos.

* **`labels (FilePath or pd.DataFrame)`:** Either the path to a CSV file with labels for training, or a dataframe of the training labels. There must be columns for `filename` and `label`.

For detailed explanations of all possible configuration arguments, see [All Optional Arguments](configurations.md).

## Default behavior

By default, the [`time_distributed`](models.md#time-distributed-model) model will be used as a starting point. The newly trained model will be saved to a folder in the current working directory called `zamba_{model_name}`. For example, a model finetuned from the provided `time_distributed` model (the default) will be saved in `zamba_time_distributed`. 

`zamba_time_distributed` contains:

* `train_configuration.yaml`: The full model configuration used to generate the given model, including `video_loader_config` and `train_config`. To continue training using the same configuration, or to train another model using the same configuration, you can pass in `train_configurations.yaml` (see [Specifying Model Configurations with a YAML File](yaml-config.md)).
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
* `time_distributed.ckpt`: Model checkpoint. The model checkpoint also includes both the model configuration in `train_configuration.yaml` and the model hyperparameters in `hparams.yaml`. You can continue training from this checkpoint by passing it to `zamba train` with the `--checkpoint` flag:
    ```console
    $ zamba train --checkpoint time_distributed.ckpt --data-dir example_vids/ --labels example_labels.csv
    ```
* `events.out.tfevents.1632250686.ip-172-31-15-179.14229.0`: [TensorBoard](https://www.tensorflow.org/tensorboard/get_started) logs
* `test_metrics.json`: The model's performance on the test subset
* `val_metrics.json`: The model's performance on the validation subset
* `splits.csv`: Which files were used for training, validation, and as a holdout set. If split is specified in the labels file passed to training, `splits.csv` will not be saved out.

## Step-by-step tutorial

### 1. Specify the path to your videos 

Save all of your videos within one folder.

* They can be in nested directories within the folder.
* Your videos should all be saved in formats that are suppored by FFmpeg, [which are listed here](https://www.ffmpeg.org/general.html#Supported-File-Formats_002c-Codecs-or-Features). Any videos that fail a set of FFmpeg checks will be skipped during inference or training.

Add the path to your video folder with `--data-dir`. For example, if your videos are in a folder called `example_vids`, add `--data-dir example_vids/` to your command.

=== "CLI"
    ```console
    $ zamba train --data-dir example_vids
    ```

=== "Python"
    ```python
    from zamba.models.model_manager import train_model
    from zamba.models.config import TrainConfig

    train_config = TrainConfig(data_directory='example_vids/')
    train_model(train_config=train_config)
    ```
Note that the above will not run yet because labels are not specified.

The more training data you have, the better the resulting model will be. It is recommended to have a minimum of 100 example videos per species. Having an imbalanced dataset - for example, one where most of the vidoes are blank - will not negatively impact training, as long as there are sufficient videos per species.

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
        data_directory='example_vids/', labels=labels_dataframe
    )
    train_model(train_config=train_config)
    ```

#### Labels `zamba` has seen before

Your labels may be included in the list of [`zamba` class labels](models.md#species-classes) that the provided models are trained to predict. If so, the relevant model that ships with `zamba` will essentially be used as a checkpoint, and model training will resume from that checkpoint.

#### Completely new labels

You can also train a model to predict completely new labels - the world is your oyster! (We'd love to see a model trained to predict oysters.) If this is the case, the model architecture will replace the final [neural network](https://www.youtube.com/watch?v=aircAruvnKk&t=995s) layer with a new head that predicts *your* labels instead of those that ship with `zamba`. [Backpropogation](https://www.youtube.com/watch?v=Ilg3gGewQ5U) will continue from that point with the new head. This process is called [transfer learning](https://keras.io/guides/transfer_learning/).

### 3. Choose a model for training

If your videos contain species common to central or west Africa, use the [`time_distributed` model](models.md#time-distributed). If they contain species common to western Europe, use the [`european` model](models.md#european). We do not recommend using the [`slowfast` model](models.md#slowfast) for training because it is much more computationally intensive and slower to run.

Add the model name to your command with `--model`. The `time_distributed` model will be used if no model is specified. For example, if you want to continue training the `european` model based on the videos in `example_euro_vids` and the labels in `example_euro_labels.csv`:

=== "CLI"
    ```console
    $ zamba train --data-dir example_euro_vids/ --labels example_euro_labels.csv --model european
    ```
=== "Python"
    ```python
    train_config = TrainConfig(
        data_directory="example_euro_vids/",
        labels="example_euro_labels.csv",
        model_name="european",
    )
    train_model(train_config=train_config)
    ```

### 4. Specify any additional parameters

And there's so much more! You can also do things like specify your region for faster model download (`--weight-download-region`), start training from a saved model checkpoint (`--checkpoint`), or specify a different path where your model should be saved (`--save-directory`). To read about a few common considerations, see the [Guide to Common Optional Parameters](extra-options.md) page.

### 5. Test your configuration with a dry run

Before kicking off the full model training, we recommend testing your code with a "dry run". This will run one training and validation batch for one epoch to quickly detect any bugs. See the [Debugging](debugging.md) page for details.