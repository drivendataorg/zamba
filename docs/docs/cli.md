# zamba Command Line Interface

This section goes into more detail about the available options for
the `zamba` command line interface (CLI) tool. If you are new to `zamba` and just
want to classify some videos as soon as possible, see the [Quickstart](quickstart.md) guide.

## How to specify optional parameters

Each model that ships with the `zamba` package comes with a default configuration saved as a YAML file. The default value of each parameter listed below will be set based on the model being used - `time_distributed`, `slowfast`, or `european`. Default algorithm configurations can be found in `models/config`.
<!-- TODO: update path to default configs and add link to github folder><!-->

There are two ways to override the default model parameter value, listed in order or precendence: 

1. By passing an optional flag directly to the command line
2. By specifying the parameter in a custom YAML configuration file, and passing the YAML filepath to the command line with the `--config` flag. For more details on YAML configuration file options, see the [All Configuration Options](configurations.md) section. 

If a parameter is both passed directly to the CLI *and* specified in a YAMl file that is passed to the CLI, the value that is passed directly to the CLI will be used.

## `zamba predict` flags

As discussed in the [Quickstart](index.md) guide, the `--help` flag provides more information about options for `zamba`:

```
$ zamba predict --help 
Usage: zamba predict [OPTIONS]

  Identify species in a video.

  This is a command line interface for prediction on camera trap footage.
  Given a path to camera trap footage, the predict function use a deep
  learning model to predict the presence or absense of a variety of species
  of common interest to wildlife researchers working with camera trap data.

  If an argument is specified in both the command line and in a yaml file,
  the command line input will take precedence.

Options:
  --data-dir PATH                 Path to folder containing videos.
  --file-list PATH                Path to csv containing `filepath` column
                                  with videos.

  --model [time_distributed|slowfast|european]
                                  Model to use for inference. Model will be
                                  superseded by checkpoint if provided.
                                  [default: time_distributed]

  --checkpoint PATH               Model checkpoint path to use for inference.
                                  If provided, model is not required.

  --gpus INTEGER                  Number of GPUs to use for inference. If not
                                  specifiied, will use all GPUs found on
                                  machine.

  --batch-size INTEGER            Batch size to use for training.
  --save / --no-save              Whether to save out predictions to csv file.
  --dry-run / --no-dry-run        Runs one batch of inference to check for
                                  bugs.

  --config PATH                   Specify options using yaml configuration
                                  file instead of through command line
                                  options.

  --proba-threshold FLOAT         Probability threshold for classification
                                  between 0 and 1. If specified binary
                                  predictions are returned with 1 being
                                  greater than the threshold, 0 being less
                                  than or equal to. If not specified,
                                  probabilities between 0 and 1 are returned.

  --output-class-names / --no-output-class-names
                                  If True, we just return a video and the name
                                  of the most likely class. If False, we
                                  return a probability or indicator (depending
                                  on --proba_threshold) for every possible
                                  class.

  --weight-download-region [us|eu|asia]
                                  Server region for downloading weights.
  --cache-dir PATH                Path to directory for model weights.
                                  Alternatively, specify with environment
                                  variable `ZAMBA_CACHE_DIR`. If not
                                  specified, user's cache directory is used.

  -y, --yes                       Skip confirmation of configuration and
                                  proceed right to prediction.  [default:
                                  False]

  --help                          Show this message and exit.
```

All of the above besides `model` and `yes` can also be specified in a yaml file. Let's go through the flags one by one.

#### `--data-dir PATH`

Path to the folder containing your videos.

#### `--file-list PATH`

Path to a CSV file with a column for the filepath to each video you want to classify. The CSV must hae a column for `filepath`.

#### `--model [time_distributed|slowfast|european]`

There are three versions of the algorithm that ship with zamba: `time_distributed`, `slowfast`, and `european`. If `european` is passed, the model trained on European species will be run. `time_distributed` is the default.

#### `--checkpoint PATH`

Path to a model checkpoint to load and use for inference. To load a model from a checkpoint, the model name must also be specified. The default is `None`, which automatically loads the pretrained checkpoint for the model specified by `model-name`.

#### `--gpus INT`

The number of GPUs to use during inference. By default, all of the available GPUs found on the machine will be used. An error will be raised if the number of GPUs specified is more than the number that are available on the machine.

#### `--batch-size INT`

The batch size to use for prediction.

#### `--save`

Whether to save out the predictions to a CSV file. Predictions will be saved by default to `{model name}_{current timestamp}_preds.csv`.
For example, running `zamba predict` with the `time_distributed` model on 9/15/21 will save out predictions at `time_distributed_2021-09-15_preds.csv`. Specify `--no-save` if you would not like to write out predictions.

#### `--dry-run`

Specifying `--dry-run` is useful for trying out model implementations more quickly by running only a single batch of inference. The default is `--no-dry-run`.

#### `--config`

Path to a yaml configuration file with values for other arguments to `predict`. If a value is specified in both the command line and in a yaml file that is passed, the command line argument value will be used. Each default model (`time_distributed`, `slowfast`, and `european`) comes with a yaml file that sets default configurations. If `--config` is not specified, these default values will be used for any argument that is not passed directly to the command line. Default configs can be found in `models/configs`.

#### `--proba-threshold FLOAT`

For advanced uses, you may want the algorithm to be more or less sensitive to if a species is present. This parameter is a `FLOAT` number, e.g., `0.64` corresponding to the probability threshold beyond which an animal is considered to be present in the video being analyzed.

By default no threshold is passed, `proba_threshold=None`. This will return a probability from 0-1 for each species that could occur in each video. If a threshold is passed, then the final prediction value returned for each class is `probability >= proba_threshold`, so that all class values become `0` (`False`, the species does not appear) or `1` (`True`, the species does appear).

#### `--output-class-names`

Setting this option to `True` yields the most concise output `zamba` is capable of. The highest species probability in a video is taken to be the _only_ species in that video, and the output returned is simply the video name and the name of the species with the highest class probability, or `blank` if the most likely classification is no animal. See the [Quickstart](index.md) for example usage.

#### `--weight_download_region [us|eu|asia]` 

Because `zamba` needs to download pretrained weights for the neural network architecture, we make these weights available in different regions. `us` is the default, but if you are not in the US you should use either `eu` for the European Union or `asia` for Asia Pacific to make sure that these download as quickly as possible for you.

#### `--yes`

By default, the command line interface will print the specifications you have entered and ask for confirmation before starting inference. Specifying `--yes` or `-y` skips this step and kicks off prediction without confirmation. 

## `zamba train` flags

```
$ zamba train --help
Usage: zamba train [OPTIONS]

  Train a model using the provided data, labels, and model name.

  If an argument is specified in both the command line and in a yaml file,
  the command line input will take precedence.

Options:
  --data-dir PATH                 Path to folder containing videos.
  --labels PATH                   Path to csv containing video labels.
  --model [time_distributed|slowfast|european]
                                  Model to train. Model will be superseded by
                                  checkpoint if provided.  [default:
                                  time_distributed]

  --checkpoint PATH               Model checkpoint path to use for training.
                                  If provided, model is not required.

  --config PATH                   Specify options using yaml configuration
                                  file instead of through command line
                                  options.

  --batch-size INTEGER            Batch size to use for training.
  --gpus INTEGER                  Number of GPUs to use for training. If not
                                  specifiied, will use all GPUs found on
                                  machine.

  --dry-run / --no-dry-run        Runs one batch of train and validation to
                                  check for bugs.

  --weight-download-region [us|eu|asia]
                                  Server region for downloading weights.
  --cache-dir PATH                Path to directory for model weights.
                                  Alternatively, specify with environment
                                  variable `ZAMBA_CACHE_DIR`. If not
                                  specified, user's cache directory is used.

  -y, --yes                       Skip confirmation of configuration and
                                  proceed right to training.  [default: False]

  --help                          Show this message and exit.
```

All of the above besides `model` and `yes` can also be specified in a yaml file. Let's go through the flags one by one.
 
#### `--data-dir PATH`

Path to the folder containing your videos.

#### `--labels PATH`

Path to a CSV containing the video labels to use as ground truth during training.

#### `--model [time_distributed|slowfast|european]`

There are three versions of the algorithm that ship with zamba: `time_distributed`, `slowfast`, and `european`. If `european` is passed, the model trained on European species will be run. `time_distributed` is the default.

#### `--config`

Path to a yaml configuration file with values for other arguments to `predict`. If a value is specified in both the command line and in a yaml file that is passed, the command line argument value will be used. Each default model (`time_distributed`, `slowfast`, and `european`) comes with a yaml file that sets default configurations. If `--config` is not specified, these default values will be used for any argument that is not passed directly to the command line. Default configs can be found in `models/configs`.

#### `--batch-size INT`

The batch size to use for training.

#### `--gpus INT`

The number of GPUs to use during training. By default, all of the available GPUs found on the machine will be used. An error will be raised if the number of GPUs specified is more than the number that are available on the machine. 

#### `--dry-run`

Specifying `--dry-run` is useful for debugging more quickly by running only a single batch of train and validation. The default is `--no-dry-run`.

#### `--yes`

By default, the command line interface will print the specifications you have entered and ask for confirmation before starting to train. Specifying `--yes` or `-y` skips this step and kicks off training without confirmation. 
