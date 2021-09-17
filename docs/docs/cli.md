# zamba Command Line Interface

This section goes into more detail about the available options for
the `zamba` command line interface (CLI) tool. If you are new to `zamba` and just
want to classify some videos as soon as possible, see the [Quickstart](quickstart.md) guide.

## How to specify optional parameters

Each model that ships with the `zamba` package comes with a default configuration saved as a YAML file. The default value of each parameter listed below will be set based on the model being used - `time_distributed`, `slowfast`, or `european`. Default algorithm configurations can be found in `models/config`.
<!-- TODO: update path to default configs and add link to github folder><!-->

There are two ways to override the default model parameter value, listed in order of precendence: 

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

### Required arguments

To run `zamba predict`, you must specify either `--data-dir` or `--file-list`. All other flags are optional. To choose a model, either `--model` or `--checkpoint` must be specified. `--model` defaults to `time_distributed`.

#### `--data-dir PATH`
Path to the folder containing your videos.

#### `--file-list PATH`
Path to a CSV file with a column for the filepath to each video you want to classify. The CSV must hae a column for `filepath`.

### Arguments that cannot be specified in a YAML

All of the CLI flags listed by `--help` can be specified in a YAML configuration file except for `--config` and `--yes`. **For detailed explanations of the remaining command line optional flags, and additional parameters, see [All Configuration Options](configurations.md).**

#### `--config`
Path to a yaml configuration file with values for other arguments to `predict`. If a value is specified in both the command line and in a yaml file that is passed, the command line argument value will be used. Each default model (`time_distributed`, `slowfast`, and `european`) comes with a yaml file that sets default configurations. If `--config` is not specified, these default values will be used for any argument that is not passed directly to the command line. Default configs can be found in `models/configs`.

#### `--yes`
By default, the command line interface will print the specifications you have entered and ask for confirmation before starting inference. Specifying `--yes` or `-y` skips this step and kicks off prediction without confirmation. 

### Examples

<!-- TODO: add examples><!-->

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

### Required arguments.

To run `zamba train`, you must specify both `--data-directory` and `--labels`. To choose a model, either `--model` or `--checkpoint` must be specified. `--model` defaults to `time_distributed`.

#### `--data-dir PATH`
Path to the folder containing your labeled videos.

#### `--labels PATH`
Path to a CSV containing the video labels to use as ground truth during training.

### Arguments that cannot be specified in a YAML

All of the CLI flags listed by `--help` can be specified in a YAML configuration file except for `--config` and `--yes`. **For detailed explanations of the remaining command line optional flags, and additional parameters, see [All Configuration Options](configurations.md).**

#### `--config`

Path to a yaml configuration file with values for other arguments to `predict`. If a value is specified in both the command line and in a yaml file that is passed, the command line argument value will be used. Each default model (`time_distributed`, `slowfast`, and `european`) comes with a yaml file that sets default configurations. If `--config` is not specified, these default values will be used for any argument that is not passed directly to the command line. Default configs can be found in `models/configs`.

#### `--yes`

By default, the command line interface will print the specifications you have entered and ask for confirmation before starting to train. Specifying `--yes` or `-y` skips this step and kicks off training without confirmation. 

### Examples

<!-- TODO: add examples><!-->