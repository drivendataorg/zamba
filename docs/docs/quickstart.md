# Quickstart

<script id="asciicast-1mXKsDiPzgyAZwk8CbdkrG2ac" src="https://asciinema.org/a/1mXKsDiPzgyAZwk8CbdkrG2ac.js" async data-autoplay="true" data-loop=1 data-cols=90></script>

This section assumes you have successfully installed `zamba` and are ready to train a model or identify species in your videos!

`zamba` can be used "out of the box" to generate predictions or train a model using your own videos. To perform inference, you simply need to run `zamba predict` followed by a set of arguments that let zamba know where your videos are located, which model you want to use, and where to save your output. To train a model, you can similarly run `zamba train` and specify your labels. The following sections provide details about these separate modules.

There are two ways to interact with the `zamba` package:

1. Use `zamba` as a command line interface tool. This page provides an overview of how to use the CLI.
2. Import `zamba` in Python and use it as a Python package.

This guide uses the CLI, but you can see the [prediction tutorial](predict-tutorial.md) or the [training tutorial](train-tutorial.md), which have both the CLI and Python approaches documented.

Installation is the same for both the command line interface tool and the Python package.

All of the commands on this page should be run at the command line. On
macOS, this can be done in the terminal (⌘+space, "Terminal"). On Windows, this can be done in a command prompt, or if you installed Anaconda an anaconda prompt (Start > Anaconda3 > Anaconda Prompt).

## How do I organize my videos for `zamba`?

You can specify the path to a directory of videos or specify a list of filepaths in a `.csv` file. `zamba` supports the same video formats as FFmpeg, [which are listed here](https://www.ffmpeg.org/general.html#Supported-File-Formats_002c-Codecs-or-Features). Any videos that fail a set of FFmpeg checks will be skipped during inference or training.

For example, say we have a directory of videos called `example_vids` that we want to generate predictions for using `zamba`. Let's list the videos:

```console
$ ls example_vids/
blank.mp4
chimp.mp4
eleph.mp4
leopard.mp4
```

Here are some screenshots from those videos:
<table class="table">
  <tbody>
    <tr>
      <td style="text-align:center">blank.mp4<br/>
        <img src="https://s3.amazonaws.com/drivendata-public-assets/zamba-2-blank-sm.jpg" alt="Blank frame seen from a camera trap" style="width:400px;"/>
      </td>
      <td style="text-align:center">chimp.mp4<br/>
        <img src="https://s3.amazonaws.com/drivendata-public-assets/zamba-2-chimp-sm.jpg" alt="Leopard seen from a camera trap" style="width:400px;"/>
      </td>
    </tr>
    <tr>
      <td style="text-align:center">eleph.mp4<br/>
        <img src="https://s3.amazonaws.com/drivendata-public-assets/zamba-2-eleph-sm.jpg" alt="Elephant seen from a camera trap" style="width:400px">
      </td>
      <td style="text-align:center">leopard.mp4<br/>
        <img src="https://s3.amazonaws.com/drivendata-public-assets/zamba-2-leopard-sm.jpg" alt="cat" style="width:400px;"/>
      </td>
    </tr>
  </tbody>
</table>

In this example, the videos have meaningful names so that we can easily
compare the predictions made by `zamba`. In practice, your videos will
probably be named something much less useful!

## Generating predictions

To generate and save predictions for your videos using the default settings, run:

```console
$ zamba predict --data-dir example_vids/
```

`zamba` will output a `.csv` file with rows labeled by each video filename and columns for each class (ie. species). The default prediction will store all class probabilities, so that cell `(i,j)` is *the probability that animal `j` is present in video `i`.*  Comprehensive predictions are helpful when a single video contains multiple species.

Predictions will be saved to `zamba_predictions.csv` in the current working directory by default. You can save out predictions to a different folder using the `--save-dir` argument.

Adding the argument `--output-class-names` will simplify the predictions to return only the *most likely* animal in each video:

```console
$ zamba predict --data-dir example_vids/ --output-class-names
$ cat zamba_predictions.csv
blank.mp4,blank
chimp.mp4,chimpanzee_bonobo
eleph.mp4,elephant
leopard.mp4,leopard
```

There are pretrained models that ship with `zamba`: `blank_nonblank`, `time_distributed`, `slowfast`, and `european`. Which model you should use depends on your priorities and geography (see the [Available Models](models/species-detection.md) page for more details). By default `zamba` will use the `time_distributed` model. Add the `--model` argument to specify one of other options:

```console
$ zamba predict --data-dir example_vids/ --model slowfast
```

## Training a model

You can continue training one of the [models](models/species-detection.md) that ships with `zamba` by either:

* Finetuning with additional labeled videos where the species are included in the list of [`zamba` class labels](models/species-detection.md#species-classes)
* Finetuning with labeled videos that include new species

In either case, the commands for training are the same. Say that we have labels for the videos in the `example_vids` folder saved in `example_labels.csv`. To train a model, run:

```console
$ zamba train --data-dir example_vids/ --labels example_labels.csv
```

The labels file must have columns for both filepath and label. The filepath column should contain either absolute paths or paths relative to the `data-dir`. Optionally, there can also be columns for `split` (`train`, `val`, or `holdout`) and `site`. Let's print the example labels:

```console
$ cat example_labels.csv
filepath,label
blank.MP4,blank
chimp.MP4,chimpanzee_bonobo
eleph.MP4,elephant
leopard.MP4,leopard
```

By default, the trained model and additional training output will be saved to a `version_n` folder in the current working directory. For example,

```console
$ zamba train --data-dir example_vids/ --labels example_labels.csv
$ ls version_0/
hparams.yaml
time_distributed.ckpt
train_configuration.yamml
val_metrics.json
...
```

## Downloading model weights

**`zamba` needs to download the "weights" files for the models it uses to make predictions. On first run, it will download ~200-500 MB of files with these weights depending which model you choose.**
Once a model's weights are downloaded, `zamba` will use the local version and will not need to perform this download again. If you are not in the United States, we recommend running the above command with the additional flag either `--weight_download_region eu` or `--weight_download_region asia` depending on your location. The closer you are to the server, the faster the downloads will be.

<a id='getting-help'></a>

## Getting help

Once zamba is installed, you can see more details of each function with `--help`. For example, you can run `zamba predict --help`:

```console
Usage: zamba predict [OPTIONS]

  Identify species in a video.

  This is a command line interface for prediction on camera trap footage.
  Given a path to camera trap footage, the predict function use a deep
  learning model to predict the presence or absense of a variety of species of
  common interest to wildlife researchers working with camera trap data.

  If an argument is specified in both the command line and in a yaml file, the
  command line input will take precedence.

Options:
  --data-dir PATH                 Path to folder containing videos.
  --filepaths PATH                Path to csv containing `filepath` column
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
  --save / --no-save              Whether to save out predictions. If you want
                                  to specify the output directory, use
                                  save_dir instead.
  --save-dir PATH                 An optional directory in which to save the
                                  model predictions and configuration yaml.
                                  Defaults to the current working directory if
                                  save is True.
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
  --num-workers INTEGER           Number of subprocesses to use for data
                                  loading.
  --weight-download-region [us|eu|asia]
                                  Server region for downloading weights.
  --skip-load-validation / --no-skip-load-validation
                                  Skip check that verifies all videos can be
                                  loaded prior to inference. Only use if
                                  you're very confident all your videos can be
                                  loaded.
  -o, --overwrite                 Overwrite outputs in the save directory if
                                  they exist.
  -y, --yes                       Skip confirmation of configuration and
                                  proceed right to prediction.
  --help                          Show this message and exit.
```

Or if you are training a model, you can run `zamba train --help`:

```console
$ zamba train --help

Usage: zamba train [OPTIONS]

  Train a model on your labeled data.

  If an argument is specified in both the command line and in a yaml file, the
  command line input will take precedence.

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
  --save-dir PATH                 An optional directory in which to save the
                                  model checkpoint and configuration file. If
                                  not specified, will save to a `version_n`
                                  folder in your working directory.
  --num-workers INTEGER           Number of subprocesses to use for data
                                  loading.
  --weight-download-region [us|eu|asia]
                                  Server region for downloading weights.
  --skip-load-validation / --no-skip-load-validation
                                  Skip check that verifies all videos can be
                                  loaded prior to training. Only use if you're
                                  very confident all your videos can be
                                  loaded.
  -y, --yes                       Skip confirmation of configuration and
                                  proceed right to training.
  --help                          Show this message and exit.
```
