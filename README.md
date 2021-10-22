# zamba

[![Docs Status](https://img.shields.io/badge/docs-stable-informational)](https://zamba.drivendata.org/docs/)
[![PyPI](https://img.shields.io/pypi/v/zamba.svg)](https://pypi.org/project/zamba/)
[![tests](https://github.com/drivendataorg/zamba/workflows/tests/badge.svg?branch=master)](https://github.com/drivendataorg/zamba/actions?query=workflow%3Atests+branch%3Amaster)
[![codecov](https://codecov.io/gh/drivendataorg/zamba/branch/master/graph/badge.svg)](https://codecov.io/gh/drivendataorg/zamba)

https://user-images.githubusercontent.com/46792169/138346340-98ee196a-5ecd-4753-b9df-380528091f9e.mp4

> *zamba* means "forest" in Lingala, a Bantu language spoken throughout the Democratic Republic of the Congo and the Republic of the Congo.

**`zamba` is a tool built in Python that uses machine learning and computer vision to automatically detect and classify animals in camera trap videos.** You can use `zamba` to:

- Filter out blank videos
- Identify which species appear in each video

The tool is already trained to identify 42 species common to Africa and Europe (as well as blank, or "no species present"). Users can also input their own labeled videos to finetune a model and make predictions for new species or new contexts.

`zamba` can be used both as a command-line tool and as a Python package. It is also available as a user-friendly website application, [Zamba Cloud](https://www.zambacloud.com/).

Please visit https://zamba.drivendata.org/docs/ for documentation and tutorials.

Check out the [Wiki](https://github.com/drivendataorg/zamba/wiki) for community-submmitted models.

## Installing `zamba`

First, make sure you have the prerequisites installed:
* Python 3.7 or 3.8
* FFmpeg

Then run:
```console
pip install zamba
```

See the [Installation](https://zamba.drivendata.org/docs/install.html) page of the documentation for details.

## Getting started

Once you have `zamba` installed, some good starting points are:

- The [Quickstart](https://zamba.drivendata.org/docs/quickstart/) page for basic examples of usage
- The user tutorial for either [classifying videos](https://zamba.drivendata.org/docs/predict-tutorial/) or [training a model](https://zamba.drivendata.org/docs/train-tutorial/) depending on what you want to do with `zamba`
- 
## Example usage

Once `zamba` is installed, you can see the basic command options with:
```console
$ zamba --help
Usage: zamba [OPTIONS] COMMAND [ARGS]...

  Zamba is a tool built in Python to automatically identify the species seen
  in camera trap videos from sites in Africa and Europe. Visit
  https://zamba.drivendata.org/docs for more in-depth documentation.

Options:
  --version             Show zamba version and exit.
  --install-completion  Install completion for the current shell.
  --show-completion     Show completion for the current shell, to copy it or
                        customize the installation.
  --help                Show this message and exit.

Commands:
  predict  Identify species in a video.
  train    Train a model on your labeled data.
```

`zamba` can be used "out of the box" to generate predictions or train a model using your own videos. `zamba` supports the same video formats as FFmpeg, [which are listed here](https://www.ffmpeg.org/general.html#Supported-File-Formats_002c-Codecs-or-Features). Any videos that fail a set of FFmpeg checks will be skipped during inference or training.

**Classifying unlabeled videos**

```console
$ zamba predict --data-dir path/to/videos
```

By default, predictions will be saved to `zamba_predictions.csv`. Run `zamba predict --help` to list all possible options to pass to `predict`.

See the [Quickstart](https://zamba.drivendata.org/docs/quickstart/) page or the user tutorial on [classifying videos](https://zamba.drivendata.org/docs/predict-tutorial/) for more details.

**Training a model**

```console
$ zamba train --data-dir path/to/videos --labels path_to_labels.csv
```

The newly trained model will be saved to a folder in the current working directory called `zamba_{model_name}`. For example, a model finetuned from the pretrained `time_distributed` model (the default) will be saved in `zamba_time_distributed`. The folder will contain a model checkpoint as well as training configuration, model hyperparameters, and test and validation metrics. Run `zamba train --help` to list all possible options to pass to `train`.

See the [Quickstart](https://zamba.drivendata.org/docs/quickstart/) page or the user tutorial on [training a model](https://zamba.drivendata.org/docs/train-tutorial/) for more details.

### Running the `zamba` test suite

The included [`Makefile`](https://github.com/drivendataorg/zamba/blob/master/Makefile) contains code that uses pytest to run all tests in `zamba/tests`.

The command is (from the project root):

```console
$ make tests
```

See the docs page on [contributing to `zamba`](https://zamba.drivendata.org/docs/contribute.html) for details.