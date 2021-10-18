# zamba

[![Docs Status](https://img.shields.io/badge/docs-stable-informational)](https://zamba.drivendata.org/docs/)
[![PyPI](https://img.shields.io/pypi/v/zamba.svg)](https://pypi.org/project/zamba/)
[![tests](https://github.com/drivendataorg/zamba/workflows/tests/badge.svg?branch=master)](https://github.com/drivendataorg/zamba/actions?query=workflow%3Atests+branch%3Amaster)
[![codecov](https://codecov.io/gh/drivendataorg/zamba/branch/master/graph/badge.svg)](https://codecov.io/gh/drivendataorg/zamba)

*Zamba means "forest" in the Lingala language.*

Zamba is a tool built in Python that uses machine learning and computer vision to automatically detect and classify animals in camera trap videos. You can use Zamba to:

- Filter out blank videos
- Identify which species appear in each video

The tool is already trained to recognize 42 species common to Africa and Europe (as well as blank, or "no species present"). Zamba can also finetune an existing model based on additional labeled videos to make predictions for new species or new contexts. Zamba can be used both as a command-line tool and as a Python package. It is also available as an accessible website application, [Zamba Cloud](https://www.zambacloud.com/).

Please visit https://zamba.drivendata.org/docs/ for documentation and tutorials.

Check out the [Wiki](https://github.com/drivendataorg/zamba/wiki) for community-submmitted models.

https://user-images.githubusercontent.com/46792169/137787221-de590183-042e-4d30-b32b-1d1c2cc96589.mov

## Installing `zamba`

First, make sure you have the prerequisites installed:
* Python 3.7 or 3.8
* FFmpeg

Then run:
```console
pip install zamba
```

See the [Installation](https://zamba.drivendata.org/docs/install.md) page of the documentation for details.

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

You can run zamba on any directory of video files. `zamba` supports the same video formats as FFmpeg, [which are listed here](https://www.ffmpeg.org/general.html#Supported-File-Formats_002c-Codecs-or-Features). Any videos that fail a set of FFmpeg checks will be skipped during inference or training.

**`zamba` needs to download the "weights" files for the neural networks that it uses to make predictions. On first run it will download ~200-500 MB of files with these weights depending which model you choose.**

### Classifying unlabeled videos

```console
$ zamba predict --data-dir path/to/videos
```

By default, predictions will be saved to `zamba_predictions.csv`. The output csv will have rows labeled by each video filename and columns for each species. The default prediction will store all class probabilities, so that cell (i,j) is the probability that animal j is present in video i.

To see all possible options to can pass to `predict`, run `zamba predict --help` or see the All Optional Arguments page of the [docs](https://zamba.drivendata.org/docs/). For more details, see the [tutorial](https://zamba.drivendata.org/docs/predict-tutorial/).

### Training a model

```console
$ zamba train --data-dir path/to/videos --labels path_to_labels.csv
```

The newly trained model will be saved to a folder in the current working directory called `zamba_{model_name}`. For example, a model finetuned from the provided `time_distributed` model (the default) will be saved in `zamba_time_distributed`. The folder will contain a model checkpoint as well as training configuration, model hyperparameters, and test and validation metrics.

The labels csv must have columns for both filepath and label. Filepaths should be either full paths or relative to `data-dir`.

To see all possible options to can pass to `train`, run `zamba train --help` or see the All Optional Arguments page of the [docs](https://zamba.drivendata.org/docs/). For more details, see the [tutorial](https://zamba.drivendata.org/docs/train-tutorial/).

### Running the `zamba` test suite

The included `Makefile` contains code that uses pytest to run all tests in `zamba/tests`.

The command is (from the project root):

```console
$ make tests
```

See the docs page on [contributing to `zamba`](https://zamba.drivendata.org/docs/contribute.html) for details.




