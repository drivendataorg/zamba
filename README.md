# Zamba

[![Docs Status](https://img.shields.io/badge/docs-stable-informational)](https://zamba.drivendata.org/docs/)
[![tests](https://github.com/drivendataorg/zamba/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/drivendataorg/zamba/actions/workflows/tests.yml?query=branch%3Amaster)
[![codecov](https://codecov.io/gh/drivendataorg/zamba/branch/master/graph/badge.svg)](https://codecov.io/gh/drivendataorg/zamba)
<!-- [![PyPI](https://img.shields.io/pypi/v/zamba.svg)](https://pypi.org/project/zamba/) -->

https://user-images.githubusercontent.com/46792169/138346340-98ee196a-5ecd-4753-b9df-380528091f9e.mp4

> *Zamba* means "forest" in Lingala, a Bantu language spoken throughout the Democratic Republic of the Congo and the Republic of the Congo.

**`zamba` is a tool built in Python that uses machine learning and computer vision to automatically detect and classify animals in camera trap _images_ and _videos_.** You can use `zamba` to:

- Identify which species appear in each image or video
- Filter out blank images or videos
- Create your own custom models that identify your species in your habitats
- Estimate the distance between animals in the frame and the video camera
- And more! 🙈 🙉 🙊

The official video models in `zamba` can identify blank videos (where no animal is present) along with 32 species common to Africa and 11 species common to Europe. The official image models can identify 178 species from throughout the world. Users can also finetune models using their own labeled images and videos to then make predictions for new species and/or new ecologies.

`zamba` can be used both as a command-line tool and as a Python package. It is also available as a user-friendly website application, [Zamba Cloud](https://www.zambacloud.com/).

We encourage people to share their custom models trained with Zamba. If you train a model and want to make it available, please add it to the [Model Zoo Wiki](https://github.com/drivendataorg/zamba/wiki) for others to be able to use!

Visit https://zamba.drivendata.org/docs/ for full documentation and tutorials.

## Installing `zamba`

First, make sure you have the prerequisites installed:

* Python >= 3.11
* FFmpeg > 4.3

Then run:
```console
pip install https://github.com/drivendataorg/zamba/releases/latest/download/zamba.tar.gz
```

See the [Installation](https://zamba.drivendata.org/docs/stable/install/) page of the documentation for details.

## Getting started

Once you have `zamba` installed, some good starting points are:

- The [Quickstart](https://zamba.drivendata.org/docs/stable/quickstart/) page for basic examples of usage
- The user tutorials for [classifying images](https://zamba.drivendata.org/docs/stable/images-predict-tutorial/), [classifying videos](https://zamba.drivendata.org/docs/stable/predict-tutorial/), or [training a model](https://zamba.drivendata.org/docs/stable/train-tutorial/)

## Example usage

Once `zamba` is installed, you can see the basic command options with:
```console
$ zamba --help

 Usage: zamba [OPTIONS] COMMAND [ARGS]...

 Zamba is a tool built in Python to automatically identify the species seen in camera trap
 videos from sites in Africa and Europe. Visit https://zamba.drivendata.org/docs for more
 in-depth documentation.

╭─ Options ─────────────────────────────────────────────────────────────────────────────────╮
│ --version                     Show zamba version and exit.                                │
│ --install-completion          Install completion for the current shell.                   │
│ --show-completion             Show completion for the current shell, to copy it or        │
│                               customize the installation.                                 │
│ --help                        Show this message and exit.                                 │
╰───────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ────────────────────────────────────────────────────────────────────────────────╮
│ densepose      Run densepose algorithm on videos.                                         │
│ depth          Estimate animal distance at each second in the video.                      │
│ predict        Identify species in a video.                                               │
│ train          Train a model on your labeled data.                                        │
| image          Tools for working with images instead of videos.                           |
╰───────────────────────────────────────────────────────────────────────────────────────────╯
```

`zamba` can be used "out of the box" to generate predictions or train a model using your own images and videos. `zamba` supports the same image formats as [`pillow`](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#fully-supported-formats) and the same video formats as FFmpeg, [which are listed here](https://www.ffmpeg.org/general.html#Supported-File-Formats_002c-Codecs-or-Features). Any images or videos that fail a set of validation checks will be skipped during inference or training.

### Classifying unlabeled images and videos

Zamba classifies videos by default, but can easily be set to classify images instead. To get classifications for videos:

```console
$ zamba predict --data-dir path/to/videos
```
and for images:

```console
$ zamba image predict --data-dir path/to/videos
```

By default, predictions will be saved to `zamba_predictions.csv`. Run `zamba predict --help` or `zamba image predict --help` to list all possible options to pass to `predict`.

See the [Quickstart](https://zamba.drivendata.org/docs/stable/quickstart/) page or the user tutorial on [classifying images](https://zamba.drivendata.org/docs/stable/images-predict-tutorial/) or [classifying videos](https://zamba.drivendata.org/docs/stable/predict-tutorial/) for more details.

### Training a model

Zamba defaults to training a model for classifying videos:
```console
$ zamba train --data-dir path/to/videos --labels path_to_labels.csv --save_dir my_trained_model
```

Training a model for images is similar:
```console
$ zamba image train --data-dir path/to/images --labels path_to_labels.csv --save_dir my_trained_model
```

The newly trained model will be saved to the specified save directory. The folder will contain a model checkpoint as well as training configuration, model hyperparameters, and validation and test metrics. Run `zamba train --help` or `zamba image train --help` to list all possible options to pass to `train`.

You can use your trained model on new images or videos by editing the `train_configuration.yaml` that is generated by `zamba`. Add a `predict_config` section to the yaml that points to the checkpoint file that is generated:

```yaml
...
# generated train_config
...

predict_config:
  checkpoint: PATH_TO_YOUR_CHECKPOINT_FILE

```

Now you can pass this configuration to the command line. See the [Quickstart](https://zamba.drivendata.org/docs/stable/quickstart/) page or the user tutorial on [training a model](https://zamba.drivendata.org/docs/stable/train-tutorial/) for more details.

You can then share your model with others by adding it to the [Model Zoo Wiki](https://github.com/drivendataorg/zamba/wiki).

### Estimating distance between animals and the camera

Depth-estimation models are also supported, but only for video files. For example:
```console
$ zamba depth --data-dir path/to/videos
```

By default, predictions will be saved to `depth_predictions.csv`. Run `zamba depth --help` to list all possible options to pass to `depth`.

See the [depth estimation page](https://zamba.drivendata.org/docs/stable/models/depth/) for more details.


## Contributing

We would love your contributions of code fixes, new models, additional training data, docs revisions, and anything else you can bring to the project!

See the docs page on [contributing to `zamba`](https://zamba.drivendata.org/docs/stable/contribute/) for details.
