# zamba

[![Docs Status](https://img.shields.io/badge/docs-stable-informational)](https://zamba.drivendata.org/docs/)
[![PyPI](https://img.shields.io/pypi/v/zamba.svg)](https://pypi.org/project/zamba/)
[![tests](https://github.com/drivendataorg/zamba/workflows/tests/badge.svg?branch=master)](https://github.com/drivendataorg/zamba/actions?query=workflow%3Atests+branch%3Amaster)
[![codecov](https://codecov.io/gh/drivendataorg/zamba/branch/master/graph/badge.svg)](https://codecov.io/gh/drivendataorg/zamba)

https://user-images.githubusercontent.com/46792169/138346340-98ee196a-5ecd-4753-b9df-380528091f9e.mp4

## [DOCUMENTATION](https://zamba.drivendata.org/docs/)

**`zamba` is a tool built in Python that uses machine learning and computer vision to automatically detect and classify animals in camera trap videos.** You can use `zamba` to:

- Filter out blank videos
- Identify which species appear in each video

Using state-of-the-art neural networks, the tool is trained to identify 42 common species from sites in Africa and Europe (as well as blank, or "no species present"). Users can also input their own labeled videos to finetune a model and make predictions for new species or new contexts.

> *zamba* means "forest" in Lingala, a Bantu language spoken throughout the Democratic Republic of the Congo and the Republic of the Congo.

### How do I access `zamba`?

`zamba` can be used both as a command-line tool and as a Python package. It is also available as a user-friendly website application, [Zamba Cloud](https://www.zambacloud.com/).

### Getting started

Follow the [Installation](https://zamba.drivendata.org/docs/install/) page. Once you have `zamba` installed, some good starting points are:

- The [Quickstart](https://zamba.drivendata.org/docs/quickstart/) page for basic examples of usage
- The user tutorial for either [classifying videos](https://zamba.drivendata.org/docs/predict-tutorial/) or [training a model](https://zamba.drivendata.org/docs/train-tutorial/) depending on what you want to do with `zamba`

### What models does `zamba` use to generate predictions?

`zamba` ships with [three model options](https://zamba.drivendata.org/docs/models/index/). `time_distributed` and `slowfast` are trained on 32 common species from central and west Africa. `european` is trained
on 11 common species from western Europe. `time_distributed` and `european` are image-based models while `slowfast` is a video-based model. Check out the [Wiki](https://github.com/drivendataorg/zamba/wiki) for community-submmitted models.

<br/><br/>

https://user-images.githubusercontent.com/46792169/137787221-de590183-042e-4d30-b32b-1d1c2cc96589.mov