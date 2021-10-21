# zamba

[![Docs Status](https://img.shields.io/badge/docs-stable-informational)](https://zamba.drivendata.org/docs/)
[![PyPI](https://img.shields.io/pypi/v/zamba.svg)](https://pypi.org/project/zamba/)
[![tests](https://github.com/drivendataorg/zamba/workflows/tests/badge.svg?branch=master)](https://github.com/drivendataorg/zamba/actions?query=workflow%3Atests+branch%3Amaster)
[![codecov](https://codecov.io/gh/drivendataorg/zamba/branch/master/graph/badge.svg)](https://codecov.io/gh/drivendataorg/zamba)

*Zamba means "forest" in the Lingala language.*

https://user-images.githubusercontent.com/46792169/138346340-98ee196a-5ecd-4753-b9df-380528091f9e.mp4

<div class="embed-responsive embed-responsive-16by9" width=500>
    <iframe width=600 height=340 class="embed-responsive-item" src="https://s3.amazonaws.com/drivendata-public-assets/monkey-vid.mp4" frameborder="0" allowfullscreen=""></iframe>
</div>

**Zamba is a tool built in Python that uses machine learning and computer vision to automatically detect and classify animals in camera trap videos.** You can use Zamba to:

- Filter out blank videos
- Identify which species appear in each video

Using state-of-the-art computer vision and machine learning, the tool is trained to identify 42 common species from sites in Africa and Europe (as well as blank, or "no species present"). Users can also input their own labeled videos to finetune a model and make predictions for new species or new contexts. 

## How do I access `zamba`?

`zamba` can be used both as a command-line tool and as a Python package. It is also available as a user-friendly website application, [Zamba Cloud](https://www.zambacloud.com/).

## What models does `zamba` use?

Zamba ships with three model options. `time_distributed` and `slowfast` are
trained on 32 common species from central and west Africa. `european` is trained
on 11 common species from western Europe. `time_distributed` and `european` are image-based models while `slowfast` is a video-based model. Check out the [Wiki](https://github.com/drivendataorg/zamba/wiki) for community-submmitted models.

https://user-images.githubusercontent.com/46792169/137787221-de590183-042e-4d30-b32b-1d1c2cc96589.mov

# Documentation

### Getting Started
- [Installing Zamba](https://zamba.drivendata.org/docs/install.md)
- [Quickstart](https://zamba.drivendata.org/docs/quickstart.md)

### User Tutorials
- [Classifying Unlabeled Videos](https://zamba.drivendata.org/docs/predict-tutorial.md)
- [Training a Model on Labeled Videos](https://zamba.drivendata.org/docs/train-tutorial.md)
- [Debugging](https://zamba.drivendata.org/docs/debugging.md)

### [Available Models](https://zamba.drivendata.org/docs/models/index.md)

### Advanced Options
- [All Optional Arguments](https://zamba.drivendata.org/docs/configurations.md)
- [Using YAML Configuration Files](https://zamba.drivendata.org/docs/yaml-config.md)
- [Guide to Common Optional Parameters](https://zamba.drivendata.org/docs/extra-options.md)

### [Contribute](https://zamba.drivendata.org/docs/contribute/index.md)

### Changelog
- [Version 2](https://zamba.drivendata.org/docs/v2_updates.md)