# Installing `zamba`

Zamba has been developed and tested on macOS and Ubuntu Linux for both CPU and
GPU configurations.

## To install `zamba`

### 1. Install prerequisites

Prerequisites:

 - Python 3.11 or 3.12
 - FFmpeg

#### [Python](https://www.python.org/) 3.11 or 3.12

We recommend [Python installation using Anaconda](https://www.anaconda.com/download/) for all platforms. For more information about how to install Anaconda, here are some useful YouTube videos of installation:

 - [Anaconda download link](https://www.anaconda.com/download/)
 - [macOS installation video](https://www.youtube.com/watch?v=nVlrpNf3EdM)


#### FFmpeg version 4

[FFmpeg](https://ffmpeg.org/ffmpeg.html) is an open source library for loading videos of different codecs. Using FFmpeg means that `zamba` can be flexible in terms of the video formats we support. FFmpeg can be installed on all different platforms, but requires some additional configuration depending on the platform. Here are some videos and instructions walking through FFmpeg installation:

 - [FFmpeg download link](https://www.ffmpeg.org/download.html)
 - [Install on Ubuntu or Linux](https://www.tecmint.com/install-ffmpeg-in-linux/).
     - In the command line, enter `sudo apt update` and then `sudo apt install ffmpeg`.
 - [MacOS install video](https://www.youtube.com/watch?v=8nbuqYw2OCw&t=5s)
     - First, install [Homebrew](https://brew.sh/). Then run `brew install ffmpeg@4`
     - Follow the brew instructions to add FFmpeg to your path.

To check that `FFmpeg` is installed, run `ffmpeg`:

```console
$ ffmpeg

ffmpeg version 4.4 Copyright (c) 2000-2021 the FFmpeg developers
  built with Apple clang version 12.0.0 (clang-1200.0.32.29)
...
```

To check your installed version, run `ffmpeg -version`.

### 2. Install `zamba`

On macOS, run these commands in the terminal (⌘+space, "Terminal"). On Windows, run them in a command prompt, or if you installed Anaconda an anaconda prompt (Start > Anaconda3 > Anaconda Prompt).

To install zamba:
```console
$ pip install https://github.com/drivendataorg/zamba/releases/latest/download/zamba.tar.gz
```

To check what version of zamba you have installed:
```console
$ pip show zamba
```

To update zamba to the most recent version if needed:
```console
$ pip install -U https://github.com/drivendataorg/zamba/releases/latest/download/zamba.tar.gz
```


## Operating systems that have been tested

### macOS

`zamba` has been tested on macOS High Sierra.

### Linux

`zamba` has been tested on [Ubuntu](https://www.ubuntu.com/) regularly since 16.04. Tests run every week against the [`ubuntu-latest` Github runner environment](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners/about-github-hosted-runners#supported-runners-and-hardware-resources), so the version that Github uses is most likely to work. As of October 2023, that is 22.04.

Note, for Linux, you may need to install additional system packages to get `zamba` to work. For example, on Ubuntu, you may need to install `build-essentials` to have compilers.

FFMpeg 4, build-essentials, and some other packages that include more codecs to support additional videos and some other utilities can be installed with:

```bash
apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:savoury1/ffmpeg4 && \
    apt-get update && \
    apt-get -y install \
    build-essential \
    ffmpeg \
    git \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    libsm6 \
    libswresample-dev \
    libswscale-dev \
    libxext6 \
    pkg-config \
    wget \
    x264 \
    x265
```

### Windows

**Note: `zamba` does not currently work on Windows because one of our dependencies fails to build.**

You can try using [Docker](https://www.docker.com/products/docker-desktop/) or [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) to run `zamba` on a Linux OS like Ubuntu on top of your Windows machine.

## Using GPU(s)

`zamba` is much faster on a machine with a graphics processing unit (GPU), but has also been developed and tested for machines without GPU(s).

To use a GPU, you must be using an
[NVIDIA GPU](https://www.nvidia.com/Download/index.aspx?lang=en-us),
have installed and configured [CUDA](https://developer.nvidia.com/cuda-downloads),
and have installed and configured [CuDNN](https://developer.nvidia.com/cudnn) per
their specifications.

If you are using `conda`, these dependencies can be installed through the [`cudatoolkit` package](https://anaconda.org/anaconda/cudatoolkit). If using a GPU, you will also want to make sure that you install a compatible version of PyTorch with the version of CUDA you use. See the [PyTorch installation docs](https://pytorch.org/get-started/locally/) for the easiest way to install the right version on your system.
