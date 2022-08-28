# Installing `zamba`

Zamba has been developed and tested on macOS and Ubuntu Linux for both CPU and
GPU configurations.

## To install `zamba`

### 1. Install prerequisites

Prerequisites:

 - Python 3.7 or 3.8
 - FFmpeg

#### [Python](https://www.python.org/) 3.7 or 3.8

We recommend [Python installation using Anaconda](https://www.anaconda.com/download/) for all platforms. For more information about how to install Anaconda, here are some useful YouTube videos of installation:

 - [Anaconda download link](https://www.anaconda.com/download/)
 - [Windows install video](https://www.youtube.com/watch?v=0OXBHvFeH_U)
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

`zamba`'s test suite runs regularly on the latest versions of [Ubuntu](https://www.ubuntu.com/).

### Windows

Note: Windows is not officially supported, but can be made to work. We recommend [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install) as a way to run the linux version of the tools on your Windows machine, which is supported.

If you want to use Windows natively, we recommend using the [git bash console](https://gitforwindows.org/) to run commands, and we recommend [Anaconda](https://docs.anaconda.com/anaconda/install/) (or [miniconda](https://docs.conda.io/en/latest/miniconda.html#) or [mamba](https://mamba.readthedocs.io/en/latest/installation.html)) as the easiest way to manage a number of requried dependencies.

Within the git bash console, you should [create an empty conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-environments) and then install a supported version of Python (e.g., `conda install python=3.8`).

Next, you should install some of the required dependencies for building zamba (particularly those for yolox):

```console
$ conda install cmake ffmpeg=4 ninja protobuf onnx=1.8.1 -c conda-forge
```

Once this is done, you can use the [pip command to install zamba](#2-install-zamba). As long as you use this conda environment with all the prerequisites, zamba should work.

## Using GPU(s)

`zamba` is much faster on a machine with a graphics processing unit (GPU), but has also been developed and tested for machines without GPU(s).

To use a GPU, you must be using an
[NVIDIA GPU](https://www.nvidia.com/Download/index.aspx?lang=en-us),
have installed and configured [CUDA](https://developer.nvidia.com/cuda-downloads),
and have installed and configured [CuDNN](https://developer.nvidia.com/cudnn) per
their specifications.

If you are using `conda`, these dependencies can be installed through the [`cudatoolkit` package](https://anaconda.org/anaconda/cudatoolkit). If using a GPU, you will also want to make sure that you install a compatible version of PyTorch with the version of CUDA you use. See the [PyTorch installation docs](https://pytorch.org/get-started/locally/) for the easiest way to install the right version on your system.
