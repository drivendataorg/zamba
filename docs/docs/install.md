# Installing `zamba`

Zamba has been developed and tested on macOS, Ubuntu Linux, and Windows for both CPU and
GPU configurations.

## To install `zamba`

### 1. Install prerequisites

Prerequisites:

 - Python >= 3.11
 - FFmpeg

#### [Python](https://www.python.org/) >= 3.11

We recommend [Python installation using Anaconda](https://www.anaconda.com/download/) for all platforms. For more information about how to install Anaconda, here are some useful YouTube videos of installation:

 - [Anaconda download link](https://www.anaconda.com/download/)
 - [macOS installation video](https://www.youtube.com/watch?v=nVlrpNf3EdM)


#### FFmpeg > 4.3 (only needed for video workflows)

[FFmpeg](https://ffmpeg.org/ffmpeg.html) is an open source library for loading videos of different codecs. Using FFmpeg means that `zamba` can be flexible in terms of the video formats we support. FFmpeg can be installed on all different platforms, but requires some additional configuration depending on the platform.

 - **Linux (Ubuntu/Debian):** `sudo apt update && sudo apt install ffmpeg`
 - **macOS:** Install [Homebrew](https://brew.sh/), then `brew install ffmpeg`
 - **Windows:** Install via [Chocolatey](https://chocolatey.org/) (`choco install ffmpeg`) or download from [ffmpeg.org](https://www.ffmpeg.org/download.html) and add the `bin` folder to your PATH.

To check that FFmpeg is installed, run `ffmpeg -version`:

```console
$ ffmpeg -version
ffmpeg version 4.4 Copyright (c) 2000-2021 the FFmpeg developers
...
```

### 2. Install `zamba`

We recommend [uv](https://docs.astral.sh/uv/) for installing Python packages and managing environments. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already. On macOS, run these commands in the terminal (⌘+space, "Terminal"). On Windows, run them in a command prompt, or if you installed Anaconda an anaconda prompt (Start > Anaconda3 > Anaconda Prompt).

`zamba` is split into optional extras so you only install the dependencies you need. With uv (recommended):

- **Video workflows** (species classification in camera trap videos):
```console
$ uv pip install zamba[video]
```

- **Image workflows** (species classification in camera trap images):
```console
$ uv pip install zamba[image]
```

- **Both video and image workflows**:
```console
$ uv pip install zamba[video,image]
```

With pip, use the same extras: `pip install zamba[video]`, `pip install zamba[image]`, or `pip install zamba[video,image]`. To install from the latest GitHub source instead of PyPI: `pip install "zamba[video] @ https://github.com/drivendataorg/zamba/releases/latest/download/zamba.tar.gz"` (and similarly for other extras).

To check what version of zamba you have installed:
```console
$ uv pip show zamba
```
Or with pip: `pip show zamba`.

To update zamba to the most recent version if needed:
```console
$ uv pip install -U zamba[video,image]
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

`zamba` works on Windows. The **image** extra installs without any additional tools. The **video** extra requires a C++ compiler because some dependencies (e.g. `pycocotools`) have native extensions.

#### Image workflows (no extra tools needed)

```console
uv pip install zamba[image]
```

#### Video workflows

1. **Install Visual Studio Build Tools** — download the [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/) installer and select the **"Desktop development with C++"** workload. This provides the `cl.exe` compiler needed to build native extensions.

2. **Install FFmpeg** — use [Chocolatey](https://chocolatey.org/) (`choco install ffmpeg`) or download from [ffmpeg.org](https://www.ffmpeg.org/download.html) and add the `bin` folder to your system PATH.

3. **Install zamba** (we recommend [uv](https://docs.astral.sh/uv/)):
```console
uv pip install zamba[video]
```
Or with pip: `pip install zamba[video]`.

Alternatively, you can use [Docker](https://www.docker.com/products/docker-desktop/) or [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) to run `zamba` in a Linux environment on Windows.

## Using GPU(s)

`zamba` is much faster on a machine with a graphics processing unit (GPU), but has also been developed and tested for machines without GPU(s).

To use a GPU, you must be using an
[NVIDIA GPU](https://www.nvidia.com/Download/index.aspx?lang=en-us),
have installed and configured [CUDA](https://developer.nvidia.com/cuda-downloads),
and have installed and configured [CuDNN](https://developer.nvidia.com/cudnn) per
their specifications.

If you are using `conda`, these dependencies can be installed through the [`cudatoolkit` package](https://anaconda.org/anaconda/cudatoolkit). If using a GPU, you will also want to make sure that you install a compatible version of PyTorch with the version of CUDA you use. See the [PyTorch installation docs](https://pytorch.org/get-started/locally/) for the easiest way to install the right version on your system.
