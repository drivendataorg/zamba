# Installing `zamba`

Zamba has been developed and tested on macOS, Ubuntu Linux, and Windows for both CPU and
GPU configurations.

`zamba` is split into optional extras so you only install the dependencies you need:

 - **Image workflows:** `zamba[image]`
 - **Video workflows:** `zamba[video]`
 - **Both image and video workflows:** `zamba[video,image]`

We recommend [uv](https://docs.astral.sh/uv/) for installing Python packages and managing environments. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already.

## Linux

`zamba` has been tested on [Ubuntu](https://www.ubuntu.com/) regularly since 16.04. Tests run every week against the [`ubuntu-latest` Github runner environment](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners/about-github-hosted-runners#supported-runners-and-hardware-resources), so the version that Github uses is most likely to work. As of October 2023, that is 22.04.

### 1. Install Python

Install [Python](https://www.python.org/) >= 3.11. With [uv](https://docs.astral.sh/uv/) you can install and pin a suitable version with `uv python install 3.11`.

### 2. Install FFmpeg for video workflows

[FFmpeg](https://ffmpeg.org/ffmpeg.html) is required for video workflows. Use FFmpeg > 4.3 and < 5. On Ubuntu/Debian:

```console
$ sudo apt update && sudo apt install ffmpeg
```

To check that FFmpeg is installed:

```console
$ ffmpeg -version
ffmpeg version 4.4 Copyright (c) 2000-2021 the FFmpeg developers
...
```

Note, for Linux, you may need to install additional system packages to get `zamba` to work. For example, on Ubuntu, you may need to install `build-essential` to have compilers.

FFmpeg 4, build-essential, and some other packages that include more codecs to support additional videos and some other utilities can be installed with:

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

### 3. Install `zamba`

For image workflows:

```console
$ uv pip install zamba[image]
```

For video workflows:

```console
$ uv pip install zamba[video]
```

For both image and video workflows:

```console
$ uv pip install zamba[video,image]
```

## macOS

`zamba` has been tested on macOS High Sierra.

### 1. Install Python

Install [Python](https://www.python.org/) >= 3.11. With [uv](https://docs.astral.sh/uv/) you can install and pin a suitable version with `uv python install 3.11`.

### 2. Install FFmpeg for video workflows

[FFmpeg](https://ffmpeg.org/ffmpeg.html) is required for video workflows. Use FFmpeg > 4.3 and < 5. Install [Homebrew](https://brew.sh/), then run:

```console
$ brew install ffmpeg@4
$ echo 'export PATH="/opt/homebrew/opt/ffmpeg@4/bin:$PATH"' >> ~/.zshrc
```

Open a new terminal and check that FFmpeg is installed:

```console
$ ffmpeg -version
ffmpeg version 4.4 Copyright (c) 2000-2021 the FFmpeg developers
...
```

### 3. Install `zamba`

For image workflows:

```console
$ uv pip install zamba[image]
```

For video workflows:

```console
$ uv pip install zamba[video]
```

For both image and video workflows:

```console
$ uv pip install zamba[video,image]
```

## Windows

`zamba` works on Windows. The **image** extra installs without any additional tools. The **video** extra requires a C++ compiler because some dependencies have native extensions, and FFmpeg for loading videos.

### Image workflows

Install [Python](https://www.python.org/) >= 3.11. With [uv](https://docs.astral.sh/uv/) you can install and pin a suitable version with `uv python install 3.11`. Then run:

```console
$ uv pip install zamba[image]
```

### Video workflows

1. Install [Python](https://www.python.org/) >= 3.11. With [uv](https://docs.astral.sh/uv/) you can install and pin a suitable version with `uv python install 3.11`.

2. Install Visual Studio Build Tools. Download the [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/) installer and select the **"Desktop development with C++"** workload. This provides the `cl.exe` compiler needed to build native extensions.

3. Install FFmpeg > 4.3 and < 5. Download a 4.x build from [ffmpeg.org](https://www.ffmpeg.org/download.html) and add the `bin` folder to your system PATH.

4. Install zamba:

```console
$ uv pip install zamba[video]
```

For both image and video workflows:

```console
$ uv pip install zamba[video,image]
```

Alternatively, you can use [Docker](https://www.docker.com/products/docker-desktop/) or [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) to run `zamba` in a Linux environment on Windows.

## pip and source installs

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

## Using GPU(s)

`zamba` is much faster on a machine with a graphics processing unit (GPU), but has also been developed and tested for machines without GPU(s).

To use a GPU, you must be using an
[NVIDIA GPU](https://www.nvidia.com/Download/index.aspx?lang=en-us),
have installed and configured [CUDA](https://developer.nvidia.com/cuda-downloads),
and have installed and configured [CuDNN](https://developer.nvidia.com/cudnn) per
their specifications.

If you are using `conda`, these dependencies can be installed through the [`cudatoolkit` package](https://anaconda.org/anaconda/cudatoolkit). With `uv`, install a CUDA-enabled PyTorch build into your environment by pointing at the matching PyTorch index, for example `uv pip install torch --index-url https://download.pytorch.org/whl/cu121` (replace `cu121` with the CUDA version you have installed). Either way, make sure the PyTorch build matches your installed CUDA version. See the [PyTorch installation docs](https://pytorch.org/get-started/locally/) for the easiest way to install the right version on your system.
