# Installing zamba


Zamba has been developed and tested on macOS and Ubuntu Linux for both CPU and
GPU configurations.

## Prerequisites

 - Python 3.7 or 3.8
 - ffmpeg

### [Python](https://www.python.org/) 3.7 or 3.8

We recommend [Python installation using Anaconda](https://www.anaconda.com/download/) for all platforms, for more information about how to install Anaconda, here are some useful YouTube videos of installation on different platforms:

 - [Anaconda download link](https://www.anaconda.com/download/)

 - [Windows install video](https://www.youtube.com/watch?v=0OXBHvFeH_U)
 - [macOS installation video](https://www.youtube.com/watch?v=nVlrpNf3EdM)


### FFMPEG version 4.3

FFMPEG is an open source library for loading videos of different codecs, and using ffmpeg means that `zamba` can be flexible in terms of the video formats we support. FFMPEG can be installed on all different platforms, but requires some additional configuration depending on the platform. Here are some videos/instructions walking through installation of FFMPEG on different platforms:

 - [ffmpeg download link](https://www.ffmpeg.org/download.html)

 - [Install on Ubuntu or Linux](https://www.tecmint.com/install-ffmpeg-in-linux/). In the command line, enter `sudo apt update` and then `sudo apt install ffmpeg`. Then to double check your installed version, run `ffmpeg -version`
 - [MacOS install video](https://www.youtube.com/watch?v=8nbuqYw2OCw&t=5s)



## Install Hardware Specific Version of Zamba

`zamba` is much faster on a machine with a graphics processing unit (GPU), but
 it has been developed and tested for machine with and without GPU(s).

If you are using Anaconda, run these commands from an Anaconda prompt (Start > Anaconda3 > Anaconda Prompt).

To install for development:

```console
$ pip install zamba
```

### GPU

To install for development with **Tensorflow for GPU**

```console
$ pip install zamba[gpu]
```

To use a GPU, you must be using an
[NVIDIA GPU](https://www.nvidia.com/Download/index.aspx?lang=en-us),
[installed and configured CUDA](https://developer.nvidia.com/cuda-downloads),
and [installed and configured CuDNN](https://developer.nvidia.com/cudnn) per
their specifications. Once this is done, you can select to install the version
 of `zamba` that is compiled for GPU.


### CPU

To install for development with **Tensorflow for CPU**

```console
$ pip install zamba[cpu]
```


## Operating Systems that have been tested


### macOS

`zamba` has been tested on macOS High Sierra.

### Linux

`zamba` has been tested on [Ubuntu](https://www.ubuntu.com/) versions 16 and 17.

### Windows

`zamba` has been tested on Windows 10.


