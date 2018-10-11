# Installing zamba


Zamba has been developed and tested on macOS and Ubuntu linux for both cpu and
gpu configurations.

When a user installs `zamba` that **user must specify to install the gpu or cpu
 version**. If the user fails to make this specification, no version of
[`tensorflow`](https://www.tensorflow.org/) will be installed, thus everything
will fail.

## Prerequisites

 - Python 3.6
 - ffmpeg
 - xgboost

### [Python](https://www.python.org/) 3.6

We recommend [Python installation using Anaconda](https://www.anaconda.com/download/) for all platforms, for more information about how to install Anaconda, here are some useful Youtube videos of installation on different platforms:

 - [Anaconda download link](https://www.anaconda.com/download/)

 - [Windows install video](https://www.youtube.com/watch?v=0OXBHvFeH_U)
 - [macOS installation video](https://www.youtube.com/watch?v=nVlrpNf3EdM)


### FFMPEG version 4.0

FFMPEG is an opensource library for loading videos of different codecs, and using ffmpeg means that `zamba` can be flexible in terms of the video formats we support. FFMPEG can be installed on all different platforms, but requires some additional configuration depending on the platform. Here are some videos/instructions walking through installation of FFMPEG on different platforms:

 - [ffmpeg download link](https://www.ffmpeg.org/download.html)

 - [Windows install video](https://www.youtube.com/watch?v=pHR3ttH5t-w)
 - [Windows install instructions with screenshots](https://video.stackexchange.com/questions/20495/how-do-i-set-up-and-use-ffmpeg-in-windows/20496#20496)

 - [MacOS install video](https://www.youtube.com/watch?v=8nbuqYw2OCw&t=5s)


### XGBOOST 0.71

XGBoost is a library for gradient boosting trees, which is oftenused in ensembled machine learning architectures like `zamba`. XGBoost may require extra steps on your platform. See below:

#### XGBoost on Windows

 - [Download precompiled xgboost](https://www.lfd.uci.edu/~gohlke/pythonlibs/#xgboost) for Python 3.6 and either 32 or 64 bit [depending on your version of windows](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64)

 - Open a command prompt. If you installed Anaconda, you will want to use an Anaconda command prompt: **Start > Anaconda3 > Anaconda Prompt**
 - `cd Downloads` - change directories to your download folder where the precompiled binary is
 - `pip install xgboost-0.71-cp36-cp36m-win_amd64.whl` - your filename may be different based on the version of Windows

#### XGBoost on Linux and macOS

XGBoost should install with `zamba` automatically. If you see a problem with xgboost when installing zamba, the easiest fix is to run `conda install xgboost==0.71 -c conda-forge` in an Anaconda prompt.


## Install Hardware Specific Version of Zamba

`zamba` is much faster on a machine with a graphics processing unit (gpu), but
 it has been developed and tested for machine with and without gpu(s).

If you are using Anaconda, run these commands from an Anaconda prompt (Start > Anaconda3 > Anaconda Prompt).

### gpu

To install for development with **Tensorflow for gpu**

```console
$ pip install zamba[gpu]
```

To use a GPU, you must be using an
[nvidia gpu](https://www.nvidia.com/Download/index.aspx?lang=en-us),
[installed and configured CUDA](https://developer.nvidia.com/cuda-downloads),
and [installed and configured CuDNN](https://developer.nvidia.com/cudnn) per
their specifications. Once this is done, you can select to install the version
 of `zamba` that uses `tensorflow` compiled for GPU.


### cpu

To install for development with **Tensorflow for cpu**

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


