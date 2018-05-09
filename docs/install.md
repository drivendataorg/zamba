# Installing zamba


Zamba has been developed and tested on macOS and Ubuntu linux for both cpu and
gpu configurations.

When a user installs `zamba` that **user must specify to install the gpu or cpu
 version**. If the user fails to make this specification, no version of
[`tensorflow`](https://www.tensorflow.org/) will be installed, thus everything
will fail.

## Prerequisites

 - [Python](https://www.python.org/) 3.6, we recommend [Python installation using Anaconda](https://www.anaconda.com/download/)
 - [ffmpeg](https://www.ffmpeg.org/download.html), codecs for handling loading video files


## Hardware Specifications

`zamba` is much faster on a machine with a graphics processing unit (gpu), but
 it has been developed and tested for machine with and without gpu(s).


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


## Operating System Specifications

In addition to [installing the correct version](#hardware) of `zamba`, a user
may need to take additional steps before `zamba` works with their operating system.


### macOS

`zamba` has been tested on macOS High Sierra.


### Linux

`zamba` has been tested on [Ubuntu](https://www.ubuntu.com/) versions 16 and 17.


