# Installing `zamba`


Zamba has been developed and tested on macOS and Ubuntu linux for both cpu and
gpu configurations. Currently testing on Windows.

When a user installs `zamba` that **user must specify to install the gpu or cpu
 version**. If the user fails to make this specification, no version of
[`tensorflow`](https://www.tensorflow.org/) will be installed, thus everything
will fail.

## Prerequisites

 - [Python](https://www.python.org/) 3.6

## Hardware Specifications
<a name="hardware"></a>

`zamba` is much faster on a machine with a graphics processing unit (gpu), but
 it has been developed and tested for machine with and without gpu(s).


### gpu

To install for development with **Tensorflow for gpu**
```
$ git clone https://github.com/drivendataorg/zamba.git
$ cd zamba
$ pip install --editable .[gpu]
```

To use a GPU, you must be using an
[nvidia gpu](https://www.nvidia.com/Download/index.aspx?lang=en-us),
[installed and configured CUDA](https://developer.nvidia.com/cuda-downloads),
and [installed and configured CuDNN](https://developer.nvidia.com/cudnn) per
their specifications. Once this is done, you can select to install the version
 of `zamba` that uses `tensorflow` compiled for GPU.

### cpu

To install for development with **Tensorflow for cpu**
```
$ git clone https://github.com/drivendataorg/zamba.git
$ cd zamba
$ pip install --editable .[cpu]
```

## Operating System Specifications

In addition to [installing the correct version](#hardware) of `zamba`, a user
may need
to
take additional steps before `zamba` works with their operating system.

### macOS

For users of macOS, the initial install of
[`lightgbm`](https://github.com/Microsoft/LightGBM/tree/master/python-package)
that is handled by `zamba` will not produce any errors, but as of this writing
it will not successfully import either, causing `zamba` to fail.

To overcome this issue
 - ensure that that the Mac package manager [homebrew](https://brew.sh/) is
 installed
 - follow [LightGBM's instructions](http://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#macos) for a successful install of LightGBM

### Linux

`zamba` has been tested on [Ubuntu](https://www.ubuntu.com/) versions 16 and 17.

### Windows

NOT IMPLEMENTED