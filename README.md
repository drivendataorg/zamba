# zamba - a command line interface for species classification

[ ![Codeship Status for drivendataorg/chimps-tool](https://app.codeship.com/projects/03e3a040-0b6d-0136-afe4-3aeedc3a22e1/status?branch=master)](https://app.codeship.com/projects/281856)


### [HOMEPAGE](http://zamba.drivendata.org/)

### [DOCUMENTATION](http://zamba.drivendata.org/docs/)

_Zamba means "forest" in the Lingala language._

Zamba is a command-line tool built in Python to automatically identify the species seen in camera trap videos from sites in central Africa. The tool makes predictions for 24 common species in these videos. For more information, see the documentation.

The `zamba` command will be the entry point for users (see example usage below).


## Prerequisites (for more detail, see [the documentation](http://zamba.drivendata.org/docs/))

 - [Python](https://www.python.org/) 3.6
 - [ffmpeg](https://www.ffmpeg.org/download.html), codecs for handling the video loading


## Installing `zamba` (for more detail, see [the documentation](http://zamba.drivendata.org/docs/))

### GPU or CPU

`zamba` is significantly faster when using a machine with a GPU instead of just a CPU. To use a GPU, you must be using an [nvidia gpu](https://www.nvidia.com/Download/index.aspx?lang=en-us), [installed and configured CUDA](https://developer.nvidia.com/cuda-downloads), and [installed and configured CuDNN](https://developer.nvidia.com/cudnn) per their specifications. Once this is done, you can select to install the version of zamaba that uses `tensorflow` compiled for GPU.

When a user installs `zamba` that user must specify to install the GPU or CPU version. If the user fails to make this specification, **no version of tensorflow will be installed, thus everything will fail.**

To install with **tensorflow cpu** (you do not have a GPU)
```
$ pip install zamba[cpu]
```

To install with **tensorflow gpu**
```
$ pip install zamba[gpu]
```


## Example usage

Once zamba is installed, you can see the commands with `zamba`:

`zamba`

```
Usage: zamba [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  predict  Identify species in a video.
  train    [NOT IMPLEMENTED] Retrain network from...
  tune     [NOT IMPLEMENTED] Update network with new...
```

And you can see the options you can pass to the `predict` command with:

`zamba predict --help`

```
Usage: zamba predict [OPTIONS] [DATA_PATH] [PRED_PATH]

  Identify species in a video.

  This is a command line interface for prediction on camera trap footage.
  Given a path to camera trap footage, the predict function use a deep
  learning model to predict the presence or absense of a variety of species
  of common interest to wildlife researchers working with camera trap data.

Options:
  --tempdir PATH                 Path to temporary directory. If not
                                 specified, OS temporary directory is used.
  --proba_threshold FLOAT        Probability threshold for classification. if
                                 specified binary predictions are returned
                                 with 1 being greater than the threshold, 0
                                 being less than or equal to. If not
                                 specified, probabilities between 0 and 1 are
                                 returned.
  --output_class_names           If True, we just return a video and the name
                                 of the most likely class. If False, we return
                                 a probability or indicator (depending on
                                 --proba_threshold) for every possible class.
  --model_profile TEXT           Defaults to 'full' which is slow and
                                 accurate; can be 'fast' which is faster and
                                 less accurate.
  --weight_download_region TEXT  Defaults to 'us', can also be 'eu' or 'asia'.
                                 Region for server to download weights.
  --verbose                      Displays additional logging information
                                 during processing.
  --help                         Show this message and exit.
```

![demo](https://s3.amazonaws.com/drivendata-public-assets/zamba-demo.gif)


Once `zamba` is installed, you can execute it on any directory of video files. The tool does not recursively search directories, so all of the files must be at the top level of the directory. The algorithm will work the best with 15 second videos since that is what it is trained on, though it will sample frames from longer videos, which may be less reliable.

**NOTE: `zamba` needs to download the "weights" files for the neural networks that it uses to make predictions. On first run it will download ~1GB of files with these weights.** Once these are downloaded, the tool will use the local versions and will not need to perform this download again.

`zamba predict path/to/videos`

By default the output will be written to the file `output.csv` in the current directory. If the file exists, it will be overwritten.

## Running the `zamba` test suite

The included `Makefile` contains code that uses pytest to run all tests in `zamba/tests`.

The command is (from the project root),

```
$ make test
```

### Testing End-To-End Prediction With `test_cnnensemble.py`
The test `tests/test_cnnensemble.py` runs an end-to-end prediction with `CnnEnsemble.predict(data_dir)` using a video that automatically gets downloaded along with the `input` directory (this and all required directories are downloaded upon instantiation of `CnnEnsemble` if they are not already present in the project).

By default this test is skipped due to the `pytest` decorator

```
@pytest.mark.skip(reason="This test takes hours to run, makes network calls, and is really for local dev only.")
def test_predict():
    data_dir = Path(__file__).parent.parent / "models" / "cnnensemble" / "input" / "raw_test"

    manager = ModelManager('', model_class='cnnensemble', proba_threshold=0.5)
    manager.predict(data_dir, save=True)
```

It is reccomended that the **decorator be commented out in order to test end-to-end prediction locally**. However, this change should never be pushed, as the lightweight machines on codeship will not be happy, or able, to complete the end-to-end prediction.

To test end-to-end prediction using `make test` on a different set of videos, simply edit `data_dir`.


