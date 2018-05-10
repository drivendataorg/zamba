# zamba Command Line Interface

This section goes into a bit more detail concerning the available options for
the `zamba` command line interface tool. If you are new to `zamba` and just
want to classify some videos as soon as possible, see the [Quickstart]
(quickstart.html) guide.

## zamba's Optional Flags

### `zamba predict`

As discussed in the [Quickstart](quickstart.html), the `--help` flag provides
more information about options for `zamba`:

```
$ zamba predict --help
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

Let's go through these one by one.

#### --tempdir PATH

This option specifies the `PATH` to be used for temporary storage during
prediction. The model that is shipped with `zamba` is able to process within
memory (assuming reasonably large modern memory of ~16 GB). If a custom model
is being used, it may be necessary to point `zamba` to a mounted drive or some
 other large-capacity directory. By default this uses the operating system's temporary directory.

#### --proba_threshold FLOAT

For advanced uses, you may want the algorithm to be more or less sensitive to if a species is present. This parameter is a `FLOAT` number, e.g., `0.64` corresponding to the probability
threshold beyond which an animal is considered to be present in the video being
 analyzed.

By default no threshold is passed, `proba_threshold=None`. This will return a probability from 0-1 for each species that could occur in each video. If a threshold is passed,
then the final prediction value returned for each class is `probability >= proba_threshold`, so that all class values become `0` (`False`, the species does not appear) or `1` (`True`, the species does appear).

#### --output_class_names

Setting this option to `True` yields the most concise output `zamba` is capable
 of. The highest species probability in a video is taken to be the _only_
 species in that video, and the output returned is simply the video name and
  the name of the species (or `blank`) with the highest class probability. See
  the [Quickstart](quickstart.html) for example usage.

#### --model_profile TEXT

There are two versions of the algorithm that ship with zamba. If you pass `fast` there is a faster algorithm that can be less accurate that is used. If you pass `full` (the default) a slower algorithm that has 4 sub-models instead of 1 is used.

#### --weight_download_region TEXT

Because `zamba` needs to download pretrained weights for the neural network architecture, we make these weights available in different regions. 'us' is the default, but if you are not in the US you should use either `eu` for the European Union or `asia` for Asia Pacific to make sure that these download as quickly as possible for you.

#### --verbose BOOLEAN

This option currently controls only whether or not the command line shows additional information during processing.

### `zamba train`

#### NOT YET IMPLEMENTED

### `zamba tune`

#### NOT YET IMPLEMENTED
