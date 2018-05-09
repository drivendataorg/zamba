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
  --tempdir PATH           Path to temporary directory. If not specified, OS
                           temporary directory is used.
  --proba_threshold FLOAT  Probability threshold for classification. if
                           specified binary predictions are returned with 1
                           being greater than the threshold, 0 being less than
                           or equal to. If not specified, probabilities
                           between 0 and 1 are returned.
  --output_class_names     If True, we just return a video and the name of the
                           most likely class. If False, we return a
                           probability or indicator (depending on
                           --proba_threshold) for every possible class.
  --model_path PATH        Path to model files to be loaded into model object.
  --model_class TEXT       Class of model, controls whether or not sample
                           model is used.
  --verbose BOOLEAN        Controls verbosity of the command line predict
                           function.
  --help                   Show this message and exit.
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

#### --model_path PATH

This option tells `zamba` where to look for the model. By default the
[algorithm shipped](algorithms.html) with `zamba` is used.

If an alternative model path is used, the model pointed to must be able to
inherit from the `zamba.models.model.Model` class in order to appropriately
work with the rest of the code.

#### --model_class TEXT

This flag is used primarily for development, in which a sample model is
sometimes used to test functionality. By default the `cnnensemble` [algorithm
class shipped](algorithms.html) with `zamba` is used.

#### --verbose BOOLEAN

This option currently controls only whether or not the command line shows the
 `data_path` (input) and `pred_path` (output) being used during prediction.

### `zamba train`

#### NOT IMPLEMENTED

### `zamba tune`

#### NOT IMPLEMENTED
