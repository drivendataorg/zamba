# Quickstart

This section assumes you have successfully installed `zamba` and want to get
right to making species predictions for some videos!

## Input videos

For example, we might have the directory `vids_to_classify` with camera trap videos in it:

```console
$ ls vids_to_classify/
blank1.mp4
blank2.mp4
eleph.mp4
small-cat.mp4
ungulate.mp4
```

Here are some screenshots from those videos:

<table class="table">
  <tbody>
    <tr>
      <td>
        <img src="https://s3.amazonaws.com/drivendata-public-assets/zamba-ele-sm.png" alt="elephant"/>
      </td>
      <td>
        <img src="https://s3.amazonaws.com/drivendata-public-assets/zamba-ung-sm.png" alt="ungulate"/>
      </td>
    </tr>
    <tr>
      <td>
        <img src="https://s3.amazonaws.com/drivendata-public-assets/zamba-blank-sm.png" alt="blank"/>
      </td>
      <td>
        <img src="https://s3.amazonaws.com/drivendata-public-assets/zamba-cat-sm.png" alt="cat"/>
      </td>
    </tr>
  </tbody>
</table>

In this example, the videos have meaningful names so that we can easily
compare the predictions made by `zamba`. In practice, your videos will
probably be named something much less useful!

### Predict Using Concise Output Format

If you just want to know the most likely animal in each video, the
`--output_class_names` flag is useful. In this case, the final output as well as the resulting `output.csv`
are simplified to show the _most probable_ animal in each video:

```console
$ zamba predict vids_to_classify/ --output_class_names
...
blank2.mp4                blank
eleph.mp4              elephant
blank1.mp4                blank
ungulate.mp4     large ungulate
small-cat.mp4         small cat
```

**NOTE: `zamba` needs to download the "weights" files for the neural networks
that it uses to make predictions. On first run it will download ~1GB of files
with these weights.** Once these are downloaded, the tool will use the local
versions and will not need to perform this download again.

## Getting Help from the Command Line

Once zamba is installed, you can see the available commands with `zamba`:

```console
$ zamba

Usage: zamba [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  predict  Identify species in a video.
  train    [NOT IMPLEMENTED] Retrain network from...
  tune     [NOT IMPLEMENTED] Update network with new...
```

To see more detailed information about a command as well as the
options available to pass to it, use the `--help` flag. For example, get more
information about the `predict` command and its options:

```console
$ zamba predict --help

Usage: zamba predict [OPTIONS] [DATA_PATH] [PRED_PATH]

  Identify species in a video.

  This is a command line interface for prediction on camera trap footage.
  Given a path to camera trap footage, the predict function use a deep
  learning model to predict the presence or absense of a variety of species
  of common interest to wildlife researchers working with camera trap data.

Options:
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

## Next Steps

This is just the tip of the iceberg. `zamba` has more options for command line
use, and can alsoe be used as a Python module, e.g., `import zamba`! See the
docs for more information.
