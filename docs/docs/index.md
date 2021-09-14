Welcome to zamba's documentation!
=================================


<div class="embed-responsive embed-responsive-16by9" width=500>
    <iframe width=600 height=340 class="embed-responsive-item" src="https://s3.amazonaws.com/drivendata-public-assets/monkey-vid.mp4" frameborder="0" allowfullscreen=""></iframe>
</div>

*Zamba means "forest" in the Lingala language.*

Zamba is a command-line tool built in Python to automatically identify the
species seen in camera trap videos from sites in central and west Africa. Using the
combined input of various deep learning models, the tool makes predictions
for 31 common species in these videos (as well as blank, or, "no species
present").

**New in Zamba v2:** Zamba now has an additional model trained on 11 common European species. <!--TODO: add more details about where from><!-->

# Quickstart

This section assumes you have successfully installed `zamba` and want to get
right to making species predictions for some videos! All of the commands here should be run at the command line. On
macOS, this can be done in the terminal (⌘+space, "Terminal"). On Windows, this can be done in a command prompt, or if you installed Anaconda an anaconda prompt (Start > Anaconda3 > Anaconda Prompt).

## Input videos

For example, we might have the directory `vids_to_classify` with camera trap videos in it:

```console
$ ls vids_to_classify/
blank1.mp4
blank2.mp4
eleph.mp4
small-cat.mp4
chimp.mp4
```
<!-- TODO: update ungulate to species in the new labels><!-->
<!-- TODO: make order or vids above match order of pics and order of labels later><!-->

Here are some screenshots from those videos:

<table class="table">
  <tbody>
    <tr>
      <td>eleph.mp4
        <img src="https://s3.amazonaws.com/drivendata-public-assets/zamba-ele-sm.png" alt="Elephant seen from a camera trap" style="width:400px;height:225;">
      </td>
      <td>
        <img src="https://s3.amazonaws.com/drivendata-public-assets/zamba-chimp-sm.png" alt="Leopard seen from a camera trap" style="width:400px;height:225px;"/>
      </td>
    </tr>
    <tr>
      <td>
        <img src="https://s3.amazonaws.com/drivendata-public-assets/zamba-blank-sm.png" alt="Blank frame seen from a camera trap" style="width:400px;height:225px;"/>
      </td>
      <td>
        <img src="https://s3.amazonaws.com/drivendata-public-assets/zamba-cat-sm.png" alt="cat" style="width:400px;height:225px;"/>
      </td>
    </tr>
  </tbody>
</table>

In this example, the videos have meaningful names so that we can easily
compare the predictions made by `zamba`. In practice, your videos will
probably be named something much less useful!

### Predict Using Concise Output Format

If you just want to know the most likely animal in each video, the
`--output-class-names` flag is useful. In this case, the final output as well as the resulting `output.csv`
are simplified to show the _most probable_ animal in each video:

```console
$ zamba predict --data-dir vids_to_classify/ --output_class_names
...
blank2.mp4                blank
eleph.mp4              elephant
blank1.mp4                blank
ungulate.mp4     large ungulate
small-cat.mp4         small cat
```

**NOTE: `zamba` needs to download the "weights" files for the neural networks
that it uses to make predictions. On first run it will download ~1GB <!-- TODO: check size><!--> of files
with these weights.** Once these are downloaded, the tool will use the local
versions and will not need to perform this download again. If you are not in the US, we recommend
running the above command with the additional flag either `--weight_download_region eu` or
`--weight_download_region asia` depending on your location. The closer you are to the server
the faster the downloads will be.

## Getting Help from the Command Line

Once zamba is installed, you can see the available commands with `zamba`:

```console
$ zamba
Usage: zamba [OPTIONS] COMMAND [ARGS]...

Options:
  --install-completion  Install completion for the current shell.
  --show-completion     Show completion for the current shell, to copy it or
                        customize the installation.

  --help                Show this message and exit.

Commands:
  predict  Identify species in a video.
  train    Train a model using the provided data, labels, and model name.
```

To see more detailed information about a command as well as the
options available to pass to it, use the `--help` flag. For example, get more
information about the `train` command and its options:

```console
$ zamba train --help
Usage: zamba train [OPTIONS]

  Train a model using the provided data, labels, and model name.

  If an argument is specified in both the command line and in a yaml file,
  the command line input will take precedence.

Options:
  --data-dir PATH                 Path to folder containing videos.
  --labels PATH                   Path to csv containing video labels.
  --model [time_distributed|slowfast]
                                  Model class to train.  [default:
                                  time_distributed]

  --config PATH                   Specify options using yaml configuration
                                  file instead of through command line
                                  options.

  --batch-size INTEGER            Batch size to use for training.
  --gpus INTEGER                  Number of GPUs to use for training. If not
                                  specifiied, will use all GPUs found on
                                  machine.

  --dry-run / --no-dry-run        Runs one batch of train and validation to
                                  check for bugs.

  -y, --yes                       Skip confirmation of configuration and
                                  proceed right to training.  [default: False]

  --help                          Show this message and exit.
```

## Next Steps

This is just the tip of the iceberg. `zamba` has more options for command line
use, and can alsoe be used as a Python module, e.g., `import zamba`! See the
docs for more information.
