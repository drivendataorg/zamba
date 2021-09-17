# Quickstart

This section assumes you have successfully installed `zamba` and want to get
right to making species predictions for some videos! 

## Input videos

### What videos can I use?

`zamba` supports the same video formats as FFMPEG, [which are listed here](https://www.ffmpeg.org/general.html#Supported-File-Formats_002c-Codecs-or-Features). The built-in models were trained primarily on `.mp4` and `.avi` videos that were each between 15 seconds and 1 minute long.

### How do I input my videos to `zamba`?

You can input the path to a directory of videos to classify. For example,
suppose you have `zamba` installed, your command line is open, and you have a
directory of videos, `vids_to_classify`, that you want to classify using
`zamba`.

List the videos:

```console
$ ls vids_to_classify/
blank.mp4
chimp.mp4
eleph.mp4
leopard.mp4
```

Here are some screenshots from those videos:
<table class="table">
  <tbody>
    <tr>
      <td style="text-align:center">blank.mp4<br/>
        <img src="https://s3.amazonaws.com/drivendata-public-assets/zamba-2-blank-sm.jpg" alt="Blank frame seen from a camera trap" style="width:400px;"/>
      </td>
      <td style="text-align:center">chimp.mp4<br/>
        <img src="https://s3.amazonaws.com/drivendata-public-assets/zamba-2-chimp-sm.jpg" alt="Leopard seen from a camera trap" style="width:400px;"/>
      </td>
    </tr>
    <tr>
      <td style="text-align:center">eleph.mp4<br/>
        <img src="https://s3.amazonaws.com/drivendata-public-assets/zamba-2-eleph-sm.jpg" alt="Elephant seen from a camera trap" style="width:400px">
      </td>
      <td style="text-align:center">leopard.mp4<br/>
        <img src="https://s3.amazonaws.com/drivendata-public-assets/zamba-2-leopard-sm.jpg" alt="cat" style="width:400px;"/>
      </td>
    </tr>
  </tbody>
</table>

**The folder must contain only valid video files** since zamba will try to load all of the files in the directory.

In this example, the videos have meaningful names so that we can easily
compare the predictions made by `zamba`. In practice, your videos will
probably be named something much less useful!

You can also input a CSV of metadata that includes the path to each video
for classification. For more details on this method, see the advanced options section.
<!-- TODO: add link><!--> 

## Using the command line interface

All of the commands here should be run at the command line. On
macOS, this can be done in the terminal (⌘+space, "Terminal"). On Windows, this can be done in a command prompt, or if you installed Anaconda an anaconda prompt (Start > Anaconda3 > Anaconda Prompt).

To generate and save predictions for your videos using the default settings, run:

```console
$ zamba predict --data-dir vids_to_classify/
```

`zamba` will output a `.csv` file with rows labeled by each video filename and columns for each class (ie. species). The default prediction will store all class probabilities, so that cell (i,j) can be interpreted as *the probability that animal j is present in video i.* 
Predictions will be saved to `{model name}_{current timestamp}_preds.csv`.
For example, running `zamba predict` on 9/15/2021 with the `time_distributed` model (the default) will save out predictions to `time_distributed_2021-09-15_preds.csv`. 

`zamba` will only generate predictions for the videos in the top level of the `vids_to_classify` directory (`zamba` does not currently extract videos from nested directories).

Adding the argument `--output-class-names` will simplify the predictions to return only the *most likely* animal in each video:

```console
$ zamba predict --data-dir vids_to_classify/ --output-class-names
$ cat time_distributed_2021-09-15_preds.csv
vids/blank.mp4,blank
vids/chimp.mp4,chimpanzee_bonobo
vids/eleph.mp4,elephant
vids/leopard.mp4,leopard
```

### Getting help

Once zamba is installed, you can see available commands with `zamba --help`:

```console
$ zamba --help
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
options available to pass to it, use the `--help` flag. For example, to get more
information about the `train` command and its options:

```console
$ zamba train --help
```

## Using the Python module

The main API for `zamba` is the `ModelManager` class. The `ModelManager` is used behind the scenes by `zamba`’s command line interface to handle preprocessing the files, loading the videos, serving them to the model, and saving predictions. Therefore any functionality available to the command line interface is accessible via the `ModelManager` class.

To generate predictions using the same directory, `vids_to_classify`:
<!-- TODO: does it still default to time_distributed or does a model name have to be passed?><!-->
<!-- TODO: placeholder, come  back to this when clearer how python module works><!-->
```python
from zamba.models.manager import ModelManager
manager = ModelManager()
```

Just like in the command line, the default output has a row for each filename and a column for each possible class. 
We can generate the simplified most probable class by adjusting the model configuration:
<!-- TODO: add><!-->

<!-- TODO: add how to specify weight download region><!-->

## Downloading model weights

**`zamba` needs to download the "weights" files for the neural networks that it uses to make predictions. On first run it will download ~200-500 MB of files with these weights depending which model you choose.** 
Once a model's weights are downloaded, the tool will use the local version and will not need to perform this download again. If you are not in the US, we recommend running the above command with the additional flag either `--weight_download_region eu` or `--weight_download_region asia` depending on your location. The closer you are to the server the faster the downloads will be.

## Next Steps

This is just the tip of the iceberg! `zamba` has many more options for the command line
and the Python module. See the docs for more information.
