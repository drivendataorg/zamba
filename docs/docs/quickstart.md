# Quickstart

This section assumes you have successfully installed `zamba` and want to get
right to either making species predictions for some videos, or training a model! 

All of the commands on this page should be run at the command line. On
macOS, this can be done in the terminal (⌘+space, "Terminal"). On Windows, this can be done in a command prompt, or if you installed Anaconda an anaconda prompt (Start > Anaconda3 > Anaconda Prompt).

## How do I input my videos to `zamba`?

You can input the path to a directory of videos to classify. 

* **The folder must contain only valid video files**, since `zamba` will try to load all of the files in the directory. 
* `zamba` supports the same video formats as FFMPEG, [which are listed here](https://www.ffmpeg.org/general.html#Supported-File-Formats_002c-Codecs-or-Features).
* `zamba` will only generate predictions for the videos in the top level of a directory (`zamba` does not currently extract videos from nested directories).

For example, say we have a directory of videos called `example_vids` that we want to generate predictions for using `zamba`. Let's list the videos:

```console
$ ls example_vids/
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

In this example, the videos have meaningful names so that we can easily
compare the predictions made by `zamba`. In practice, your videos will
probably be named something much less useful!

## Generating predictions

To generate and save predictions for your videos using the default settings, run:

```console
$ zamba predict --data-dir example_vids/
```

`zamba` will output a `.csv` file with rows labeled by each video filename and columns for each class (ie. species). The default prediction will store all class probabilities, so that cell (i,j) is *the probability that animal j is present in video i.* 
Predictions will be saved to `zamba_predictions.csv` in the current working directory by default. You can save out predictions under a different name or in a different folder using the `--save-path` argument.

Adding the argument `--output-class-names` will simplify the predictions to return only the *most likely* animal in each video:

```console
$ zamba predict --data-dir example_vids/ --output-class-names
$ cat zamba_predictions.csv
vids/blank.mp4,blank
vids/chimp.mp4,chimpanzee_bonobo
vids/eleph.mp4,elephant
vids/leopard.mp4,leopard
```

## Training a model

You can continue training one of the [models](models.md) that ships with `zamba` by either:

* Fine-tuning with additional labeled videos where the species are included in the list of [`zamba` class labels](models.md#species-classes)
* Retraining a model to predict a new set of species based on a set of labeled videos

In either case, the commands for training are the same. Say that we have labels for the videos in the `example_vids` folder saved in `example_labels.csv`. To train a model, run:

```console
$ zamba train --data-dir example_vids/ --labels example_labels.csv
```

The labels file must have columns for both filepath and label. Let's print the example labels:

```console
$ cat example_labels.csv
filepath,label
example_vids/eleph.MP4,elephant
example_vids/leopard.MP4,leopard
example_vids/blank.MP4,blank
example_vids/chimp.MP4,chimpanzee_bonobo
```

By default, the model will be saved to a folder in the current working directory called `zamba_<model_name>`. For example, a model finetuned from the provided `time_distributed` model will be saved in `zamba_time_distributed`. 

```console
$ zamba train --data-dir example_vids/ --labels example_labels.csv
$ ls zamba_time_distributed
configuration.yaml  events.out.tfevents.1632250686.ip-172-31-15-179.14229.0  hparams.yaml
```

## Downloading model weights

**`zamba` needs to download the "weights" files for the neural networks that it uses to make predictions. On first run it will download ~200-500 MB of files with these weights depending which model you choose.** 
Once a model's weights are downloaded, the tool will use the local version and will not need to perform this download again. If you are not in the US, we recommend running the above command with the additional flag either `--weight_download_region eu` or `--weight_download_region asia` depending on your location. The closer you are to the server the faster the downloads will be.

<a id='getting-help'></a>

## Getting help

Once zamba is installed, you can see more details of each function with `--help`. 

To get help with `zamba predict`:

```console
$ zamba predict --help

Usage: zamba predict [OPTIONS]

  Identify species in a video.

  This is a command line interface for prediction on camera trap footage.
  Given a path to camera trap footage, the predict function use a deep
  learning model to predict the presence or absense of a variety of species
  of common interest to wildlife researchers working with camera trap data.

  If an argument is specified in both the command line and in a yaml file,
  the command line input will take precedence.

Options:
  --data-dir PATH                 Path to folder containing videos.
  --filepaths PATH                Path to csv containing `filepath` column
                                  with videos.

  --model [time_distributed|slowfast|european]
                                  Model to use for inference. Model will be
                                  superseded by checkpoint if provided.
                                  [default: time_distributed]

  --checkpoint PATH               Model checkpoint path to use for inference.
                                  If provided, model is not required.

  --gpus INTEGER                  Number of GPUs to use for inference. If not
                                  specifiied, will use all GPUs found on
                                  machine.

  --batch-size INTEGER            Batch size to use for training.
  --save / --no-save              Whether to save out predictions to a csv
                                  file. If you want to specify the location of
                                  the csv, use save_path instead.

  --save-path PATH                Full path for prediction CSV file. Any
                                  needed parent directories will be created.

  --dry-run / --no-dry-run        Runs one batch of inference to check for
                                  bugs.

  --config PATH                   Specify options using yaml configuration
                                  file instead of through command line
                                  options.

  --proba-threshold FLOAT         Probability threshold for classification
                                  between 0 and 1. If specified binary
                                  predictions are returned with 1 being
                                  greater than the threshold, 0 being less
                                  than or equal to. If not specified,
                                  probabilities between 0 and 1 are returned.

  --output-class-names / --no-output-class-names
                                  If True, we just return a video and the name
                                  of the most likely class. If False, we
                                  return a probability or indicator (depending
                                  on --proba_threshold) for every possible
                                  class.

  --weight-download-region [us|eu|asia]
                                  Server region for downloading weights.
  --cache-dir PATH                Path to directory for model weights.
                                  Alternatively, specify with environment
                                  variable `ZAMBA_CACHE_DIR`. If not
                                  specified, user's cache directory is used.

  --skip-load-validation / --no-skip-load-validation
                                  Skip check that verifies all videos can be
                                  loaded prior to inference. Only use if
                                  you're very confident all your videos can be
                                  loaded.

  -y, --yes                       Skip confirmation of configuration and
                                  proceed right to prediction.  [default:
                                  False]

  --help                          Show this message and exit.
```

To get help with `zamba train`:

```console
$ zamba train --help

Usage: zamba train [OPTIONS]

  Train a model on your labeled data.

  If an argument is specified in both the command line and in a yaml file,
  the command line input will take precedence.

Options:
  --data-dir PATH                 Path to folder containing videos.
  --labels PATH                   Path to csv containing video labels.
  --model [time_distributed|slowfast|european]
                                  Model to train. Model will be superseded by
                                  checkpoint if provided.  [default:
                                  time_distributed]

  --checkpoint PATH               Model checkpoint path to use for training.
                                  If provided, model is not required.

  --config PATH                   Specify options using yaml configuration
                                  file instead of through command line
                                  options.

  --batch-size INTEGER            Batch size to use for training.
  --gpus INTEGER                  Number of GPUs to use for training. If not
                                  specifiied, will use all GPUs found on
                                  machine.

  --dry-run / --no-dry-run        Runs one batch of train and validation to
                                  check for bugs.

  --save-dir PATH                 Directory in which to save model checkpoint
                                  and configuration file. If not specified,
                                  will save to a folder called
                                  'zamba_{model_name}' in your working
                                  directory.

  --weight-download-region [us|eu|asia]
                                  Server region for downloading weights.
  --cache-dir PATH                Path to directory for model weights.
                                  Alternatively, specify with environment
                                  variable `ZAMBA_CACHE_DIR`. If not
                                  specified, user's cache directory is used.

  --skip-load-validation / --no-skip-load-validation
                                  Skip check that verifies all videos can be
                                  loaded prior to training. Only use if you're
                                  very confident all your videos can be
                                  loaded.

  -y, --yes                       Skip confirmation of configuration and
                                  proceed right to training.  [default: False]

  --help                          Show this message and exit.
```
