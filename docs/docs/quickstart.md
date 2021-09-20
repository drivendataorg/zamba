# Quickstart

This section assumes you have successfully installed `zamba` and want to get
right to either making species predictions for some videos, or training a model! 

## How do I input my videos to `zamba`?

You can input the path to a directory of videos to classify. 

* **The folder must contain only valid video files**, since `zamba` will try to load all of the files in the directory. 
* `zamba` supports the same video formats as FFMPEG, [which are listed here](https://www.ffmpeg.org/general.html#Supported-File-Formats_002c-Codecs-or-Features).
* `zamba` will only generate predictions for the videos in the top level of a directory (`zamba` does not currently extract videos from nested directories).

For example, say that we have a directory of videos called `vids_to_classify` that we want to generate predictions for using `zamba`. Let's list the videos:

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

In this example, the videos have meaningful names so that we can easily
compare the predictions made by `zamba`. In practice, your videos will
probably be named something much less useful!

<h2 id='using-cli'></h2>

## Using the command line interface

All of the commands here should be run at the command line. On
macOS, this can be done in the terminal (⌘+space, "Terminal"). On Windows, this can be done in a command prompt, or if you installed Anaconda an anaconda prompt (Start > Anaconda3 > Anaconda Prompt).

<h3 id='basic-cli-prediction'></h3>

### Generating predictions
To generate and save predictions for your videos using the default settings, run:

```console
$ zamba predict --data-dir vids_to_classify/
```

`zamba` will output a `.csv` file with rows labeled by each video filename and columns for each class (ie. species). The default prediction will store all class probabilities, so that cell (i,j) can be interpreted as *the probability that animal j is present in video i.* 
Predictions will be saved to `{model name}_{current timestamp}_preds.csv`.
For example, running `zamba predict` on 9/15/2021 with the `time_distributed` model (the default) will save out predictions to `time_distributed_2021-09-15_preds.csv`. 

Adding the argument `--output-class-names` will simplify the predictions to return only the *most likely* animal in each video:

```console
$ zamba predict --data-dir vids_to_classify/ --output-class-names
$ cat time_distributed_2021-09-15_preds.csv
vids/blank.mp4,blank
vids/chimp.mp4,chimpanzee_bonobo
vids/eleph.mp4,elephant
vids/leopard.mp4,leopard
```

### Training a model

To train a model based on the videos in `vids_to_classify` and the labels in `example_labels.csv`:

```console
$ zamba train --data-dir vids/ --labels example_labels.csv
```

<!-- TODO: where will the model be saved?><!-->

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

For more details, see the page on the [CLI](cli.md).

## Using the Python module

Any functionality available in the command line interface is also accessible in the Python package.

<h3 id='basic-python-prediction'></h3>

### Generating predictions

To generate predictions for the same folder, `vids_to_classify`, the minimum you have to run is:
```python
from zamba.models.model_manager import predict_model
from zamba.models.config import PredictConfig
from zamba_algorithms.data.video import VideoLoaderConfig

predict_config = PredictConfig(data_directory='vids_to_classify/')
video_loader_config = VideoLoaderConfig()

predict_model(predict_config=predict_config, video_loader_config=video_loader_config)
```

To specify various parameters when running `predict_model`, the first step is to initiate `PredictConfig` and `VideoLoaderConfig` with any specifications for prediction and video loading respectively. The above uses all of the default settings.

The output will be the same as in the CLI - a `.csv` with rows labeled by each video filename and columns for each class, saved to `{model name}_{current timestamp}_preds.csv` (see the [CLI section](quickstart.md#basic-cli-prediction) for more details).

Just like in the CLI, setting `output_class_names` to `True` will simplify the predictions to return only the *most likely* animal in each video. Another option is to set `proba_threshold`. For each video, this will return whether each class is either present (`1`) or not present (`0`) based on whether the probability is above a certain threshold:

```python
predict_config = PredictConfig(data_directory='vids_to_classify/', proba_threshold=0.5)
video_loader_config = VideoLoaderConfig()

predict_model(predict_config=predict_config, video_loader_config=video_loader_config)
predictions = pd.read_csv('time_distributed_2021-09-15_preds.csv')
predictions
```

| filepath                     | aardvark | antelope_duiker | badger | bat | bird | blank | cattle | cheetah | chimpanzee_bonobo | civet_genet | elephant | equid | forest_buffalo | fox | giraffe | gorilla | hare_rabbit | hippopotamus | hog | human | hyena | large_flightless_bird | leopard | lion | mongoose | monkey_prosimian | pangolin | porcupine | reptile | rodent | small_cat | wild_dog_jackal |
| ---------------------------- | -------- | --------------- | ------ | --- | ---- | ----- | ------ | ------- | ----------------- | ----------- | -------- | ----- | -------------- | --- | ------- | ------- | ----------- | ------------ | --- | ----- | ----- | --------------------- | ------- | ---- | -------- | ---------------- | -------- | --------- | ------- | ------ | --------- | --------------- |
| vids_to_classify/blank.MP4   | 0        | 0               | 0      | 0   | 0    | 1     | 0      | 0       | 0                 | 0           | 0        | 0     | 0              | 0   | 0       | 0       | 0           | 0            | 0   | 0     | 0     | 0                     | 0       | 0    | 0        | 0                | 0        | 0         | 0       | 0      | 0         | 0               |
| vids_to_classify/chimp.MP4   | 0        | 0               | 0      | 0   | 0    | 0     | 0      | 0       | 1                 | 0           | 0        | 0     | 0              | 0   | 0       | 0       | 0           | 0            | 0   | 0     | 0     | 0                     | 0       | 0    | 0        | 0                | 0        | 0         | 0       | 0      | 0         | 0               |
| vids_to_classify/eleph.MP4   | 0        | 0               | 0      | 0   | 0    | 0     | 0      | 0       | 0                 | 0           | 1        | 0     | 0              | 0   | 0       | 0       | 0           | 0            | 0   | 0     | 0     | 0                     | 0       | 0    | 0        | 0                | 0        | 0         | 0       | 0      | 0         | 0               |
| vids_to_classify/leopard.MP4 | 0        | 0               | 0      | 0   | 0    | 0     | 0      | 0       | 0                 | 0           | 0        | 0     | 0              | 0   | 0       | 0       | 0           | 0            | 0   | 0     | 0     | 0                     | 1       | 0    | 0        | 0                | 0        | 0         | 0       | 0      | 0         | 0               |

### Training

To train a model based on the videos in `vids_to_classify` and the labels in `example_labels.csv`:

```python
from zamba.models.model_manager import train_model
from zamba.models.config import TrainConfig, VideoLoaderConfig

train_config = TrainConfig(labels='example_labels.csv', data_directory='vids_to_classify/')
video_loader_config = VideoLoaderConfig()

train_model(train_config=train_config, video_loader_config=video_loader_config)
```

<!-- TODO: where will the model be saved?><!-->

### Getting help

Once you have import the functions and classes in Python, you can see the details of each with `help`:

```python
>> help(predict_model)

predict_model(predict_config: zamba_algorithms.models.config.PredictConfig, video_loader_config: zamba_algorithms.data.video.VideoLoaderConfig, return_preds: bool = False)
    Predicts from a model and writes out predictions to a csv.
    
    Args:
        predict_config (PredictConfig): Pydantic config for performing inference.
        video_loader_config (VideoLoaderConfig): Pydantic config for preprocessing videos.
        return_preds (bool): If True, return dataframe containing predictions (in addition
            to whatever save behavior is specified by predict_config.save). Defaults to False.
```

To see the options for `PredictConfig` and for `VideoLoaderConfig`, you can run:

```python
help(PredictConfig)
help(VideoLoaderConfig)
```

Advanced options are explained in greater detail on the [All Configuration Options](configurations.md) page. 


## Downloading model weights

**`zamba` needs to download the "weights" files for the neural networks that it uses to make predictions. On first run it will download ~200-500 MB of files with these weights depending which model you choose.** 
Once a model's weights are downloaded, the tool will use the local version and will not need to perform this download again. If you are not in the US, we recommend running the above command with the additional flag either `--weight_download_region eu` or `--weight_download_region asia` depending on your location. The closer you are to the server the faster the downloads will be.

## Next Steps

This is just the tip of the iceberg! `zamba` has many more options for the command line
and the Python module. See the docs for more information.
