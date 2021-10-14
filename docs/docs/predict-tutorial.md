# User Tutorial: Classifying Unlabeled Videos

This section walks through how to classify videos using `zamba`. If you are new to `zamba` and just want to classify some videos as soon as possible, see the [Quickstart](quickstart.md) guide.

This tutorial goes over the steps for using `zamba` if:

* You already have `zamba` installed (for details see the [Installation](install.md) page)
* You have unlabeled videos that you want to generate labels for
* The possible class species labels for your videos are included in the list of possible [zamba labels](models.md#species-classes). If your species are not included in this list, you can [retrain a model](train-tutorial.md) using your own labeled data and then run inference.

## Basic usage: command line interface

Say that we want to classify the videos in a folder called `example_vids` as simply as possible using all of the default settings.

Minimum example for prediction in the command line:

```console
$ zamba predict --data-dir example_vids/
```

### Required arguments

To run `zamba predict` in the command line, you must specify either `--data-dir` or `--filepaths`. 

* **`--data-dir PATH`:** Path to the folder containing your videos.
* **`--filepaths PATH`:** Path to a CSV file with a column for the filepath to each video you want to classify. The CSV must have a column for `filepath`.

All other flags are optional. To choose a model, either `--model` or `--checkpoint` must be specified. Use `--model` to specify one of the three [pretrained models](models.md) that ship with `zamba`. Use `--checkpoint` to run inference with a locally saved model. `--model` defaults to `time_distributed`.

## Basic usage: Python package

Say that we want to classify the videos in a folder called `example_vids` as simply as possible using all of the default settings.

Minimum example for prediction using the Python package:

```python
from zamba.models.model_manager import predict_model
from zamba.models.config import PredictConfig

predict_config = PredictConfig(data_directory="example_vids/")
predict_model(predict_config=predict_config)
```

The only two arguments that can be passed to `predict_model` are `predict_config` and (optionally) `video_loader_config`. The first step is to instantiate [`PredictConfig`](configurations.md#prediction-arguments). Optionally, you can also specify video loading arguments by instantiating and passing in [`VideoLoaderConfig`](configurations.md#video-loading-arguments). 

### Required arguments

To run `predict_model` in Python, you must specify either `data_directory` or `filepaths` when `PredictConfig` is instantiated.

* **`data_directory (DirectoryPath)`:** Path to the folder containing your videos.

* **`filepaths (FilePath)`:** Path to a CSV file with a column for the filepath to each video you want to classify. The CSV must have a column for `filepath`.

For detailed explanations of all possible configuration arguments, see [All Optional Arguments](configurations.md).

## Default behavior

By default, the [`time_distributed`](models.md#time-distributed-model) model will be used. `zamba` will output a `.csv` file with rows labeled by each video filename and columns for each class (ie. species). The default prediction will store all class probabilities, so that cell (i,j) can be interpreted as *the probability that animal j is present in video i.* 

By default, predictions will be saved to `zamba_predictions.csv`. You can save predictions to a custom path using the `--save-path` argument.

```console
$ cat zamba_predictions.csv
filepath,aardvark,antelope_duiker,badger,bat,bird,blank,cattle,cheetah,chimpanzee_bonobo,civet_genet,elephant,equid,forest_buffalo,fox,giraffe,gorilla,hare_rabbit,hippopotamus,hog,human,hyena,large_flightless_bird,leopard,lion,mongoose,monkey_prosimian,pangolin,porcupine,reptile,rodent,small_cat,wild_dog_jackal
eleph.MP4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
leopard.MP4,0.0,0.0,0.0,0.0,2e-05,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0125,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
blank.MP4,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
chimp.MP4,0.0,0.0,0.0,0.0,0.0,0.0,1e-05,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1e-05,4e-05,0.00162,0.0,0.0,0.0,0.0,0.0,2e-05,2e-05,0.0,1e-05,0.0,0.0038,4e-05,0.0
```

The full prediction and video loadering configuration for the process will also be saved out, in the same folder as the predictions under `predict_configuration.yaml`. To run the exact same inference process a second time, you can pass this YAML file to `zamba predict` per the [Using YAML Configuration Files](yaml-config.md) page:
```console
$ zamba predict --config predict_configuration.yaml
```

## Step-by-step tutorial

### 1. Specify the path to your videos

Save all of your videos within one folder.

* They can be in nested directories within the folder.
* Your videos should all be saved in formats that are suppored by FFmpeg, [which are listed here](https://www.ffmpeg.org/general.html#Supported-File-Formats_002c-Codecs-or-Features). Any videos that fail a set of FFmpeg checks will be skipped during inference or training.

Add the path to your video folder. For example, if your videos are in a folder called `example_vids`:

=== "CLI"
    ```console
    $ zamba predict --data-dir example_vids/
    ```
=== "Python"
    ```python
    predict_config = PredictConfig(data_directory='example_vids/')
    predict_model(predict_config=predict_config)
    ```

### 2. Choose a model for prediction

If your camera videos contain species common to central or west Africa:

* Use the [`time_distributed` model](models.md#time-distributed) if your priority is species classification
* Use the [`slowfast` model](models.md#slowfast) if your priority is blank vs. non-blank video detection

If your videos contain species common to Europe, use the [`european` model](models.md#european).

Add the model name to your command. The `time_distributed` model will be used if no model is specified. For example, if you want to use the `slowfast` model to classify the videos in `example_vids`:

=== "CLI"
    ```console
    $ zamba predict --data-dir example_vids/ --model slowfast
    ```
=== "Python"
    ```python
    predict_config = PredictConfig(
        data_directory='example_vids/', model_name='slowfast'
    )
    predict_model(predict_config=predict_config)
    ```

### 3. Choose the output format

There are three options for how to format predictions, listed from most information to least:

1. **Store all probabilities (default):** Return predictions with a row for each filename and a column for each class label, with probabilities between 0 and 1. Cell (i,j) is the probability that animal j is present in video i.
2. **Presence/absence:** Return predictions with a row for each filename and a column for each class label, with cells indicating either presence or absense based on a user-specified probability threshold. Cell (i, j) indicates whether animal j is present (`1`) or not present (`0`) in video i. The probability threshold cutoff is specified with `--proba-threshold` in the CLI. 
3. **Most likely class:** Return predictions with a row for each filename and one column for the most likely class in each video. The most likely class can also be blank. To get the most likely class, add `--output-class-names` to your command. In Python, it can be specified by adding `output_class_names=True` when `PredictConfig` is instantiated. This is not recommended if you'd like to detect more than one species in each video.

Say we want to generate predictions for the videos in `example_vids` indicating which animals are present in each video based on a probability threshold of 50%:

=== "CLI"
    ```console
    $ zamba predict --data-dir example_vids/ --proba-threshold 0.5
    $ cat zamba_predictions.csv
    filepath,aardvark,antelope_duiker,badger,bat,bird,blank,cattle,cheetah,chimpanzee_bonobo,civet_genet,elephant,equid,forest_buffalo,fox,giraffe,gorilla,hare_rabbit,hippopotamus,hog,human,hyena,large_flightless_bird,leopard,lion,mongoose,monkey_prosimian,pangolin,porcupine,reptile,rodent,small_cat,wild_dog_jackal
    eleph.MP4,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    leopard.MP4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0
    blank.MP4,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    chimp.MP4,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    ```
=== "Python"
    ```python
    predict_config = PredictConfig(
        data_directory="example_vids/", proba_threshold=0.5
    )
    predict_model(predict_config=predict_config)
    predictions = pd.read_csv("zamba_predictions.csv")
    predictions
    ```

    | filepath    | aardvark | antelope_duiker | badger | bat | bird | blank | cattle | cheetah | chimpanzee_bonobo | civet_genet | elephant | equid | forest_buffalo | fox | giraffe | gorilla | hare_rabbit | hippopotamus | hog | human | hyena | large_flightless_bird | leopard | lion | mongoose | monkey_prosimian | pangolin | porcupine | reptile | rodent | small_cat | wild_dog_jackal |
    | ----------- | -------- | --------------- | ------ | --- | ---- | ----- | ------ | ------- | ----------------- | ----------- | -------- | ----- | -------------- | --- | ------- | ------- | ----------- | ------------ | --- | ----- | ----- | --------------------- | ------- | ---- | -------- | ---------------- | -------- | --------- | ------- | ------ | --------- | --------------- |
    | blank.MP4   | 0        | 0               | 0      | 0   | 0    | 1     | 0      | 0       | 0                 | 0           | 0        | 0     | 0              | 0   | 0       | 0       | 0           | 0            | 0   | 0     | 0     | 0                     | 0       | 0    | 0        | 0                | 0        | 0         | 0       | 0      | 0         | 0               |
    | chimp.MP4   | 0        | 0               | 0      | 0   | 0    | 0     | 0      | 0       | 1                 | 0           | 0        | 0     | 0              | 0   | 0       | 0       | 0           | 0            | 0   | 0     | 0     | 0                     | 0       | 0    | 0        | 0                | 0        | 0         | 0       | 0      | 0         | 0               |
    | eleph.MP4   | 0        | 0               | 0      | 0   | 0    | 0     | 0      | 0       | 0                 | 0           | 1        | 0     | 0              | 0   | 0       | 0       | 0           | 0            | 0   | 0     | 0     | 0                     | 0       | 0    | 0        | 0                | 0        | 0         | 0       | 0      | 0         | 0               |
    | leopard.MP4 | 0        | 0               | 0      | 0   | 0    | 0     | 0      | 0       | 0                 | 0           | 0        | 0     | 0              | 0   | 0       | 0       | 0           | 0            | 0   | 0     | 0     | 0                     | 1       | 0    | 0        | 0                | 0        | 0         | 0       | 0      | 0         | 0               |

### 4. Specify any additional parameters

And there's so much more! You can also do things like specify your region for faster model download (`--weight-download-region`), use a saved model checkpoint (`--checkpoint`), or specify a different path where your predictions should be saved (`--save`). To read about a few common considerations, see the [Guide to Common Optional Parameters](extra-options.md) page.

### 5. Test your configuration with a dry run

Before kicking off a full run of inference, we recommend testing your code with a "dry run". This will run one batch of inference to quickly detect any bugs. See the [Debugging](debugging.md) page for details.