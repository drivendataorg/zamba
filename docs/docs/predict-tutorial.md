# User Tutorial: Classifying Unlabeled Videos

This section walks through how to classify videos using `zamba`. If you are new to `zamba` and just want to classify some videos as soon as possible, see the [Quickstart](quickstart.md) guide.

This tutorial goes over the steps for using `zamba` if:

* You already have `zamba` installed (for details see the [Installation](install.md) page)
* You have unlabeled videos that you want to generate labels for
* The possible class species labels for your videos are included in the list of possible [zamba labels](models.md#species-classes)

## Basic usage: command line interface

Say that we want to classify the videos in a folder called `example_vids` as simply as possible using all of the default settings.

Minimum example for prediction in the command line:

```console
$ zamba predict --data-dir example_vids/
```

### Required arguments

To run `zamba predict` in the command line, you must specify either `--data-dir` or `--file-list`. 

* **`--data-dir PATH`:** Path to the folder containing your videos.
* **`--file-list PATH`:** Path to a CSV file with a column for the filepath to each video you want to classify. The CSV must have a column for `filepath`.

All other flags are optional. To choose a model, either `--model` or `--checkpoint` must be specified. `--model` defaults to `time_distributed`.

## Basic usage: Python package

Say that we want to classify the videos in a folder called `example_vids` as simply as possible using all of the default settings.

Minimum example for prediction using the Python package:

```python
from zamba.models.model_manager import predict_model
from zamba.models.config import PredictConfig
from zamba_algorithms.data.video import VideoLoaderConfig

predict_config = PredictConfig(data_directory='example_vids/')
video_loader_config = VideoLoaderConfig(video_height=224, 
                                        video_width=224, 
                                        total_frames=16)

predict_model(predict_config=predict_config, 
              video_loader_config=video_loader_config)
```

To specify various parameters when running `predict_model`, the first step is to instantiate [`PredictConfig`](configurations.md#prediction-arguments) and [`VideoLoaderConfig`](configurations.md#video-loading-arguments) with any specifications for prediction and video loading respectively. The only two arguments that can be specified in `predict_model` are `predict_config` and `video_loader_config`.

### Required arguments

To run `predict_model` in Python, you must specify either `data_directory` or `filepaths` when `PredictConfig` is instantiated.

* **`data_directory (DirectoryPath)`:** Path to the folder containing your videos.

* **`filepaths (FilePath)`:** Path to a CSV file with a column for the filepath to each video you want to classify. The CSV must have a column for `filepath`.

In the command line, video loading configurations are loaded by default based on the model being used. This is not the case in Python. There are additional requirements for `VideoLoaderConfig` based on the model you are using.

* **`video_height (int)`, `video_width (int)`:** Dimensions for resizing videos as they are loaded. 
    - `time_distributed` or `european`: The suggested dimensions are 224x224, but any integers are acceptable
    - `slowfast`: Both must be greater than or equal to 200
* **`total_frames (int)`:** The number of frames to select from each video and use during inference. 
    * `time_distributed` or `european`: Must be 16
    * `slowfast`: Must be 32

The full recommended `VideoLoaderConfig` for the `time_distributed` or `european` model is:
```python
from zamba_algorithms.data.video import VideoLoaderConfig
from zamba.models.megadetector_lite_yolox import MegadetectorLiteYoloXConfig

megadetector_config = MegadetectorLiteYoloXConfig(confidence=0.25,
                                                  fill_mode="score_sorted",
                                                  n_frames=16)
video_loader_config = VideoLoaderConfig(video_height=224,
                                        video_width=224,
                                        crop_bottom_pixels=50,
                                        ensure_total_frames=True,
                                        megadetector_list_config=megadetector_config,
                                        total_frames=16)
```

The full recommended `VideoLoaderConfig` for the `slowfast` model is:
```python
megadetector_config = MegadetectorLiteYoloXConfig(confidence=0.25,
                                                  fill_mode="score_sorted",
                                                  n_frames=32)
video_loader_config = VideoLoaderConfig(video_height=224,
                                        video_width=224,
                                        crop_bottom_pixels=50,
                                        ensure_total_frames=True,
                                        megadetector_list_config=megadetector_config,
                                        total_frames=32)
```

You can see the full default configuration for each model in `models/config`<!-- TODO: add link to source and update if needed><!-->. For detailed explanations of all possible configuration arguments, see [All Optional Arguments](configurations.md).

## Default behavior

In each case, `zamba` will output a `.csv` file with rows labeled by each video filename and columns for each class (ie. species). The default prediction will store all class probabilities, so that cell (i,j) can be interpreted as *the probability that animal j is present in video i.* 

Predictions will be saved to `{model name}_{current timestamp}_preds.csv`. For example, running `zamba predict` on 9/15/2021 with the `time_distributed` model (the default) will save predictions to `time_distributed_2021-09-15_preds.csv`. 

```console
$ cat time_distributed_2021-09-15_preds.csv
filepath,aardvark,antelope_duiker,badger,bat,bird,blank,cattle,cheetah,chimpanzee_bonobo,civet_genet,elephant,equid,forest_buffalo,fox,giraffe,gorilla,hare_rabbit,hippopotamus,hog,human,hyena,large_flightless_bird,leopard,lion,mongoose,monkey_prosimian,pangolin,porcupine,reptile,rodent,small_cat,wild_dog_jackal
example_vids/eleph.MP4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
example_vids/leopard.MP4,0.0,0.0,0.0,0.0,2e-05,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0125,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
example_vids/blank.MP4,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
example_vids/chimp.MP4,0.0,0.0,0.0,0.0,0.0,0.0,1e-05,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1e-05,4e-05,0.00162,0.0,0.0,0.0,0.0,0.0,2e-05,2e-05,0.0,1e-05,0.0,0.0038,4e-05,0.0
```

## Step-by-step tutorial

### 1. Specify the path to your videos

Save all of your videos in one folder.

* Your videos should all be saved in formats that are suppored by FFMPEG, [which are listed here](https://www.ffmpeg.org/general.html#Supported-File-Formats_002c-Codecs-or-Features).
* Your video folder must contain only valid video files, since zamba will try to load all of the files in the directory.
* Your videos must all be in the top level of the video folder - `zamba` does not extract videos from nested directories.

Add the path to your video folder with `--data-dir`. For example, if your videos are in a folder called `example_vids`:

```console
$ zamba predict --data-dir example_vids/
```

In the python package, the data directory is specified when `PredictConfig` is instantiated:

```python
predict_config = PredictConfig(data_directory='example_vids/')
```

### 2. Choose a model for prediction

If your camera videos contain species common to central or west Africa:

* Use the [`time_distributed` model](models.md#time-distributed) if your priority is species classification
* Use the [`slowfast` model](models.md#slowfast) if your priority is blank vs. non-blank video detection

If your videos contain species common to Europe, use the [`european` model](models.md#european).

Add the model name to your command with `--model`. The `time_distributed` model will be used if no model is specified. For example, if you want to use the `slowfast` model to classify the videos in `example_vids`:

```console
$ zamba predict --data-dir example_vids/ --model slowfast
```

In the Python package, model is specified when `PredictConfig` is instantiated:

```python
predict_config = PredictConfig(data_directory='example_vids/', 
                               model_name='slowfast')
```

### 3. Choose the output format

There are three options for how to format predictions, listed from most information to least:

1. **Store all probabilities (default):** Return predictions with a row for each filename and a column for each class label, with probabilities between 0 and 1. Cell (i,j) is the probability that animal j is present in video i.
2. **Presence/absence:** Return predictions with a row for each filename and a column for each class label, with cells indicating either presence or absense based on a user-specified probability threshold. Cell (i, j) indicates whether animal j is present (`1`) or not present (`0`) in video i. The probability threshold cutoff is specified with `--proba-threshold` in the CLI. 
3. **Most likely class:** Return predictions with a row for each filename and one column for the most likely class in each video. The most likely class can also be blank. To get the most likely class, add `--output-class-names` to your command. In Python, it can be specified by adding `output_class_names=True` when `PredictConfig` is instantiated.

Say we want to generate predictions for the videos in `example_vids` indicating which animals are present in each video based on a probability threshold of 50%:
```console
$ zamba predict --data-dir example_vids/ --proba-threshold 0.5
$ cat time_distributed_2021-09-16_preds.csv
filepath,aardvark,antelope_duiker,badger,bat,bird,blank,cattle,cheetah,chimpanzee_bonobo,civet_genet,elephant,equid,forest_buffalo,fox,giraffe,gorilla,hare_rabbit,hippopotamus,hog,human,hyena,large_flightless_bird,leopard,lion,mongoose,monkey_prosimian,pangolin,porcupine,reptile,rodent,small_cat,wild_dog_jackal
example_vids/eleph.MP4,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
example_vids/leopard.MP4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0
example_vids/blank.MP4,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
example_vids/chimp.MP4,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
```

In the Python package, use the `proba_threshold` argument:

```python
predict_config = PredictConfig(data_directory='example_vids/', 
                               proba_threshold=0.5
predictions = pd.read_csv('time_distributed_2021-09-16_preds.csv')
predictions
```

| filepath                 | aardvark | antelope_duiker | badger | bat | bird | blank | cattle | cheetah | chimpanzee_bonobo | civet_genet | elephant | equid | forest_buffalo | fox | giraffe | gorilla | hare_rabbit | hippopotamus | hog | human | hyena | large_flightless_bird | leopard | lion | mongoose | monkey_prosimian | pangolin | porcupine | reptile | rodent | small_cat | wild_dog_jackal |
| ------------------------ | -------- | --------------- | ------ | --- | ---- | ----- | ------ | ------- | ----------------- | ----------- | -------- | ----- | -------------- | --- | ------- | ------- | ----------- | ------------ | --- | ----- | ----- | --------------------- | ------- | ---- | -------- | ---------------- | -------- | --------- | ------- | ------ | --------- | --------------- |
| example_vids/blank.MP4   | 0        | 0               | 0      | 0   | 0    | 1     | 0      | 0       | 0                 | 0           | 0        | 0     | 0              | 0   | 0       | 0       | 0           | 0            | 0   | 0     | 0     | 0                     | 0       | 0    | 0        | 0                | 0        | 0         | 0       | 0      | 0         | 0               |
| example_vids/chimp.MP4   | 0        | 0               | 0      | 0   | 0    | 0     | 0      | 0       | 1                 | 0           | 0        | 0     | 0              | 0   | 0       | 0       | 0           | 0            | 0   | 0     | 0     | 0                     | 0       | 0    | 0        | 0                | 0        | 0         | 0       | 0      | 0         | 0               |
| example_vids/eleph.MP4   | 0        | 0               | 0      | 0   | 0    | 0     | 0      | 0       | 0                 | 0           | 1        | 0     | 0              | 0   | 0       | 0       | 0           | 0            | 0   | 0     | 0     | 0                     | 0       | 0    | 0        | 0                | 0        | 0         | 0       | 0      | 0         | 0               |
| example_vids/leopard.MP4 | 0        | 0               | 0      | 0   | 0    | 0     | 0      | 0       | 0                 | 0           | 0        | 0     | 0              | 0   | 0       | 0       | 0           | 0            | 0   | 0     | 0     | 0                     | 1       | 0    | 0        | 0                | 0        | 0         | 0       | 0      | 0         | 0               |

### 4. Specify any additional parameters

And there's so much more! You can also do things like specify your region for faster model download (`--weight-download-region`), use a saved model checkpoint (`--checkpoint`), or run only one batch for faster debugging (`--dry-run`). We'll go through a few common options to consider. If you using the command line interface, all of the parameters in this section must be passed as part of a [YAML configuration file](yaml-config.md) rather than directly to the command line.

#### Video size

`zamba` can resize all videos before running inference. Higher resolution videos will lead to more detailed accuracy in prediction, but will use more memory and take longer to run inference.

The default for all pretrained models is 224x224 pixels. If you have only a few videos for which you want highly detailed predictions, you could instead specify a larger size like 500x500 pixels. Your [YAML configuration file](yaml-config.md) would include:
```yaml
video_loader_config:
  video_height: 500
  video_width: 500
```

In Python, video resizing can be specified when `VideoLoaderConfig` is instantiated:

```python
video_loader_config = VideoLoaderConfig(video_height=500, video_width=500)
```

#### Frame selection

During inference, the model only generates prediction based on a subset of the frames in a video. There are a number of different ways to select frames (see the section on [Video loading arguments](configurations.md#video-loading-arguments) for details). A few possible methods:

* If animals are more likely to be seen early in the video because that is closer to when the camera trap was triggered, you may want to set `early_bias` to True. This selects 16 frames towards the beginning of the video.
* A simple option is to sample frames that are evenly distributed throughout a video. For example, to select 32 evenly distributed frames, add the following to a [YAML configuration file](yaml-config.md):
```
video_loader_config:
    total_frames: 32
    evenly_sample_total_frames: True
    ensure_total_frames: True
```
In Python, these arguments can be specified when `VideoLoaderConfig` is instantiated:
```python
video_loader_config = VideoLoaderConfig(total_frames=32,
                                        evenly_sample_total_frames=True,
                                        ensure_total_frames=True)
```
* You can use a pretrained object detection model called MegadetectorLiteYoloX to select only the frames that are mostly likely to contain an animal - this is the default method. The parameter `megadetector_lite_config` is used to specify any arguments that should be passed to the megadetector model. For example, to take the 16 frames with the highest probability of detection based on the megadetector, add the following to a [YAML configuration file](yaml-config.md):
```yaml
video_loader_config:
    megadetector_lite_config:
        n_frames: 16
        fill_mode: "score_sorted"
```

In Python, these can be specified in the `megadetector_lite_config` argument passed to `VideoLoaderConfig`:
```python
video_loader_config = VideoLoaderConfig(
    video_height=224,
    video_width=224,
    crop_bottom_pixels=50,
    ensure_total_frames=True,
    megadetector_lite_config={"confidence": 0.25, "fill_mode": "score_sorted", "n_frames": 16},
    total_frames=16,
)

predict_config = PredictConfig(data_directory='example_vids/')

predict_model(video_loader_config=video_loader_config, 
              predict_config=predict_config)
```

And that's just the tip of the iceberg! See the [All Optional Arguments](configurations.md) page for more possibilities.