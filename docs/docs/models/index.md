# Available Models

The algorithms in `zamba` are designed to identify species of animals that appear in camera trap videos. There are three models that ship with the `zamba` package: `time_distributed`, `slowfast`, and `european`. For more details of each, read on!

## Basic usage

<table>
  <tr>
    <th>Model</th>
    <th>Use cases</th>
    <th>Strengths</th>
    <th>Geography</th>
  </tr>
  <tr>
    <td><code>time_distributed</code></td>
    <td rowspan=2>Model training or fine tuning</td>
    <td rowspan=2>Classifying species<br/>Running more quickly</td>
    <td>Central and west Africa</td>
  </tr>
  <tr>
    <td><code>european</code></td>
    <td>Western Europe</td>
  </tr>
<tr>
    <td><code>slowfast</code></td>
    <td>Detailed prediction of blank vs. non-blank</td>
    <td>Identifying blank vs. non-blank videos</td>
    <td>Central and west Africa</td>
  </tr>
</table>

`time_distributed` and `european` use the same basic algorithm. The main difference is that they predict different species based on their intended geography.

For training or fine tuning, either the `time_distributed` and `european` model is recommended. These run much more quickly thatn the `slowfast` model.

For inference, `slowfast` is recommended if the highest priority is differentiating between blank and non-blank videos. If the priority is species classification, either `time_distributed` or `european` is recommended based on the given geography.

<h2 id="species-classes"></h2>

## What species can `zamba` detect?

`time_distributed` and `slowfast` are both trained to identify 32 common species from central and west Africa. The possible class labels in these models are:

* `aardvark`
* `antelope_duiker`
* `badger`
* `bat`
* `bird`
* `blank`
* `cattle`
* `cheetah`
* `chimpanzee_bonobo`
* `civet_genet`
* `elephant`
* `equid`
* `forest_buffalo`
* `fox`
* `giraffe`
* `gorilla`
* `hare_rabbit`
* `hippopotamus`
* `hog`
* `human`
* `hyena`
* `large_flightless_bird`
* `leopard`
* `lion`
* `mongoose`
* `monkey_prosimian`
* `pangolin`
* `porcupine`
* `reptile`
* `rodent`
* `small_cat`
* `wild_dog_jackal`

`european` is trained to identify 11 common species in western Europe. The possible class labels are:

* `bird`
* `blank`
* `domestic_cat`
* `european_badger`
* `european_beaver`
* `european_hare`
* `european_roe_deer`
* `north_american_raccoon`
* `red_fox`
* `unidentified`
* `weasel`
* `wild_boar`

<a id='time-distributed'></a>

## `time_distributed` model

### Algorithm

The `time_distributed` model was built by re-training a well-known image classification architecture called [EfficientNetV2](https://arxiv.org/abs/1905.11946) to identify the species in our camera trap videos (Tan, M., & Le, Q., 2019). EfficientNetV2 models are convolutional [neural networks](https://www.youtube.com/watch?v=aircAruvnKk&t=995s) designed to jointly optimize model size and training speed. EfficientNetV2 is image native, meaning it classifies each frame separately when generating predictions. It does take into account the relationship between frames in the video.

<a id='time-distributed-training-data'></a>

### Training data

`time_distributed` was trained using data collected and annotated by partners at [The Max Planck Institute for
Evolutionary Anthropology](https://www.eva.mpg.de/index.html) and [Chimp &
See](https://www.chimpandsee.org/). The data included camera trap videos from:

* Dzanga-Sangha Protected Area, Central African Republic
* Gorongosa National Park, Mozambique
* Grumeti Game Reserve, Tanzania
* Lopé National Park, Gabon
* Moyen-Bafing National Park, Guinea
* Nouabale-Ndoki National Park, Republic of the Congo
* Salonga National Park, Democratic Republic of the Congo
* Taï National Park, Côte d'Ivoire

### Default configuration

The full default configuration is available on [Github](https://github.com/drivendataorg/zamba/blob/v2/zamba/models/official_models/time_distributed/config.yaml).

By default, an efficient object detection model called [MegadetectorLite](#megadetectorlite) is run on all frames to determine which are the most likely to contain an animal. Then `time_distributed` is run on only the 16 frames with the highest predicted probability of detection. By default, videos are resized to 240x426 pixels.

The full default video loading configuration is:
```yaml
video_loader_config:
  model_input_height: 240
  model_input_width: 426
  crop_bottom_pixels: 50
  ensure_total_frames: True
  megadetector_lite_config:
    confidence: 0.25
    fill_model: "score_sorted"
    n_frames: 16
  total_frames: 16
```

### Requirements

The above is pulled in by default if `time_distributed` is used in the command line. If you are passing in a custom [YAML configuration file](../yaml-config.md) or using `zamba` as a Python package, at a minimum you must specify:
=== "YAML file"
    ```yaml
    video_loader_config:
      model_input_height: # any integer
      model_input_width: # any integer
      total_frames: 16
    ```
=== "Python"
    ```python
    video_loader_config = VideoLoaderConfig(
      model_input_height=..., # any integer
      model_input_width=..., # any integer
      total_frames=16
    )
    ```

<a id='slowfast'></a>

## `slowfast` model

### Algorithm

The `slowfast` model was built by re-training a video classification backbone called [SlowFast](https://arxiv.org/abs/1812.03982) (Feichtenhofer, C., Fan, H., Malik, J., & He, K., 2019). SlowFast refers to the two model pathways involved: one that operates at a low frame rate to capture spatial semantics, and one that operates at a high frame rate to capture motion over time. The basic architectures are deep [neural networks](https://www.youtube.com/watch?v=aircAruvnKk&t=995s) using [pytorch](https://pytorch.org/).

<div style="text-align:center;">
<img src="https://s3.amazonaws.com/drivendata-public-assets/zamba-slowfast-diagram.png" alt="Architecture showing the two pathways of the slowfast model" style="width:400px;"/>
<br/><br/>
<i>Source:</i> Feichtenhofer, C., Fan, H., Malik, J., & He, K. (2019). Slowfast networks for video recognition. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 6202-6211).
</div>

Unlike `time_distributed`, `slowfast` is video native. This means it takes into account the relationship between frames in a video, rather than running independently on each frame.

### Training data

The `slowfast` model was trained using the same data as the [`time_distributed` model](#time-distributed-training-data).

### Default configuration

The full default configuration is available on [Github](https://github.com/drivendataorg/zamba/blob/v2/zamba/models/official_models/slowfast/config.yaml).

By default, an efficient object detection model called [MegadetectorLite](#megadetectorlite) is run on all frames to determine which are the most likely to contain an animal. Then `slowfast` is run on only the 32 frames with the highest predicted probability of detection. By default, videos are resized to 240x426 pixels.

The full default video loading configuration is:

```yaml
video_loader_config:
  model_input_height: 240
  model_input_width: 426
  crop_bottom_pixels: 50
  ensure_total_frames: True
  megadetector_lite_config:
    confidence: 0.25
    fill_model: "score_sorted"
    n_frames: 32
  total_frames: 32
```

### Requirements

The above is pulled in by default if `slowfast` is used in the command line. If you are passing in a custom [YAML configuration file](../yaml-config.md) or using `zamba` as a Python package, at a minimum you must specify:
=== "YAML file"
    ```yaml
    video_loader_config:
      model_input_height: # any integer >= 200
      model_input_width: # any integer >= 200
      total_frames: 32
    ```
=== "Python"
    ```python
    video_loader_config = VideoLoaderConfig(
      model_input_height=..., # any integer >= 200
      model_input_width=..., # any integer >= 200
      total_frames=32
    )
    ```

<a id='european'></a>

## `european` model

### Algorithm

The `european` model has the same backbone as the `time_distributed` model, but is trained on data from camera traps in western Europe instead of central and west Africa.

The `european` model was built by re-training a well-known image classification architecture called [EfficientNetV2](https://arxiv.org/abs/1905.11946) to identify the species in our camera trap videos (Tan, M., & Le, Q., 2019). EfficientNetV2 models are convolutional [neural networks](https://www.youtube.com/watch?v=aircAruvnKk&t=995s) designed to jointly optimize model size and training speed. EfficientNetV2 is image native, meaning it classifies each frame separately when generating predictions. It does take into account the relationship between frames in the video.

`european` combines the EfficientNetV2 architecture with an open-source image object detection model to implement frame selection. The [YOLOX](https://arxiv.org/abs/2107.08430) detection model is run on all frames in a video. Only the frames with the highest probability of detection are then passed to the more computationally intensive EfficientNetV2 for detailed detection and classification.

### Training data

The `european` model is built by starting with the fully trained `time_distributed` model. The network is then finetuned with data collected and annotated by partners at [The Max Planck Institute for
Evolutionary Anthropology](https://www.eva.mpg.de/index.html). The finetuning data included camera trap videos from Hintenteiche bei Biesenbrow, Germany.

### Default configuration

The full default configuration is available on [Github](https://github.com/drivendataorg/zamba/blob/v2/zamba/models/official_models/european/config.yaml).

By default, an efficient object detection model called [MegadetectorLite](#megadetectorlite) is run on all frames to determine which are the most likely to contain an animal. Then `european` is run on only the 16 frames with the highest predicted probability of detection. By default, videos are resized to 240x426 pixels.

The full default video loading configuration is:
```yaml
video_loader_config:
  model_input_height: 240
  model_input_width: 426
  crop_bottom_pixels: 50
  ensure_total_frames: True
  megadetector_lite_config:
    confidence: 0.25
    fill_model: "score_sorted"
    n_frames: 16
  total_frames: 16
```

### Requirements

The above is pulled in by default if `european` is used in the command line. If you are passing in a custom [YAML configuration file](../yaml-config.md) or using `zamba` as a Python package, at a minimum you must specify:

=== "YAML file"
    ```yaml
    video_loader_config:
      model_input_height: # any integer
      model_input_width: # any integer
      total_frames: 16
    ```
=== "Python"
    ```python
    video_loader_config = VideoLoaderConfig(
      model_input_height=..., # any integer
      model_input_width=..., # any integer
      total_frames=16
    )
    ```

<a id='megadetectorlite'></a>

## MegadetectorLite

Running any of the three models that ship with `zamba` on all frames of a video would be incredibly time consuming and computationally intensive. Instead, `zamba` uses a more efficient object detection model called MegadetectorLite to determine the likelihood that each frame contains an animal. Then, only the frames with the highest probability of detection can be passed to the model.

MegadetectorLite combines two open-source models:

* [Megadetector](https://github.com/microsoft/CameraTraps/blob/master/megadetector.md) is a pretrained image model designed to detect animals, people, and vehicles in camera trap videos.
* [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) is a high-performance, lightweight object detection model that is much less computationally intensive than Megadetector.

While highly accurate, Megadetector is too computationally intensive to run on every frame. MegadetectorLite was created by training a YOLOX model using the predictions of the Megadetector as ground truth - this method is called [student-teacher training](https://towardsdatascience.com/knowledge-distillation-simplified-dd4973dbc764).
