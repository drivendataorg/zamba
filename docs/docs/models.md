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

`time_distributed` and `european` use the same basic algorithm. The main difference is that they are trained on different geographies.

For training or fine tuning, either the `time_distributed` and `european` model is recommended. These run much more quickly thatn the `slowfast` model.

For inference, `slowfast` is recommended if the highest priority is differentiating between blank and non-blank videos. If the priority is species classification, either `time_distributed` or `european` is recommended based on the given geography.

<h2 id="species-classes"></h2>

## What species can `zamba` detect?

`time_distributed` and `slowfast` are both trained to identify 31 common species from central and west Africa. The possible class labels in these models are:

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

### Default video loading configuration

<!-- TODO: add link to yaml file><!-->

Running the `time_distributed` model on every frame would be very time consuming. Instead, a more efficient object detection model called [MegadetectorLiteYoloX](#megadetectorliteyolox) is run on all frames to determine which are the most likely to contain an animal. Then `time_distributed` is only run on the 32 frames with the highest predicted probability of detection. **`time_distributed` requires that 16 frames be sampled from each video before running inference or training.** 

By default, videos are resized to 224x224 pixels. To run `time_distributed`, `video_height` and `video_width` must be specified.

The above is the default behavior if `time_distributed` is used in the command line. If you are passing in a custom [YAML configuration file](yaml-config.md) or using `zamba` as a Python package, at a minimum you must specify:
```yaml
video_loader_config:
  video_height: # any integer
  video_width: # any integer
  total_frames: 16
```

The full default video loading configuration is:
```yaml
video_loader_config:
  video_height: 224
  video_width: 224
  crop_bottom_pixels: 50
  ensure_total_frames: True
  megadetector_lite_config:
    confidence: 0.25
    fill_model: "score_sorted"
    n_frames: 16
  total_frames: 16
```

<a id='slowfast'></a>

## `slowfast` model

### Algorithm

The `slowfast` model was built by re-training a video classification backbone called [SlowFast](https://arxiv.org/abs/1812.03982) (Feichtenhofer, C., Fan, H., Malik, J., & He, K., 2019). SlowFast refers to the two model pathways involved: one that operates at a low frame rate to capture spatial semantics, and one that operatues at a high frame rate to capture motion over time. The basic architectures are deep [neural networks](https://www.youtube.com/watch?v=aircAruvnKk&t=995s) using [pytorch](https://pytorch.org/).

<div style="text-align:center;">
<img src="https://s3.amazonaws.com/drivendata-public-assets/zamba-slowfast-diagram.png" alt="Architecture showing the two pathways of the slowfast model" style="width:400px;"/>
<br/><br/>
<i>Source:</i> Feichtenhofer, C., Fan, H., Malik, J., & He, K. (2019). Slowfast networks for video recognition. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 6202-6211).
</div>

Unlike `time_distributed`, `slowfast` is video native. This means it takes into account the relationship between frames in a video, rather than running independently on each frame.

### Training data

The `slowfast` model was trained using the same data as the [`time_distributed` model](#time-distributed-training-data).

### Video loading configuration

<!-- TODO: add link to yaml file><!-->

Running the `slowfast` model on every frame would be very time consuming. Instead, a more efficient object detection model called [MegadetectorLiteYoloX](#megadetectorliteyolox) is run on all frames to determine which are the most likely to contain an animal. Then `slowfast` is only run on the 32 frames with the highest predicted probability of detection. **`slowfast` requires that 32 frames be sampled from each video before running inference or training.** 

By default, videos are resized to 224x224 pixels. To run `slowfast`, `video_height` and `video_width` must be specified and must each be greater than or equal to 200. 

The above is the default behavior if `slowfast` is used in the command line. If you are passing in a custom [YAML configuration file](yaml-config.md) or using `zamba` as a Python package, at a minimum you must specify:

```yaml
video_loader_config:
  video_height: # any integer >= 200
  video_width: # any integer >= 200
  total_frames: 32
```

The full default video loading configuration is:

```yaml
video_loader_config:
  video_height: 224
  video_width: 224
  crop_bottom_pixels: 50
  ensure_total_frames: True
  megadetector_lite_config:
    confidence: 0.25
    fill_model: "score_sorted"
    n_frames: 32
  total_frames: 32
```

<a id='european'></a>

## `european` model

### Algorithm

The `european` model has the same backbone as the `time_distributed` model, but is trained on data from camera traps in western Europe instead of central and west Africa. 

The `european` model was built by re-training a well-known image classification architecture called [EfficientNetV2](https://arxiv.org/abs/1905.11946) to identify the species in our camera trap videos (Tan, M., & Le, Q., 2019). EfficientNetV2 models are convolutional [neural networks](https://www.youtube.com/watch?v=aircAruvnKk&t=995s) designed to jointly optimize model size and training speed. EfficientNetV2 is image native, meaning it classifies each frame separately when generating predictions. It does take into account the relationship between frames in the video.

`european` combines the EfficientNetV2 architecture with an open-source image object detection model to implement frame selection. The [YOLOX](https://arxiv.org/abs/2107.08430) detection model is run on all frames in a video. Only the frames with the highest probability of detection are then passed to the more computationally intensive EfficientNetV2 for detailed detection and classification.

### Training data

`european` was trained using data collected and annotated by partners at [The Max Planck Institute for
Evolutionary Anthropology](https://www.eva.mpg.de/index.html). The data included camera trap videos from Hintenteiche bei Biesenbrow, Germany.

### Default video loading configuration

<!-- TODO: add link to yaml file><!-->

Running the `european` model on every frame would be very time consuming. Instead, a more efficient object detection model called [MegadetectorLiteYoloX](#megadetectorliteyolox) is run on all frames to determine which are the most likely to contain an animal. Then `european` is only run on the 32 frames with the highest predicted probability of detection. **`european` requires that 16 frames be sampled from each video before running inference or training.** 

By default, videos are resized to 224x224 pixels. To run `european`, `video_height` and `video_width` must be specified.

The above is the default behavior if `european` is used in the command line. If you are passing in a custom [YAML configuration file](yaml-config.md) or using `zamba` as a Python package, at a minimum you must specify:

```yaml
video_loader_config:
  video_height: # any integer
  video_width: # any integer
  total_frames: 16
```

The full default video loading configuration is:
```yaml
video_loader_config:
  video_height: 224
  video_width: 224
  crop_bottom_pixels: 50
  ensure_total_frames: True
  megadetector_lite_config:
    confidence: 0.25
    fill_model: "score_sorted"
    n_frames: 16
  total_frames: 16
```

<a id='megadetectorliteyolox'></a>

## MegaDetectorLiteYoloX

Running any of the three models that ship with `zamba` on all frames of a video would be incredibly time consuming and computationally intensive. Instead, `zamba` uses a more efficient object detection model called MegaDetectorLiteYoloX to determine the likelihood that each frame contains an animal. Then, only the frames with the highest probability of detection can be passed to the model.

MegaDetectorLiteYoloX combines two open-source models:

* [MegaDetector](https://github.com/microsoft/CameraTraps/blob/master/megadetector.md) is a pretrained image model designed to detect animals, people, and vehicles in camera trap videos.
* [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) is a high-performance, lightweight object detection model that is much less computationally intensive than MegaDetector.

MegaDetector is much better at identifying frames with animals than YOLOX, but too computationally intensive to run on every frame. MegaDetectorLiteYoloX was created by training the YOLOX model using the predictions of the MegaDetector as ground truth - this method is called [student-teacher training](https://towardsdatascience.com/knowledge-distillation-simplified-dd4973dbc764).