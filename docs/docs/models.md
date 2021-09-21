# Available Models

The algorithms in `zamba` are designed to identify species of animals that appear in camera trap videos. There are three models that ship with the `zamba` package: `time_distributed`, `slowfast`, and `european`. For more details of each, read on!

<!-- TODO: what is the final data each model is trained on? once finalized, add:
- # of videos for each model
- # of sites
- kinds of sites?><!-->

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

The `time_distributed` model was built by re-training a well-known image classification architecture called [EfficientNetV2](https://arxiv.org/abs/1905.11946) to identify the species in our camera trap videos (Tan, M., & Le, Q., 2019). EfficientNetV2 models are convolutional neural networks designed to jointly optimize model size and training speed. EfficientNetV2 is image native, meaning it classifies each frame separately when generating predictions. It does take into account the relationship between frames in the video.

`time_distributed` combines the EfficientNetV2 architecture with an open-source image object detection model to implement frame selection. The [YOLOX](https://arxiv.org/abs/2107.08430) detection model is run on all frames in a video. Only the frames with the highest probability of detection are then passed to the more computationally intensive EfficientNetV2 for detailed detection and classification.

<!-- https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/efficientnet.py><!-->


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

**`time_distributed` requires that 16 frames be sampled from each video before running inference or training.** This is the default behavior if `time_distributed` is used, but must be added if you are passing in a custom [YAML configuration file](yaml-config.md) or using `zamba` as a Python package.

Running the `time_distributed` model on every frame would be very time consuming. Instead, the Megadetector model is used to select the 16 frames that have the highest probability of detection. `time_distributed` is run only on this subset.

All videos are automatically resized to a resolution of 224x224 pixels.

<a id='slowfast'></a>

## `slowfast` model

### Algorithm

The `slowfast` model was built by re-training a video classification backbone called [SlowFast](https://arxiv.org/abs/1812.03982) (Feichtenhofer, C., Fan, H., Malik, J., & He, K., 2019). SlowFast refers to the two model pathways involved: one that operates at a low frame rate to capture spatial semantics, and one that operatues at a high frame rate to capture motion over time. The basic architectures are deep neural networks using [pytorch](https://pytorch.org/).

<div style="text-align:center;">
<img src="https://s3.amazonaws.com/drivendata-public-assets/zamba-slowfast-diagram.png" alt="Architecture showing the two pathways of the slowfast model" style="width:400px;"/>
<br/><br/>
<i>Source:</i> Feichtenhofer, C., Fan, H., Malik, J., & He, K. (2019). Slowfast networks for video recognition. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 6202-6211).
</div>

Unlike `time_distributed`, `slowfast` is video native. This means it takes into account the relationship between frames in a video, rather than running independently on each frame.

### Training data

The `slowfast` model was trained using the same data as the `time_distributed` model<!-- TODO: add link to time distributed training data section><!-->.

### Default video loading configuration

<!-- TODO: add link to yaml file><!-->

**`time_distributed` requires that 16 frames be sampled from each video before running inference or training.** This is the default behavior if `time_distributed` is used, but must be added if you are passing in a custom [YAML configuration file](yaml-config.md) or using `zamba` as a Python package.

<a id='european'></a>

## `european` model

### Algorithm

The `european` model has the same backbone as the `time_distributed` model, but is trained on data from camera traps in western Europe instead of central and west Africa. 

The `european` model was built by re-training a well-known image classification architecture called [EfficientNetV2](https://arxiv.org/abs/1905.11946) to identify the species in our camera trap videos (Tan, M., & Le, Q., 2019). EfficientNetV2 models are convolutional neural networks designed to jointly optimize model size and training speed. EfficientNetV2 is image native, meaning it classifies each frame separately when generating predictions. It does take into account the relationship between frames in the video.

`european` combines the EfficientNetV2 architecture with an open-source image object detection model to implement frame selection. The [YOLOX](https://arxiv.org/abs/2107.08430) detection model is run on all frames in a video. Only the frames with the highest probability of detection are then passed to the more computationally intensive EfficientNetV2 for detailed detection and classification.

### Training data

`european` was trained using data collected and annotated by partners at [The Max Planck Institute for
Evolutionary Anthropology](https://www.eva.mpg.de/index.html). The data included camera trap videos from Hintenteiche bei Biesenbrow, Germany.

### Default video loading configuration

<!-- TODO: add link to yaml file><!-->

**`european` requires that 16 frames be sampled from each video before running inference or training.** This is the default behavior if `european` is used, but must be added if you are passing in a custom [YAML configuration file](yaml-config.md) or using `zamba` as a Python package.


predict_config:
  data_directory: vids/
  model_name: time_distributed