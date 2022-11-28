# Species detection

The classification algorithms in `zamba` are designed to identify species of animals that appear in camera trap videos. The pretrained models that ship with the `zamba` package are: `blank_nonblank`, `time_distributed`, `slowfast`, and `european`. For more details of each, read on!

## Model summary

<table>
  <tr>
    <th>Model</th>
    <th>Geography</th>
    <th>Relative strengths</th>
    <th>Architecture</th>
    <th>Number of training videos</th>
  </tr>
  <tr>
    <td><code>blank_nonblank</code></td>
    <td>Central Africa, West Africa, and Western Europe</td>
    <td>Just blank detection, without species classification </td>
    <td>Image-based <code>TimeDistributedEfficientNet</code></td>
    <td>~263,000</td>
  </tr>
  <tr>
    <td><code>time_distributed</code></td>
    <td>Central and West Africa</td>
    <td>Recommended species classification model for jungle ecologies</td>
    <td>Image-based <code>TimeDistributedEfficientNet</code></td>
    <td>~250,000</td>
  </tr>
  <tr>
      <td><code>slowfast</code></td>
      <td>Central and West Africa</td>
      <td>Potentially better than <code>time_distributed</code> at small species detection</td>
      <td>Video-native <code>SlowFast</code></td>
    <td>~15,000</td>
    </tr>
  <tr>
    <td><code>european</code></td>
    <td>Western Europe</td>
    <td>Trained on non-jungle ecologies</td>
    <td>Finetuned <code>time_distributed</code>model</td>
    <td>~13,000</td>
  </tr>
</table>

The models trained on the largest datasets took a couple weeks to train on a single GPU machine. Some models will be updated in the future, and you can always check the [changelog](../../changelog) to see if there have been updates.

All models support training, fine-tuning, and inference. For fine-tuning, we recommend using the `time_distributed` model as the starting point.

<h2 id="species-classes"></h2>

## What species can `zamba` detect?

The `blank_nonblank` model is trained to do blank detection, without the species classification. It only outputs the probability that the video is `blank`, meaning that it does not contain an animal.

The `time_distributed` and `slowfast` models are both trained to identify 32 common species from Central and West Africa. The output labels in these models are:

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

The `european` model is trained to identify 11 common species in Western Europe. The possible class labels are:

* `bird`
* `blank`
* `domestic_cat`
* `european_badger`
* `european_beaver`
* `european_hare`
* `european_roe_deer`
* `north_american_raccoon`
* `red_fox`
* `weasel`
* `wild_boar`

<a id='blank-nonblank'></a>

## `blank_nonblank` model

### Architecture

The `blank_nonblank` uses the same [architecture](#time-distributed) as `time_distributed` model, but there is only one output class as this is a binary classification problem.

### Default configuration

The full default configuration is available on [Github](https://github.com/drivendataorg/zamba/blob/master/zamba/models/official_models/blank_nonblank/config.yaml).

The `blank_nonblank` model uses the same [default configuration](#time-distributed-config) as the `time_distributed` model. For the frame selection, an efficient object detection model called [MegadetectorLite](#megadetectorlite) is run on all frames to determine which are the most likely to contain an animal. Then the classification model is run on only the 16 frames with the highest predicted probability of detection.

### Training data

The `blank_nonblank` model was trained on all the data used for the the [`time_distributed`](#time-distributed-training-data) and [`european`](#european-training-data) models.


<a id='time-distributed'></a>

## `time_distributed` model

### Architecture

The `time_distributed` model was built by re-training a well-known image classification architecture called [EfficientNetV2](https://arxiv.org/abs/1905.11946) (Tan, M., & Le, Q., 2019) to identify the species in our camera trap videos. EfficientNetV2 models are convolutional [neural networks](https://www.youtube.com/watch?v=aircAruvnKk&t=995s) designed to jointly optimize model size and training speed. EfficientNetV2 is image native, meaning it classifies each frame separately when generating predictions. The model is wrapped in a [`TimeDistributed` layer](https://docs.fast.ai/layers.html#TimeDistributed) which enables a single prediction per video.

<a id='time-distributed-training-data'></a>

### Training data

The `time_distributed` model was trained using data collected and annotated by trained ecologists from Cameroon, Central African Republic, Democratic Republic of the Congo, Gabon, Guinea, Liberia, Mozambique, Nigeria, Republic of the Congo, Senegal, Tanzania, and Uganda, as well as citizen scientists on the [Chimp&See](https://www.chimpandsee.org/) platform.

The data included camera trap videos from:

<table>
  <tr>
    <th>Country</th>
    <th>Location</th>
  </tr>
  <tr>
    <td rowspan=2>Cameroon</td>
    <td>Campo Ma'an National Park</td>
  </tr>
  <tr>
    <td>Korup National Park</td>
  </tr>
  <tr>
    <td>Central African Republic</td>
    <td>Dzanga-Sangha Protected Area</td>
  </tr>
  <tr>
    <td rowspan=3>Côte d'Ivoire</td>
    <td>Comoé National Park</td>
  </tr>
  <tr>
    <td>Guiroutou</td>
  </tr>
  <tr>
    <td>Taï National Park</td>
  </tr>
  <tr>
    <td rowspan=2>Democratic Republic of the Congo</td>
    <td>Bili-Uele Protect Area</td>
  </tr>
  <tr>
    <td>Salonga National Park</td>
  </tr>
  <tr>
    <td rowspan=2>Gabon</td>
    <td>Loango National Park</td>
  </tr>
  <tr>
    <td>Lopé National Park</td>
  </tr>
  <tr>
    <td rowspan=2>Guinea</td>
    <td>Bakoun Classified Forest</td>
  </tr>
  <tr>
    <td>Moyen-Bafing National Park</td>
  </tr>
  <tr>
    <td rowspan=3>Liberia</td>
    <td>East Nimba Nature Reserve</td>
  </tr>
  <tr>
    <td>Grebo-Krahn National Park</td>
  </tr>
  <tr>
    <td>Sapo National Park</td>
  </tr>
  <tr>
    <td>Mozambique</td>
    <td>Gorongosa National Park</td>
  </tr>
  <tr>
    <td>Nigeria</td>
    <td>Gashaka-Gumti National Park</td>
  </tr>
  <tr>
    <td rowspan=2>Republic of the Congo</td>
    <td>Conkouati-Douli National Park</td>
  </tr>
  <tr>
    <td>Nouabale-Ndoki National Park</td>
  </tr>
  <tr>
    <td>Senegal</td>
    <td>Kayan</td>
  </tr>
  <tr>
    <td rowspan=2>Tanzania</td>
    <td>Grumeti Game Reserve</td>
  </tr>
  <tr>
    <td>Ugalla River National Park</td>
  </tr>
  <tr>
    <td rowspan=3>Uganda</td>
    <td>Budongo Forest Reserve</td>
  </tr>
  <tr>
    <td>Bwindi Forest National Park</td>
  </tr>
  <tr>
    <td>Ngogo and Kibale National Park</td>
  </tr>
</table>

<a id='time-distributed-config'></a>

### Default configuration

The full default configuration is available on [Github](https://github.com/drivendataorg/zamba/blob/master/zamba/models/official_models/time_distributed/config.yaml).

By default, an efficient object detection model called [MegadetectorLite](#megadetectorlite) is run on all frames to determine which are the most likely to contain an animal. Then `time_distributed` is run on only the 16 frames with the highest predicted probability of detection. By default, videos are resized to 240x426 pixels following frame selection.

The default video loading configuration for `time_distributed` is:
```yaml
video_loader_config:
  model_input_height: 240
  model_input_width: 426
  crop_bottom_pixels: 50
  fps: 4
  total_frames: 16
  ensure_total_frames: true
  megadetector_lite_config:
    confidence: 0.25
    fill_mode: score_sorted
    n_frames: 16
    frame_batch_size: 24
    image_height: 640
    image_width: 640
```

You can choose different frame selection methods and vary the size of the images that are used by passing in a custom [YAML configuration file](../yaml-config.md). The only requirement for the `time_distributed` model is that the video loader must return 16 frames.

<a id='slowfast'></a>

## `slowfast` model

### Architecture

The `slowfast` model was built by re-training a video classification backbone called [SlowFast](https://arxiv.org/abs/1812.03982) (Feichtenhofer, C., Fan, H., Malik, J., & He, K., 2019). SlowFast refers to the two model pathways involved: one that operates at a low frame rate to capture spatial semantics, and one that operates at a high frame rate to capture motion over time.

<div style="text-align:center;">
<img src="https://s3.amazonaws.com/drivendata-public-assets/zamba-slowfast-diagram.png" alt="Architecture showing the two pathways of the slowfast model" style="width:400px;"/>
<br/><br/>
<i>Source:</i> Feichtenhofer, C., Fan, H., Malik, J., & He, K. (2019). Slowfast networks for video recognition. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 6202-6211).
</div>

Unlike `time_distributed`, `slowfast` is video native. This means it takes into account the relationship between frames in a video, rather than running independently on each frame.

### Training data

The `slowfast` model was trained on a subset of the [data used](#time-distributed-training-data) for the `time_distributed` model.

### Default configuration

The full default configuration is available on [Github](https://github.com/drivendataorg/zamba/blob/master/zamba/models/official_models/slowfast/config.yaml).

By default, an efficient object detection model called [MegadetectorLite](#megadetectorlite) is run on all frames to determine which are the most likely to contain an animal. Then `slowfast` is run on only the 32 frames with the highest predicted probability of detection. By default, videos are resized to 240x426 pixels.

The full default video loading configuration is:

```yaml
video_loader_config:
  model_input_height: 240
  model_input_width: 426
  crop_bottom_pixels: 50
  fps: 8
  total_frames: 32
  ensure_total_frames: true
  megadetector_lite_config:
    confidence: 0.25
    fill_mode: score_sorted
    n_frames: 32
    image_height: 416
    image_width: 416
```

You can choose different frame selection methods and vary the size of the images that are used by passing in a custom [YAML configuration file](../yaml-config.md). The two requirements for the `slowfast` model are that:

- the video loader must return 32 frames
- videos inputted into the model must be at least 200 x 200 pixels

<a id='european'></a>

## `european` model

### Architecture

The `european` model starts from the a previous version of the `time_distributed` model, and then replaces and trains the final output layer to predict European species.

<a id='european-training-data'></a>

### Training data

The `european` model is finetuned with data collected and annotated by partners at the [German Centre for Integrative Biodiversity Research (iDiv) Halle-Jena-Leipzig](https://www.idiv.de/en/index.html) and [The Max Planck Institute for Evolutionary Anthropology](https://www.eva.mpg.de/index.html). The finetuning data included camera trap videos from Hintenteiche bei Biesenbrow, Germany.

### Default configuration

The full default configuration is available on [Github](https://github.com/drivendataorg/zamba/blob/master/zamba/models/official_models/european/config.yaml).

The `european` model uses the same [default configuration](#time-distributed-config) as the `time_distributed` model. 

As with all models, you can choose different frame selection methods and vary the size of the images that are used by passing in a custom [YAML configuration file](../yaml-config.md). The only requirement for the `european` model is that the video loader must return 16 frames.

<a id='megadetectorlite'></a>

## MegadetectorLite

Frame selection for video models is critical as it would be infeasible to train neural networks on all the frames in a video. For all the species detection models that ship with `zamba`, the default frame selection method is an efficient object detection model called MegadetectorLite that determines the likelihood that each frame contains an animal. Then, only the frames with the highest probability of detection are passed to the model.

MegadetectorLite combines two open-source models:

* [Megadetector](https://github.com/microsoft/CameraTraps/blob/master/megadetector.md) is a pretrained image model designed to detect animals, people, and vehicles in camera trap videos.
* [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) is a high-performance, lightweight object detection model that is much less computationally intensive than Megadetector.

While highly accurate, Megadetector is too computationally intensive to run on every frame. MegadetectorLite was created by training a YOLOX model using the predictions of the Megadetector as ground truth - this method is called [student-teacher training](https://towardsdatascience.com/knowledge-distillation-simplified-dd4973dbc764).

MegadetectorLite can be imported into Python code and used directly since it has convenient methods for `detect_image` and `detect_video`. See [the API documentation for more details](../../api-reference/object-detection-megadetector_lite_yolox/#zamba.object_detection.yolox.megadetector_lite_yolox.MegadetectorLiteYoloX).


## User contributed models

We encourage people to share their custom models trained with Zamba. If you train a model and want to make it available, please add it to the [Model Zoo Wiki](https://github.com/drivendataorg/zamba/wiki) for others to be able to use!

To use one of these models, download the weights file and the configuration file from the Model Zoo Wiki. You'll need to create a [configuration yaml](../yaml-config.md) to use that at least contains the same `video_loader_config` from the configuration yaml you downloaded. Then you can run the model with:

```console
$ zamba predict --checkpoint downloaded_weights.ckpt --config predict_config.yaml
```