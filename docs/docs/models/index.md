# Available Models

The algorithms in `zamba` are designed to identify species of animals that appear in camera trap videos. There are three models that ship with the `zamba` package: `time_distributed`, `slowfast`, and `european`. For more details of each, read on!

## Model summary

<table>
  <tr>
    <th>Model</th>
    <th>Geography</th>
    <th>Relative strengths</th>
    <th>Architecutre</th>
  </tr>
  <tr>
    <td><code>time_distributed</code></td>
    <td>Central and West Africa</td>
    <td>Better than <code>slowfast</code> at duikers, chimps, and gorillas and other larger species</td>
    <td>Image-based <code>TimeDistributedEfficientNet</code></td>
  </tr>
  <tr>
      <td><code>slowfast</code></td>
      <td>Central and West Africa</td>
      <td>Better than <code>time_distributed</code> at blank detection and small species detection</td>
      <td>Video-native <code>SlowFast</code></td>
    </tr>
  <tr>
    <td><code>european</code></td>
    <td>Western Europe</td>
    <td>Trained on non-jungle ecologies</td>
    <td>Finetuned <code>time_distributed</code>model</td>
  </tr>

</table>

All models support training, fine-tuning, and inference. For fine-tuning, we recommend using the `time_distributed` model as the starting point.

<h2 id="species-classes"></h2>

## What species can `zamba` detect?

`time_distributed` and `slowfast` are both trained to identify 32 common species from Central and West Africa. The output labels in these models are:

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

The `time_distributed` model was built by re-training a well-known image classification architecture called [EfficientNetV2](https://arxiv.org/abs/1905.11946) (Tan, M., & Le, Q., 2019) to identify the species in our camera trap videos. EfficientNetV2 models are convolutional [neural networks](https://www.youtube.com/watch?v=aircAruvnKk&t=995s) designed to jointly optimize model size and training speed. EfficientNetV2 is image native, meaning it classifies each frame separately when generating predictions. The model is wrapped in a [`TimeDistributed` layer](https://docs.fast.ai/layers.html#TimeDistributed) which enables a single prediction per video.

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
* Bili-Uere'
* Budongo'
* Bwindi'
* Campo Ma'an National Park
* Conkouati'
* Guiroutou'
* TRS_Bakoun'
* Gashaka-Gumti National Park
* TRS_Grebo'
* Comoe National Park'
* Kayan'
* Korup National Park'
* Loango'
* Ngogo'
* East Nimba'
* Sapo'
* Ugalla'

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
```

You can choose different frame selection methods and vary the size of the images that are used by passing in a custom [YAML configuration file](../yaml-config.md). The only requirement for the `time_distributed` model is that the video loader must return 16 frames.

<a id='slowfast'></a>

## `slowfast` model

### Algorithm

The `slowfast` model was built by re-training a video classification backbone called [SlowFast](https://arxiv.org/abs/1812.03982) (Feichtenhofer, C., Fan, H., Malik, J., & He, K., 2019). SlowFast refers to the two model pathways involved: one that operates at a low frame rate to capture spatial semantics, and one that operates at a high frame rate to capture motion over time.

<div style="text-align:center;">
<img src="https://s3.amazonaws.com/drivendata-public-assets/zamba-slowfast-diagram.png" alt="Architecture showing the two pathways of the slowfast model" style="width:400px;"/>
<br/><br/>
<i>Source:</i> Feichtenhofer, C., Fan, H., Malik, J., & He, K. (2019). Slowfast networks for video recognition. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 6202-6211).
</div>

Unlike `time_distributed`, `slowfast` is video native. This means it takes into account the relationship between frames in a video, rather than running independently on each frame.

### Training data

The `slowfast` model was trained using the same data as the [`time_distributed` model](#time-distributed-training-data).

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
```

You can choose different frame selection methods and vary the size of the images that are used by passing in a custom [YAML configuration file](../yaml-config.md). The two requirements for the `slowfast` model are that:
- the video loader must return 32 frames.
- videos inputted into the model must be at least 200 x 200 pixels

<a id='european'></a>

## `european` model

### Algorithm

The `european` model starts from the trained `time_distributed` model, and then replaces and trains the final output layer to predict European species.

### Training data

The `european` model is finetuned with data collected and annotated by partners at [The Max Planck Institute for Evolutionary Anthropology](https://www.eva.mpg.de/index.html). The finetuning data included camera trap videos from Hintenteiche bei Biesenbrow, Germany.

### Default configuration

The full default configuration is available on [Github](https://github.com/drivendataorg/zamba/blob/master/zamba/models/official_models/european/config.yaml).

The `european` model uses the same frame selection as the `time_distributed` model. By default, an efficient object detection model called [MegadetectorLite](#megadetectorlite) is run on all frames to determine which are the most likely to contain an animal. Then `european` is run on only the 16 frames with the highest predicted probability of detection. By default, videos are resized to 240x426 pixels following frame selection.

The full default video loading configuration is:
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
```

As with all models, you can choose different frame selection methods and vary the size of the images that are used by passing in a custom [YAML configuration file](../yaml-config.md). The only requirement for the `european` model is that the video loader must return 16 frames.

<a id='megadetectorlite'></a>

## MegadetectorLite

Running any of the three models that ship with `zamba` on all frames of a video would be incredibly time consuming and computationally intensive. Instead, `zamba` uses a more efficient object detection model called MegadetectorLite to determine the likelihood that each frame contains an animal. Then, only the frames with the highest probability of detection can be passed to the model.

MegadetectorLite combines two open-source models:

* [Megadetector](https://github.com/microsoft/CameraTraps/blob/master/megadetector.md) is a pretrained image model designed to detect animals, people, and vehicles in camera trap videos.
* [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) is a high-performance, lightweight object detection model that is much less computationally intensive than Megadetector.

While highly accurate, Megadetector is too computationally intensive to run on every frame. MegadetectorLite was created by training a YOLOX model using the predictions of the Megadetector as ground truth - this method is called [student-teacher training](https://towardsdatascience.com/knowledge-distillation-simplified-dd4973dbc764).


## Densepose

Facebook AI Research has published a model, DensePose ([Neverova et al, 2021](https://arxiv.org/abs/2011.12438v1)), which can be used to get segmentations for animals that appear in videos. This was trained on the following animals, but often works for other species as well: sheep, zebra, horse, giraffe, elephant, cow, ear, cat, dog. Here's an example of the segmentation output for a frame:

![segmentation of duiker](../media/seg_out.jpg)

Additionally, the model provides mapping of the segmentation output to specific anatomy for chimpanzees. This can be helpful for determining the orientation of chimpanzees in videos and for their behaviors. Here is an example of what that output looks like:

![chimpanzee texture output](../media/texture_out.png)

For more information on the algorithms and outputs of the DensePose model, see the [Facebook DensePose Github Repository](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose).

The Zamba package supports running Densepose on videos to generate three types of outputs:

 - A `.json` file with details of segmentations per video frame.
 - A `.mp4` file where the original video has the segmentation rendered on top of animal so that the output can be vsiually inspected.
 - A `.csv` (when `--output-type chimp_anatomy`) that contains the height and width of the bounding box around each chimpanzee, the frame number and timestamp of the observation, and the percentage of pixels in the bounding box that correspond with each anatomical part.

Generally, running the densepose model is computationally intensive. It is recommended to run the model at a relatively low framerate (e.g., 1 frame per second) to generate outputs for a video. Another caveat is that because the output JSON output contains the full embedding, these files can be quite large. These are not written out by default.

In order to use the densepose model, you must have PyTorch already installed on your system, and then you must install the `densepose` extra:

```bash
pip install torch  # see https://pytorch.org/get-started/locally/
pip install "zamba[densepose]"
```

Once that is done, here's how to run the DensePose model:

=== "CLI"
    ```bash
    # create a segmentation output video for each input video in PATH_TO_VIDEOS
    zamba densepose --data-dir PATH_TO_VIDEOS --render-output
    ```
=== "Python"
    ```python
    from zamba.models.densepose import DensePoseConfig
    densepose_conf = DensePoseConfig(data_directory="PATH_TO_VIDEOS", render_output=True)
    densepose_conf.run_model()
    ```


<video controls>
  <source src="../media/densepose_zamba_vid.mp4" type="video/mp4">
</videp>

To see all of the available options, run `zamba densepose --help`.
