# Video species classification

`zamba` ships with three models for classifying the species in camera trap **videos**:

* **[`time_distributed`](#time-distributed)** (default): the recommended species classification model for Central and West African (jungle) ecologies.
* **[`slowfast`](#slowfast)**: a video-native model that may do better than `time_distributed` at detecting small species.
* **[`european`](#european)**: a `time_distributed` model finetuned for Western European species.

If you only want to separate blank videos from videos containing animals (without species classification), see the [blank detection](blank-detection.md) page. All of these models use [MegadetectorLite](megadetector.md) for frame selection before classification.

<a id='species-classes'></a>

## What species can `zamba` detect?

The `time_distributed` and `slowfast` models are both trained to identify 32 common species from Central and West Africa. The output labels in these models are:

<details class="zamba-classlist" markdown>
<summary>Show all 32 <code>time_distributed</code> / <code>slowfast</code> classes</summary>
<div class="scroll-box" markdown>

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

</div>
</details>

The `european` model is trained to identify 11 common species in Western Europe. The possible class labels are:

<details class="zamba-classlist" markdown>
<summary>Show all 11 <code>european</code> classes</summary>
<div class="scroll-box" markdown>

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

</div>
</details>

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

By default, an efficient object detection model called [MegadetectorLite](megadetector.md) is run on all frames to determine which are the most likely to contain an animal. Then `time_distributed` is run on only the 16 frames with the highest predicted probability of detection. By default, videos are resized to 240x426 pixels following frame selection.

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

<a id='time-distributed-performance'></a>

### Performance

The African species `time_distributed` model was trained using almost **250,000 videos from 14 countries** in West, Central, and East Africa.
These videos include examples of 30 animal species, plus some blank videos and some showing humans.
To evaluate the performance of the model, we held out 30,324 videos from 101 randomly-chosen sites.

#### Removing blank videos

One use of this model is to identify blank videos so they can be discarded or ignored.
In this dataset, 42% of the videos are blank, so removing them can save substantial amounts of
viewing time and storage space.

The model assigns a probability that each video is blank, so one strategy is to remove videos
if their probability exceeds a given threshold.
Of course, the model is not perfect, so there is a chance we will wrongly remove a video that actually
contains an animal.

To assess this tradeoff, we can use the holdout set to simulate this strategy with a range of thresholds.
For each threshold, we compute the fraction of blank videos correctly discarded and the fraction of non-blank
videos incorrectly discarded.
The following figure shows the results.

<img src="https://s3.amazonaws.com/drivendata-public-assets/zamba/td_full_set_recall_recall_curve.png" alt="" style="width:600px;"/>

The markers indicate three levels of tolerance for losing non-blank videos. For example, if it's acceptable to
lose 5% of non-blank videos, we can choose a threshold that removes 63% of the blank videos.
If we can tolerate a loss of 10%, we can remove 80% of the blanks.
And if we can tolerate a loss of 15%, we can remove 90% of the blanks.
Above that, the percentage of lost videos increases steeply.

#### Accuracy

In addition to identifying blank videos, the model also computes a probability that each of 30 animal species appears in each video (plus human and blank).
We can use these probabilities to quantify the accuracy of the model for species classification.
Specifically, we computed:

* Top-1 accuracy, which is the fraction of videos where the species with the highest predicted probability is, in fact, present.

* Top-3 accuracy, which is the fraction of videos where one of the three species the model considered most likely is present.

Over all videos in the holdout set, the **top-1 accuracy is 82%; the top-3 accuracy is 94%**.
As an example, if you choose a video at random and the species with the highest predicted probability is elephant, the probability is 82% that the video contains an elephant, according to the human-generated labels.
If the three most likely species were elephant, hippopotamus, and cattle, the probability is 94% that the video
contains at least one of those species.

These results depend in part on the species represented in a particular dataset. For example, in the small
number of videos from Equatorial Guinea, only three species appear. For these videos, the top-1 accuracy is 97%, much
higher than the overall accuracy.
In the videos from Ivory Coast, 21 species are represented, so the problem is harder. For these
videos, top-1 accuracy is 80%, a little lower than the overall accuracy.

#### Recall and precision by species

One of the goals of classification is to help with retrieval, that is, efficiently finding videos containing
a particular species. To evaluate the performance of the model for retrieval, we can use

* Recall, which is the fraction of videos containing a particular species that are correctly classified, and

* Precision, which is the fraction of videos the model labels with a particular species that actually contain that species.

The following figure shows recall and precision for the species in the holdout set,
excluding 11 species where there are too few examples in the holdout set to compute meaningful estimates of these metrics.

<img src="https://s3.amazonaws.com/drivendata-public-assets/zamba/td_full_set_precision_recall_by_species.png" alt="" style="width:800px;"/>

It's clear that we are able to retrieve some species more efficiently than others. For example, elephants are relatively
easy to find. Of the videos that contain elephants, 84% are correctly classified; and of the videos that the model
labels "elephant", 94% contain elephants.
So a researcher using the model to find elephant videos could find a large majority of them while viewing only a
small number of non-elephant videos.

Not surprisingly, smaller animals are harder to find. For example, the recall for rodent videos is only 22%.
However, it is still possible to search for rodents by selecting videos that assign a relatively high probability
to "rodent", even if it assigns a higher probability to another species or "blank".

<details class="zamba-classlist" markdown>
<summary>Description of the holdout set</summary>
<div markdown>

The videos in the holdout set are a random sample from the complete set of labeled videos, but they are
selected on a transect-by-transect basis; that is, videos from each transect are assigned entirely to the
training set or entirely to the holdout set.
So the performance of the model on the holdout set should reflect its performance on videos from a transect
the model has never seen.

All 14 countries are represented in the holdout set; the following table shows the number of videos from
each country.
These proportions are roughly consistent with the proportions in the complete set.

<table>
  <thead>
    <tr>
      <th>Country</th>
      <th>Number of videos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ivory Coast</th>
      <td>10,987</td>
    </tr>
    <tr>
      <th>Guinea</th>
      <td>4,300</td>
    </tr>
    <tr>
      <th>DR Congo</th>
      <td>2,750</td>
    </tr>
    <tr>
      <th>Uganda</th>
      <td>2,497</td>
    </tr>
    <tr>
      <th>Tanzania</th>
      <td>1,794</td>
    </tr>
    <tr>
      <th>Mozambique</th>
      <td>1,168</td>
    </tr>
    <tr>
      <th>Senegal</th>
      <td>1,131</td>
    </tr>
    <tr>
      <th>Gabon</th>
      <td>1,116</td>
    </tr>
    <tr>
      <th>Cameroon</th>
      <td>1,114</td>
    </tr>
    <tr>
      <th>Liberia</th>
      <td>1,065</td>
    </tr>
    <tr>
      <th>Central African Republic</th>
      <td>997</td>
    </tr>
    <tr>
      <th>Nigeria</th>
      <td>889</td>
    </tr>
    <tr>
      <th>Congo Republic</th>
      <td>678</td>
    </tr>
    <tr>
      <th>Equatorial Guinea</th>
      <td>38</td>
    </tr>
  </tbody>
</table>

The following figure shows the number of videos containing each of 30 animal species, plus some videos showing humans and
a substantial number of blank videos.

<img src="https://s3.amazonaws.com/drivendata-public-assets/zamba/td_full_set_number_videos_by_species.png" alt="" style="width:800px;"/>

One of the challenges of this kind of classification is that some species are
much more common than others. For species that appear in a small number of videos, we expect
the model to be less accurate because it has fewer examples to learn from.
Also, for these species it is hard to assess performance precisely because there are few examples in the holdout set. If you would like to add more examples of the species you work with, see [how to build on the model](../train-tutorial.md).

</div>
</details>

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

By default, an efficient object detection model called [MegadetectorLite](megadetector.md) is run on all frames to determine which are the most likely to contain an animal. Then `slowfast` is run on only the 32 frames with the highest predicted probability of detection. By default, videos are resized to 240x426 pixels.

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
