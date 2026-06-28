# Available models

The algorithms in `zamba` are designed to detect and classify animals that appear in camera trap **images** and **videos**. This section is organized first by the type of data you have (image vs. video) and then by the task you want to perform.

## Which model should I use?

| Data type | Task | Models |
| --- | --- | --- |
| **Images** | [Species classification](image-classification.md) | [`lila.science`](image-classification.md#lila.science) (default), [`speciesnet`](image-classification.md#speciesnet) |
| **Videos** | [Species classification](video-classification.md) | [`time_distributed`](video-classification.md#time-distributed) (default), [`slowfast`](video-classification.md#slowfast), [`european`](video-classification.md#european) |
| **Videos** | [Blank detection](blank-detection.md) | [`blank_nonblank`](blank-detection.md) |
| **Videos** | [Depth estimation](depth.md) | depth |
| **Videos** | [Pose estimation](densepose.md) | DensePose |

All of the species classification models support training, fine-tuning, and inference. For fine-tuning video models, we recommend starting from `time_distributed`; for image models, we recommend starting from `lila.science`.

All of the image and video species classification models share a common first step: an efficient object detector ([MegaDetector / MegadetectorLite](megadetector.md)) is used to find the animals in a frame before classification.

## Model summaries

### Image models

<table>
  <tr>
    <th>Model</th>
    <th>Geography</th>
    <th>Relative strengths</th>
    <th>Architecture</th>
    <th>Number of training images</th>
  </tr>
  <tr>
    <td><code>lila.science</code></td>
    <td>Global based on datasets from lila.science</td>
    <td>Good base model for common global species. Default image model.</td>
    <td>ConvNextV2 backbone</td>
    <td>15 million annotations from 7 million images</td>
  </tr>
  <tr>
    <td><code>speciesnet</code></td>
    <td>Global</td>
    <td>Very large taxonomy (2,000+ classes); strong alternative starting point for fine-tuning on global species.</td>
    <td>EfficientNetV2-M backbone</td>
    <td>Google's SpeciesNet training set (tens of millions of images)</td>
  </tr>
</table>

See the [image species classification](image-classification.md) page for details.

### Video models

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

See the [video species classification](video-classification.md) and [blank detection](blank-detection.md) pages for details.

The models trained on the largest datasets took a couple weeks to train on a single GPU machine. Some models will be updated in the future, and you can always check the [changelog](../changelog) to see if there have been updates.

## User contributed models

We encourage people to share their custom models trained with Zamba. If you train a model and want to make it available, please add it to the [Model Zoo Wiki](https://github.com/drivendataorg/zamba/wiki) for others to be able to use!

To use one of these models, download the weights file and the configuration file from the Model Zoo Wiki. You'll need to create a [configuration yaml](../yaml-config.md) to use that at least contains the same `video_loader_config` from the configuration yaml you downloaded. Then you can run the model with:

```console
$ zamba predict --checkpoint downloaded_weights.ckpt --config predict_config.yaml
```
