# User tutorial: Classifying unlabeled images

This section walks through how to classify images using `zamba`. If you are new to `zamba` and just want to classify some images as soon as possible, see the [Quickstart](quickstart.md) guide.

This tutorial goes over the steps for using `zamba` if:

* You already have `zamba` installed (for details see the [Installation](install.md) page)
* You have unlabeled images that you want to generate labels for
* The possible class species labels for your images are included in the list of possible [zamba labels](models/species-detection.md#species-classes). If your species are not included in this list, you can [retrain a model](train-tutorial.md) using your own labeled data and then run inference.

## Basic usage: command line interface

Say that we want to classify the images in a folder called `example_images` as simply as possible using all of the default settings.

Minimum example for prediction in the command line:

```console
$ zamba image predict --data-dir example_images/
```

### Required arguments

To run `zamba images predict` in the command line, you must specify `--data-dir` and/or `--filepaths`.

* **`--data-dir PATH`:** Path to the folder containing your images. If you don't also provide `filepaths`, Zamba will recursively search this folder for images.
* **`--filepaths PATH`:** Path to a CSV file with a column for the filepath to each video you want to classify. The CSV must have a column for `filepath`. Filepaths can be absolute on your system or relative to the data directory that your provide in `--data-dir`.

All other flags are optional. To choose the model you want to use for prediction, either `--model` or `--checkpoint` must be specified. Use `--model` to specify one of the [pretrained models](models/species-detection.md) that ship with `zamba`. Use `--checkpoint` to run inference with a locally saved model. `--model` defaults to [`lila.science`](models/species-detection.md#what-species-can-zamba-detect).

## Basic usage: Python package

We also support using Zamba as a Python package. Say that we want to classify the images in a folder called `example_images` as simply as possible using all of the default settings.

Minimum example for prediction using the Python package:

```python
from zamba.images.manager import predict
from zamba.images.config import ImageClassificationPredictConfig

predict_config = ImageClassificationPredictConfig(data_dir="example_images/")
predict(config=predict_config)
```

The only argument that can be passed to `predict` is `config`, so the only step is to instantiate an [`ImageClassificationPredictConfig`](configurations.md#prediction-arguments) and pass it to `predict`.

### Required arguments

To run `predict` in Python, you must specify either `data_dir` or `filepaths` when `PredictConfig` is instantiated.

* **`data_dir (DirectoryPath)`:** Path to the folder containing your images. If you don't also provide `filepaths`, Zamba will recursively search this folder for images.

* **`filepaths (FilePath)`:** Path to a CSV file with a column for the filepath to each image you want to classify. The CSV must have a column for `filepath`. Filepaths can be absolute or relative to the data directory provided as `data_dir`.

For detailed explanations of all possible configuration arguments, see [All Optional Arguments](configurations.md).

## Default behavior

By default, the [`lila.science`](models/species-detection.md#lila.science) model will be used. `zamba` will output a `.csv` file with a row for each bounding box identified by megadetector (there may be multiple bounding boxes per image) and columns for each class (ie. species / species group). A cell in the CSV (i,j) can be interpreted as *the predicted likelihood that the animal present in bounding box i is of species group j.* Usually, most users want to just look at the top species prediction for each bounding box.

By default, predictions will be saved to a file called `zamba_predictions.csv` in your working directory. You can save predictions to a custom directory using the `--save-dir` argument.

```console
$ cat zamba_predictions.csv
filepath,detection_category,detection_conf,x1,y1,x2,y2,species_acinonyx_jubatus,species_aepyceros_melampus,species_alcelaphus_buselaphus...
1.jpg,1,0.85,2015,1235,2448,1544,4.924246e-06,0.0001539439,9.6043495e-06...
2.jpg,1,0.921,527,1246,1805,1501,5.061601e-05,3.6830465e-05,2.4510617e-05...
3.jpg,1,0.9,1422,528,1629,816,1.1791806e-06,4.080566e-06,3.4533906e-07...
```

The `detection_category` and `detection_conf` come from [MegaDetector](https://github.com/agentmorris/MegaDetector) which we use to find bounding boxes around individual animals in images. A `detection_category` of `"1"` indicates the presence of an animal, and `detection_conf` is the confidence the animal classifier had in the presence of an animal. The columns `x1`, `y1`, `x2`, `y2` indicate the coordinates of the top-left and bottom-right corners of the bounding box relative to the top-left corner of the image. The remaining columns are the scores assigned to each species for the individual animal in the given bounding box.


## Step-by-step tutorial

### 1. Specify the path to your images

Save all of your images within one parent folder.

* Images can be in nested subdirectories within the folder.
* Your images should be in be saved in formats that are suppored by Python's [`pillow`](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#fully-supported-formats) library. Any images that fail a set of validation checks will be skipped during inference or training. By default, `zamba` will look for files with the following suffixes: `.jpg`, `.jpeg`, `.png` and `.webp`. To use other image formats that are supported by pillow, set your `IMAGE_SUFFIXES` environment variable.

Add the path to your image folder to the command-line command. For example, if your images are in a folder called `example_images`:

=== "CLI"
    ```console
    $ zamba image predict --data-dir example_images/
    ```
=== "Python"
    ```python
    from zamba.images.manager import predict
    from zamba.images.config import ImageClassificationPredictConfig

    predict_config = ImageClassificationPredictConfig(data_dir="example_images/")
    predict(config=predict_config)
    ```

### 2. Choose a model for prediction

Right now, Zamba supports only a single model out-of-the-box for images: `lila.science`. While there's only a single model, this model has been trained to detect hundreds of species, so it's a great first pass. The `lila.science` model will be used if no model is specified.

If you've fine-tuned a model, you can select that model instead of a built-in model by using the `--checkpoint` argument. For example:

=== "CLI"
    ```console
    $ zamba predict --data-dir example_images/ --checkpoint zamba-image-classification-dummy_modelepoch=00-val_loss=48.372.ckpt
    ```
=== "Python"
    ```python
    from zamba.images.manager import predict
    from zamba.images.config import ImageClassificationPredictConfig

    predict_config = ImageClassificationPredictConfig(
        data_dir="example_images/",
        checkpoint="zamba-image-classification-dummy_modelepoch=00-val_loss=48.372.ckpt")
    predict(config=predict_config)
    ```

### 3. Choose the output format

There are two options for how to format predictions:

1. **CSV (default):** Return predictions with a row for each bounding box-filename combination and a column for each class label, with probabilities between 0 and 1. Bounding boxes are automatically created for each image, and each image can contain 0 or more bounding boxes (for example, if there are many rabbits in a single image). Cell `(i,j)` is the probability that species `j` is present in bounding box `i`.
2. **MegaDetector:** Returns much the same information as above, but in a [JSON format](https://lila.science/megadetector-output-format) similar to the [COCO image format](https://lila.science/coco-camera-traps) that's been augmented to be more useful for camera trap data.

Say we want to generate predictions for images in `example_images` in MegaDetector format:

=== "CLI"
    ```console
    $ zamba image predict --data-dir example_image/ --results-file-format megadetector
    $ cat zamba_predictions.json
    {
        "info": {},
        "detection_categories": { "1": "animal", "2": "person", "3": "vehicle" },
        "classification_categories": {
            "0": "species_acinonyx_jubatus",
            "1": "species_aepyceros_melampus",
            "2": "species_alcelaphus_buselaphus",
    ...
    },
    "images": [ {
        "file": "1.jpg",
        "detections": [
            { "category": "1", "conf": 0.85, "bbox": [ 0.7, 0.2, 0.12, 0.4 ],
                "classifications": [ [ 88, 0.9355112910270691 ],
    ...
    ```
=== "Python"
    ```python
    from zamba.images.manager import predict
    from zamba.images.config import ImageClassificationPredictConfig, ResultsFormat

    predict_config = ImageClassificationPredictConfig(
        data_dir="example_images/",
        results_file_format=ResultsFormat.MEGADETECTOR
    ```

### 4. Specify any additional parameters

And there's so much more! You can also do things like specify your region for faster model download (`--weight-download-region`), change the detection threshold above which we'll consider a bounding box contains an animal (`--detections-threshold`), or specify a different folder where your predictions should be saved (`--save-dir`). To read about a few common considerations, see the [Guide to Common Optional Parameters](extra-options.md) page.
