# User tutorial: Training a model on labeled images

This section walks through how to train an image classification model using `zamba`. If you are new to `zamba` and just want to classify some images as soon as possible, see the [Quickstart](quickstart.md) guide.

This tutorial goes over the steps for using `zamba` if:

* You already have `zamba` installed (for details see the [Installation](install.md) page)
* You have labeled images that you want to use to train or finetune a model

`zamba` can run two types of model training:

* Finetuning a model with labels that are a subset of the possible [zamba labels](models/species-detection.md#species-classes)
* Finetuning a model to predict an entirely new set of labels

The process is the same for both cases.

## Basic usage: command line interface

By default, the [`lila.science`](models/species-detection.md#lila.science) image classification model is used. Say that we want to finetune that model based on the images in `example_images` and the labels in `example_labels.csv`.

```console
$ cat example_labels.csv

filepath,label
elephant_001.jpg,elephant
leopard_002.jpg,leopard
blank_003.jpg,blank
chimp_004.jpg,chimpanzee_bonobo
```

Training at the command line would look like:

```console
$ zamba image train --data-dir example_images/ --labels example_labels.csv
```

### Required arguments

To run `zamba image train` in the command line, you must specify `labels` and `data_dir`.

* **`--labels PATH`:** Path to a CSV or JSON file containing the image labels to use as ground truth during training. For CSV files, there must be columns for both `filepath` and `label`. Optionally, there can also be columns for `split` (which can have one of the three values for each row: `train`, `val`, or `test`) or `site` (which can contain any string identifying the location of the camera, used to allocate images to splits if not already specified). For JSON files, the format should be COCO or another supported bounding box format as specified by `--labels-format`.

* **`--data-dir PATH`:** Path to the folder containing your labeled images. If the image filepaths in the labels csv are not absolute, be sure to provide the `data-dir` to which the filepaths are relative.

## Basic usage: Python package

To do the same thing as above using the library code, this would look like:

```python
from zamba.images.manager import train
from zamba.images.config import ImageClassificationTrainingConfig

train_config = ImageClassificationTrainingConfig(
    data_dir="example_images/", labels="example_labels.csv"
)
train(config=train_config)
```

The only argument that can be passed to `train` is `config`. The first step is to instantiate [`ImageClassificationTrainingConfig`](configurations.md#training-arguments).

You'll want to go over the documentation to familiarize yourself with the options in the configuration since what you choose can have a large impact on the results of your model. We've tried to include in the documentation sane defaults and recommendations for how to set these parameters. For detailed explanations of all possible configuration arguments, see [All Configuration Options](configurations.md).

## Model output classes

The classes your trained model will predict are determined by which model you choose and whether the species in your labels are a subset of that model's [default labels](models/species-detection.md#species-classes). This table outlines the default behavior for a set of common scenarios.

| Classes in labels csv | Model | What we infer | Classes trained model predicts |
| --- | --- | --- | --- |
| cat, blank | `lila.science` | multiclass but not a subset of the zamba labels | cat, blank |
| elephant, antelope_duiker, blank | `lila.science` | multiclass and a subset of the zamba labels | all zamba species (unless `use_default_model_labels=False`) |
| zebra, grizzly, blank | `lila.science` | multiclass but not a subset of the zamba labels | zebra, grizzly, blank |

## Step-by-step tutorial

### 1. Specify the path to your images

Save all of your images in a folder.

* They can be in nested directories within the folder.
* Your images should all be saved in formats that are supported by Python's [`pillow`](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#fully-supported-formats) library. Any images that fail a set of validation checks will be skipped during inference or training. By default, `zamba` will look for files with the following suffixes: `.jpg`, `.jpeg`, `.png` and `.webp`. To use other image formats that are supported by pillow, set your `IMAGE_SUFFIXES` environment variable.

Add the path to your image folder with `--data-dir`. For example, if your images are in a folder called `example_images`, add `--data-dir example_images/` to your command.

=== "CLI"
    ```console
    $ zamba image train --data-dir example_images/
    ```

=== "Python"
    ```python
    from zamba.images.config import ImageClassificationTrainingConfig
    from zamba.images.manager import train

    train_config = ImageClassificationTrainingConfig(data_dir='example_images/')
    train(config=train_config)
    ```
Note that the above will not run yet because labels are not specified.

The more training data you have, the better the resulting model will be. We recommend having a minimum of 100 images **per species**. Having an imbalanced dataset - for example, where most of the images are blank - is okay as long as there are enough examples of each individual species.

### 2. Specify your labels

Your labels should be saved in a `.csv` file with columns for filepath and label. For example:

```console
$ cat example_labels.csv
filepath,label
elephant_001.jpg,elephant
leopard_002.jpg,leopard
blank_003.jpg,blank
chimp_004.jpg,chimpanzee_bonobo
```

Add the path to your labels with `--labels`.  For example, if your images are in a folder called `example_images` and your labels are saved in `example_labels.csv`:

=== "CLI"
    ```console
    $ zamba image train --data-dir example_images/ --labels example_labels.csv
    ```
=== "Python"
    In Python, the labels are passed in when `ImageClassificationTrainingConfig` is instantiated. The Python package allows you to pass in labels as either a file path or a pandas dataframe:
    ```python
    import pandas as pd
    from zamba.images.config import ImageClassificationTrainingConfig
    from zamba.images.manager import train

    labels_dataframe = pd.read_csv('example_labels.csv')
    train_config = ImageClassificationTrainingConfig(
        data_dir='example_images/', labels=labels_dataframe
    )
    train(config=train_config)
    ```

#### Labels `zamba` has seen before

Your labels may be included in the list of [`zamba` class labels](models/species-detection.md#species-classes) that the provided models are trained to predict. If so, the relevant model that ships with `zamba` will essentially be used as a checkpoint, and model training will resume from that checkpoint.

By default, the model you train will continue to output all of the Zamba class labels, not just the ones in your dataset. For different behavior, see [`use_default_model_labels`](configurations.md#use_default_model_labels-bool-optional).

#### Completely new labels

You can also train a model to predict completely new labels - the world is your oyster! (We'd love to see a model trained to predict oysters.) If this is the case, the model architecture will replace the final [neural network](https://www.youtube.com/watch?v=aircAruvnKk&t=995s) layer with a new head that predicts *your* labels instead of those that ship with `zamba`.

You can then make your model available to others by adding it to the [Model Zoo on our wiki](https://github.com/drivendataorg/zamba/wiki).

#### Labels with bounding boxes

If your labels include bounding box annotations (e.g., in COCO format), `zamba` can use these to crop images before training. This focuses the model on regions of interest. To use bounding boxes from your labels file, ensure your labels are in a supported format (COCO by default) and set `--labels-format` if needed:

=== "CLI"
    ```console
    $ zamba image train --data-dir example_images/ --labels example_labels.json --labels-format coco
    ```
=== "Python"
    ```python
    from zamba.images.config import ImageClassificationTrainingConfig, BboxInputFormat
    from zamba.images.manager import train

    train_config = ImageClassificationTrainingConfig(
        data_dir='example_images/',
        labels='example_labels.json',
        labels_format=BboxInputFormat.COCO
    )
    train(config=train_config)
    ```

### 3. Choose a model for training

Any of the image models that ship with `zamba` can be trained. Currently, `zamba` supports two image models:

* **`lila.science`** (default): Trained on 178 species from around the world. This is the recommended model for most use cases.
* **`speciesnet`**: An alternative model architecture that may work better for certain datasets.

If you're training on entirely new species or new ecologies, we recommend starting with the [`lila.science` model](models/species-detection.md#lila.science) as it has been trained on a diverse set of species.

Add the model name to your command with `--model`. The `lila.science` model will be used if no model is specified. For example, if you want to train the `speciesnet` model:

=== "CLI"
    ```console
    $ zamba image train --data-dir example_images/ --labels example_labels.csv --model speciesnet
    ```
=== "Python"
    ```python
    train_config = ImageClassificationTrainingConfig(
        data_dir="example_images/",
        labels="example_labels.csv",
        model_name="speciesnet",
    )
    train(config=train_config)
    ```

### 4. Training from scratch vs. finetuning

By default, `zamba` will finetune from a pretrained model checkpoint. If you want to train a model from scratch (starting with only base ImageNet weights), use the `--from-scratch` flag:

=== "CLI"
    ```console
    $ zamba image train --data-dir example_images/ --labels example_labels.csv --from-scratch
    ```
=== "Python"
    ```python
    train_config = ImageClassificationTrainingConfig(
        data_dir="example_images/",
        labels="example_labels.csv",
        from_scratch=True,
    )
    train(config=train_config)
    ```

### 5. Specify any additional parameters

And there's so much more! You can also do things like:

* Specify your region for faster model download (`--weight-download-region`)
* Start training from a saved model checkpoint (`--checkpoint`)
* Specify a different path where your model should be saved (`--save-dir`)
* Adjust learning rate (`--lr`) or let `zamba` find an optimal learning rate automatically
* Use weighted loss for imbalanced datasets (`--weighted-loss`)
* Enable extra data augmentations (`--extra-train-augmentations`)
* Disable image cropping if your images are already cropped (`--no-crop-images`)

To read about a few common considerations, see the [Guide to Common Optional Parameters](extra-options.md) page.

### 6. Test your configuration with a dry run

Before kicking off the full model training, we recommend testing your code with a "dry run". This will run one training and validation batch for one epoch to quickly detect any bugs. See the [Debugging](debugging.md) page for details.

## Files that get written out during training

You can specify where the outputs should be saved with `--save-dir`. If no save directory is specified, `zamba` will write out files to your current working directory. For example, a model finetuned from the provided `lila.science` model (the default) will save outputs to the current directory.

The training outputs include:

* `train_configuration.yaml`: The full model configuration used to generate the given model, including all training parameters. To continue training using the same configuration, or to train another model using the same configuration, you can pass in `train_configuration.yaml` (see [Specifying Model Configurations with a YAML File](yaml-config.md)) along with the `labels` filepath.
* `hparams.yaml`: Model hyperparameters. These are included in the checkpoint file as well.
* `lila.science.ckpt` (or `{model_name}.ckpt`): Model checkpoint. You can continue training from this checkpoint by passing it to `zamba image train` with the `--checkpoint` flag:
    ```console
    $ zamba image train --checkpoint lila.science.ckpt --data-dir example_images/ --labels example_labels.csv
    ```
* `val_metrics.json`: The model's performance on the validation subset
* `test_metrics.json`: The model's performance on the test (holdout) subset (if a test split was created)
* `splits.csv`: Which files were used for training, validation, and as a holdout set. If split is specified in the labels file passed to training, `splits.csv` will not be saved out.
* `training.log`: A log file containing training progress and information (if `--save-dir` is specified)
* MLflow logs: Training metrics and model artifacts are logged to MLflow (by default in a local `./mlruns` directory). You can view these with:
    ```console
    $ mlflow ui
    ```

## Using your trained model

Once training is complete, you can use your trained model to make predictions on new images:

=== "CLI"
    ```console
    $ zamba image predict --data-dir new_images/ --checkpoint lila.science.ckpt
    ```
=== "Python"
    ```python
    from zamba.images.manager import predict
    from zamba.images.config import ImageClassificationPredictConfig

    predict_config = ImageClassificationPredictConfig(
        data_dir="new_images/",
        checkpoint="lila.science.ckpt"
    )
    predict(config=predict_config)
    ```

For more details on using trained models for prediction, see the [Classifying unlabeled images](images-predict-tutorial.md) tutorial.

