# Tutorial: Fine-tune image models effectively

This tutorial turns an eight-dataset camera-trap benchmark into a practical first workflow. It is not an exact reproduction of the benchmark: the benchmark used a controlled optimizer, schedule, crop-selection, and evaluation recipe, while the commands below use the supported Zamba interface and defaults. See the [experiment results](image-finetuning-results.md#methods-and-definitions) for the full protocol.

The benchmark contains 258 distinct model evaluations represented by 264 result rows. Two zero-shot results are repeated across four training-image tiers for display. None of the eight LILA datasets was used to train the `lila.science` base model, although many species were represented in its training data and seven of the eight datasets come from continents that training data covered. Training sets used nested caps of 10, 100, 1,000, and 10,000 **training images** per class, plus separate validation and test images from other camera locations.

The comparisons cover fine-tuning, ImageNet starts, linear probes, and a combined class-balanced-sampling-plus-augmentation setting. Two datasets also include DINOv3, BioCLIP, and a zero-shot vision-language model. Use the [interactive results](image-finetuning-results.md) to inspect individual datasets, class results, methods, and citations. For every training option, see [Training a model on labeled images](images-train-tutorial.md).

!!! tip "Prefer a graphical interface?"
    Use [Zamba Cloud](https://www.zambacloud.com/) to fine-tune a model and run predictions in a web browser. The data and evaluation recommendations in this tutorial also apply to Zamba Cloud.

Results will differ for other datasets. Treat these values as planning estimates and measure performance on data from your own camera sites.

## What the benchmark suggests

- **Use 10 to 100 training images per class as a starting range, not a guarantee.** Fine-tuned `lila.science` averaged about 0.74 reliable-class macro accuracy at 10 training images per class and 0.78 at 100. Individual-dataset results ranged from about 0.35 to 0.95 at 10 images and 0.43 to 0.98 at 100. Keep additional locations and images for validation and testing.
- **Add data selectively after the first run.** Results changed less on average above 1,000 training images per class, but difficult classes and datasets varied substantially. Use validation and per-class errors to decide where 500 to 1,000 or more images are worthwhile.
- **Start from a wildlife model.** The same `lila.science` architecture started from ImageNet averaged about 0.55 macro accuracy at 10 images per class, roughly 0.20 below the wildlife-pretrained start.
- **Try `lila.science` first.** It had higher macro accuracy than fine-tuned SpeciesNet in 30 of 32 dataset-by-tier comparisons. It also uses 224×224 inputs; SpeciesNet uses 480×480 inputs and typically needs more GPU memory.
- **Prioritize site diversity.** Classes present at more locations in the full source catalogs had higher accuracy. Collect across multiple camera sites before collecting many repeat images from the same sites.
- **Keep sites out of multiple splits.** A location-disjoint test better represents performance at new camera sites.
- **Use compatible labels.** Avoid mixing a broad label with one of its members, such as `squirrel` and `red_squirrel`.
- **Try the combined balanced setting when rare classes perform poorly.** In the benchmark, inverse-frequency sampling and stronger augmentation were changed together. Specific classes with fewer than 200 training images gained about 0.13 accuracy at the 10,000-image cap; the experiment does not isolate which part caused the gain.
- **Inspect class-level errors.** Aggregate accuracy can hide failures on rare classes.

The headline macro statistic first averages qualifying classes within each dataset, excluding classes with fewer than 30 scored test crops, and then gives each of the eight dataset-level values equal weight. It is based on one training seed and nested training sets evaluated on shared holdouts, so the comparisons are correlated and do not measure training-to-training variance.

## 1. Prepare the labels

Create a CSV file with `filepath`, `label`, and `site` columns.

```console
$ cat labels.csv
filepath,label,site
site_01/fox_001.jpg,red_fox,site_01
site_02/bobcat_001.jpg,bobcat,site_02
site_11/fox_101.jpg,red_fox,site_11
site_12/bobcat_101.jpg,bobcat,site_12
site_21/fox_201.jpg,red_fox,site_21
site_22/bobcat_201.jpg,bobcat,site_22
```

Zamba keeps each non-empty site ID in one split when the CSV contains a `site` column. Use complete, stable site IDs: rows with a missing site are assigned individually and can weaken the location holdout. Zamba creates the training, validation, and test splits.

Use at least three camera sites for each class when possible. Three sites are necessary, but not sufficient, for automatic allocation to place a class in all three splits because site assignment is not stratified by class.

After the first run, join `labels.csv` with the generated `splits.csv` by `filepath`. Make sure that each class occurs in all three splits. If necessary, add a `split` column with `train`, `val`, or `test` values. Make sure that one site does not occur in two splits.

Clean the labels before training:

- Use one taxonomic level for related classes.
- Remove or relabel `unknown`, `other`, and `animal` when they are not intended prediction targets.
- Do not use a broad label and one of its members. For example, do not use both `squirrel` and `red_squirrel`.
- Merge species that people cannot identify reliably in the images.
- Check the labels of rare classes carefully.

## 2. Test the workflow

First, run a small training job. The `--debug 3` option uses up to three images per class in each split.

```console
$ mkdir -p runs/check
$ zamba image train \
    --data-dir images/ \
    --labels labels.csv \
    --model lila.science \
    --debug 3 \
    --max-epochs 1 \
    --save-dir runs/check \
    --yes
```

This job checks file paths, labels, image loading, and the training process. Do not use its metrics as final results.

## 3. Train a baseline model

Use `lila.science` as the first model. It is the default image model and is trained on camera-trap data.

```console
$ mkdir -p runs/lila
$ zamba image train \
    --data-dir images/ \
    --labels labels.csv \
    --model lila.science \
    --max-epochs 40 \
    --early-stopping-patience 5 \
    --save-dir runs/lila \
    --yes
```

Do not add `--from-scratch`. The experiments found that pretrained wildlife models needed much less data than models trained from base ImageNet weights. This difference was largest when there were few labeled images.

Zamba uses MegaDetector to create crop samples by default. One source image can produce more than one classifier sample when the detector returns multiple boxes, and an image with no detection produces no classifier sample. If the label file contains COCO or MegaDetector bounding boxes, Zamba uses those boxes instead. See [Labels with bounding boxes](images-train-tutorial.md#labels-with-bounding-boxes).

The benchmark used a stricter, precomputed pipeline: it selected the highest-confidence animal box per image, stored a full-frame fallback for detector misses during training, and scored only held-out images where MegaDetector found an animal. Expect the supported commands here to differ from the published benchmark numbers.

## 4. Correct class imbalance

First, count the training images for each class. Use class-balanced sampling if the counts have large differences.

```console
$ mkdir -p runs/lila-balanced
$ zamba image train \
    --data-dir images/ \
    --labels labels.csv \
    --model lila.science \
    --balanced-sampling \
    --max-epochs 40 \
    --early-stopping-patience 5 \
    --save-dir runs/lila-balanced \
    --yes
```

`--balanced-sampling` shows rare classes more often during training and applies stronger image augmentation. Treat it as one combined setting: the benchmark did not separate the effect of sampling from the effect of augmentation.

Keep the first baseline run. Compare it with the combined-setting run on the same **validation** split. In the experiments, the combined setting improved rare-class and macro results. The cost to overall accuracy was small or absent, but the setting can decrease accuracy slightly when the data is already balanced.

Do not also enable `--weighted-loss` in this first comparison; keep every other setting fixed.

## 5. Evaluate the model

Select settings with validation metrics:

```console
$ cat runs/lila/val_metrics.json
$ cat runs/lila-balanced/val_metrics.json
```

Use these metrics together:

- `val_accuracy` gives equal weight to each scored crop. Common classes have a large effect on this value.
- `val_recall` is macro recall. It gives equal weight to each class.
- `val_f1` is macro F1. It also gives equal weight to each class.

Prefer the model with good macro metrics when all classes are important. A high `val_accuracy` can hide failure on rare classes.

After selecting one run, read its `test_metrics.json` once for the final check. These JSON files contain aggregate metrics, not per-class results. To diagnose individual classes, open the run in MLflow (`mlflow ui`) and inspect its test confusion-matrix artifact. Keep in mind that the current Zamba metrics are per detector crop, not necessarily per source image.

## 6. Compare SpeciesNet when necessary

In these experiments, fine-tuned `lila.science` had higher macro accuracy than fine-tuned SpeciesNet on the same data in 30 of 32 comparisons. Train SpeciesNet when `lila.science` does not give sufficient results. Use the same labels and splits for a fair comparison.

```console
$ mkdir -p runs/speciesnet
$ zamba image train \
    --data-dir images/ \
    --labels labels.csv \
    --model speciesnet \
    --max-epochs 40 \
    --early-stopping-patience 5 \
    --save-dir runs/speciesnet \
    --yes
```

`lila.science` uses 224×224 images. SpeciesNet uses 480×480 images and usually needs more GPU memory. A new species does not require training from scratch. Fine-tune a pretrained model with the new labels first.

## 7. Use the fine-tuned model

### Keep the model files

Keep these files from the selected run:

- The `.ckpt` model checkpoint
- `train_configuration.yaml`
- The labels CSV
- `splits.csv` when Zamba generated the splits
- `val_metrics.json` and `test_metrics.json`

The configuration records how you trained the model. The source labels record the class names; `splits.csv` records automatically generated filepath-to-split assignments. Keep all of them with the checkpoint so future training cannot accidentally include the original test locations.

### Predict on future images

Use the checkpoint from the selected run.

```console
$ mkdir -p predictions
$ zamba image predict \
    --data-dir new_images/ \
    --checkpoint runs/lila/lila.science.ckpt \
    --save-dir predictions \
    --yes
```

Zamba writes the results to `predictions/zamba_predictions.csv`.

Use the same image preparation method that you used for training. The example model used MegaDetector crops, so the prediction command keeps the default MegaDetector step.

The checkpoint contains the trained class names. You do not need to provide the training labels CSV during prediction.

### Continue training with more images

You can fine-tune the saved weights further when you collect more images for exactly the same classes. Keep the original validation and test locations out of the new training data.

The image-training CLI does not expose the setting that preserves an existing classification head, so use a YAML configuration. Set `use_default_model_labels: true`; otherwise the normal fine-tuning path replaces the head even when the class names match.

```yaml title="resume.yaml"
train_config:
  data_dir: updated_images/
  labels: updated_labels.csv
  checkpoint: runs/lila/lila.science.ckpt
  use_default_model_labels: true
  max_epochs: 40
  early_stopping_patience: 5
  save_dir: runs/lila-update
```

```console
$ mkdir -p runs/lila-update
$ zamba image train --config resume.yaml --yes
```

This loads the checkpoint weights, including the trained head, but starts a new trainer and optimizer rather than restoring the exact prior epoch and optimizer state. Compare the updated model with the previous model on the same validation locations, select one, and then report the selected model on the unchanged test locations.

If you add, remove, or rename classes, start a new fine-tune from `lila.science` with the complete new label set. Do not preserve a head whose class set differs.

## Diagnose poor results

- If validation results are high but test results are low, check for site leakage and limited site coverage.
- If common classes are good but rare classes are poor, use `--balanced-sampling`. Also collect more rare-class images.
- If related species are often confused, check the taxonomy. Merge labels when the visual difference is not reliable.
- If small animals are missing, check the MegaDetector crops. More classifier training data might not correct a detection problem.
- If more images do not improve results, check label quality and site diversity before you collect more data.

## Recommended first experiment

1. Prepare clean labels from at least three sites per class.
2. Start with approximately 100 training images per class, plus separate validation and test images from other sites.
3. Train `lila.science` without `--from-scratch`.
4. Train a second run with `--balanced-sampling` if the data is imbalanced.
5. Select the model with validation macro metrics.
6. Report final metrics from held-out camera sites.
