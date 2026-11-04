# Tutorial: Fine-tune image models effectively

We produced these recommendations from 264 completed evaluations across eight LILA camera-trap datasets. None of these datasets was used to train the `lila.science` base model that ships with `zamba`, so the results evaluate fine-tuning on new data. Many of the species were already present in that base model's training data. The main experiment used nested training sets with caps of 10, 100, 1,000, and 10,000 images per class. Each camera location occurred in only one of the training, validation, or test splits.

We compared:

- Fine-tuning `lila.science` and SpeciesNet
- Training the same model architectures from base ImageNet weights
- Standard training and class-balanced sampling
- Linear probes (a small linear classifier on a frozen base model) for `lila.science` and SpeciesNet
- DINOv3 and BioCLIP as frozen feature extractors, and a zero-shot vision-language model, on two of the datasets

On those two datasets, a strong frozen feature extractor or a general vision-language model was worth comparing where fine-tuning did poorly. See the [experiment results](image-finetuning-results.md) for that comparison.

We evaluated the models on MegaDetector animal crops from held-out camera locations. We reviewed overall metrics, per-class metrics, label quality, camera-location coverage, and animal size in the image.

For a full description of each training option, see [Training a model on labeled images](images-train-tutorial.md).

Use the [interactive experiment results](image-finetuning-results.md) to inspect the charts, run metrics, class results, methods, and citations.

!!! tip "Prefer a graphical interface?"
    Use [Zamba Cloud](https://www.zambacloud.com/) to fine-tune a model and run predictions in a web browser. The data and evaluation recommendations in this tutorial also apply to Zamba Cloud.

Results will differ for other datasets. Treat the values below as planning estimates.

## Main lessons

- Start with 10 to 100 images per class. Fine-tuned `lila.science` reached about 0.74 macro accuracy at 10 images per class and about 0.78 at 100.
- Use 500 to 1,000 images for important or difficult classes. Expect smaller gains above approximately 1,000 images per class.
- Start with a wildlife model that is already trained. An ImageNet start had lower accuracy. It was about 0.55 at 10 images per class.
- Prefer `lila.science` for the first fine-tune. It had higher macro accuracy than fine-tuned SpeciesNet on the same data in 30 of 32 comparisons. It uses 224×224 images. SpeciesNet uses 480×480 images and more GPU memory.
- Collect images from more camera sites before you collect many images from the same sites. More images helped classes already seen at many sites. More images lowered accuracy for classes seen at few sites.
- Put each camera site in only one data split. This gives a more realistic test.
- Use one clear label for each type of animal. Do not mix species labels with broad labels that include those species.
- Use class-balanced sampling when rare classes perform poorly. It improved rare classes. Classes with 500 or more images changed by about zero.
- Check per-class performance. Overall accuracy can hide poor results for rare classes.

## Evidence for the lessons

The study used macro accuracy to give equal weight to each class.

- Fine-tuned `lila.science` models had approximately 0.74 macro accuracy with 10 images per class. They were near their maximum result at approximately 100 images per class.
- Fine-tuned `lila.science` had higher macro accuracy than fine-tuned SpeciesNet on the same data in 30 of 32 comparisons. The mean difference was approximately 0.10 at 10 images per class. The mean stayed higher at every data level. `lila.science` uses 224×224 images. SpeciesNet uses 480×480 images.
- Models trained from base ImageNet weights were approximately 0.20 lower at 10 images per class. They started to close the gap at approximately 1,000 images per class.
- Results usually changed little after 1,000 images per class. Severe class imbalance sometimes decreased macro accuracy at higher data levels.
- Class-balanced sampling increased accuracy for classes with fewer than 200 training images by about 0.13. Classes with 500 or more images changed by about zero. Mean macro accuracy increased by approximately 0.04 at 10,000 images per class.
- With 10 training images per class, classes seen at many camera locations scored about 0.84. Classes seen at few locations scored about 0.71. Adding images up to 10,000 raised many-location classes to about 0.94. Few-location classes fell to about 0.57.
- Small animals in the image were approximately 0.15 lower in accuracy. More training data did not fully correct this difference.

Treat these values as planning estimates. Measure performance on data from your own camera sites.

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

Zamba keeps each site in one split when the CSV contains a `site` column. Zamba creates the training, validation, and test splits.

Use at least three camera sites for each class when possible. Three sites are necessary for a class to occur in all three splits. More sites improve performance more than more images from the same sites. In the experiments, more images lowered accuracy when a class was seen at few sites.

After the first run, compare `labels.csv` with `splits.csv`. Make sure that each class occurs in all three splits. If necessary, add a `split` column with `train`, `val`, or `test` values. Make sure that one site does not occur in two splits.

Clean the labels before training:

- Use one taxonomic level for related classes.
- Remove labels such as `unknown`, `other`, and `animal`.
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

Zamba uses MegaDetector to crop animals by default. If the label file contains COCO or MegaDetector bounding boxes, Zamba uses those boxes. See [Labels with bounding boxes](images-train-tutorial.md#labels-with-bounding-boxes).

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

`--balanced-sampling` shows rare classes more often during training. It also applies stronger image augmentation. The augmentation reduces memorization of repeated rare images.

Keep the first baseline run. Compare it with the balanced run on the same test split. In the experiments, balanced sampling improved the rare-class and macro results. The cost to overall accuracy was small or absent. Balanced sampling can decrease accuracy slightly when the data is already balanced.

Do not enable `--balanced-sampling` and `--weighted-loss` in the first comparison. Test one change at a time.

## 5. Evaluate the model

Read the metrics from the held-out test sites.

```console
$ cat runs/lila/test_metrics.json
$ cat runs/lila-balanced/test_metrics.json
```

Use these metrics together:

- `test_accuracy` gives equal weight to each image. Common classes have a large effect on this value.
- `test_recall` is macro recall. It gives equal weight to each class.
- `test_f1` is macro F1. It also gives equal weight to each class.

Prefer the model with good macro metrics when all classes are important. A high `test_accuracy` can hide failure on rare classes.

Use the validation split to select settings. Use the test split only for the final check.

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
- `val_metrics.json` and `test_metrics.json`

The configuration records how you trained the model. The labels file records the class names and data splits. Keep both files with the checkpoint.

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

You can continue training when you collect more images for the same classes. Keep the class names unchanged. Keep the original test locations out of the new training data.

Create a new output directory. Pass the previous checkpoint with `--model-checkpoint`:

```console
$ mkdir -p runs/lila-update
$ zamba image train \
    --data-dir updated_images/ \
    --labels updated_labels.csv \
    --model-checkpoint runs/lila/lila.science.ckpt \
    --max-epochs 40 \
    --early-stopping-patience 5 \
    --save-dir runs/lila-update \
    --yes
```

Compare the updated model with the previous model on the same test locations.

If you add or remove classes, start a new fine-tune from `lila.science`. Train with the complete new label set. Do not resume from a checkpoint that has a different class set.

## Diagnose poor results

- If validation results are high but test results are low, check for site leakage and limited site coverage.
- If common classes are good but rare classes are poor, use `--balanced-sampling`. Also collect more rare-class images.
- If related species are often confused, check the taxonomy. Merge labels when the visual difference is not reliable.
- If small animals are missing, check the MegaDetector crops. More classifier training data might not correct a detection problem.
- If more images do not improve results, check label quality and site diversity before you collect more data.

## Recommended first experiment

1. Prepare clean labels from at least three sites per class.
2. Start with approximately 100 images per class.
3. Train `lila.science` without `--from-scratch`.
4. Train a second run with `--balanced-sampling` if the data is imbalanced.
5. Select the model with validation macro metrics.
6. Report final metrics from held-out camera sites.
