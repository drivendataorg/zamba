# Camtrap DP compatibility

`zamba` is compatible with [Camtrap DP](https://camtrap-dp.tdwg.org/) (Camera Trap Data Package), the [TDWG](https://www.tdwg.org/) community standard for exchanging camera trap data. You can:

- **Train** an image model directly from a Camtrap DP package (input), and
- **Export predictions** as a Camtrap DP package (output).

This means data published to repositories such as [GBIF](https://www.gbif.org/) in Camtrap DP can be used as training labels without conversion, and `zamba` predictions can be handed off to Camtrap DP–aware tooling.

## Using Camtrap DP as training labels (input)

Point `--labels` at a Camtrap DP package and set `--labels-format camtrap_dp`. The package may be provided as any of:

- a **directory** containing `datapackage.json`,
- a path to a **`datapackage.json`** file, or
- a **`.zip`** archive of the package.

=== "CLI"
    ```console
    $ zamba image train --data-dir example_images/ --labels path/to/camtrap_dp/ --labels-format camtrap_dp
    ```
=== "Python"
    ```python
    from zamba.images.config import ImageClassificationTrainingConfig, BboxInputFormat
    from zamba.images.manager import train

    train_config = ImageClassificationTrainingConfig(
        data_dir="example_images/",
        labels="path/to/camtrap_dp/",
        labels_format=BboxInputFormat.CAMTRAP_DP,
    )
    train(config=train_config)
    ```

**What `zamba` reads:**

- `observations` are joined to `media` on `mediaID`, and non-image media (by `fileMediatype`, or file extension when absent) are dropped.
- The label comes from `scientificName`, falling back to `observationType` when `scientificName` is empty.
- `deploymentID` is mapped to `site`, which `zamba` can use to allocate images to train/validation/test splits.
- The local `filePath` is preferred; `fileName` is used when `filePath` is missing or a remote URL. Image paths are resolved relative to `--data-dir`.

**Bounding boxes are optional:**

- If observations include usable relative boxes (`bboxX`, `bboxY`, `bboxWidth`, `bboxHeight`), `zamba` trains on the cropped detections.
- If the package has no usable boxes (a common case, since these fields are optional), `zamba` falls back to whole-image labels. If `crop_images` is enabled, MegaDetector is used to generate crops at train time.

## Exporting predictions as Camtrap DP (output)

Set `--results-file-format camtrap_dp` when predicting. `zamba` writes a Camtrap DP package **directory** (named after the results file's stem) containing `datapackage.json`, `deployments.csv`, `media.csv`, and `observations.csv`.

=== "CLI"
    ```console
    $ zamba image predict --data-dir example_images/ --results-file-format camtrap_dp
    ```
=== "Python"
    ```python
    from zamba.images.manager import predict
    from zamba.images.config import ImageClassificationPredictConfig, ResultsFormat

    predict_config = ImageClassificationPredictConfig(
        data_dir="example_images/",
        results_file_format=ResultsFormat.CAMTRAP_DP,
    )
    predict(config=predict_config)
    ```

**What `zamba` writes:** one `observations` row per detection, with `observationType` (`animal` / `human` / `vehicle`), the top predicted `scientificName` and `classificationProbability` for animals, and relative bounding boxes (`bboxX`, `bboxY`, `bboxWidth`, `bboxHeight`). Each image becomes a `media` row.

**Placeholder fields:** deployment, timestamp, and location fields required by the Camtrap DP spec cannot be inferred from images alone, so they are written as empty values under a single placeholder deployment. The package is intended for downstream ingestion and round-trips back through `zamba`'s own Camtrap DP reader; if you need a package that passes strict spec validation, fill in these deployment/media metadata fields with your own camera trap records.

## Incorporating predictions into an existing Camtrap DP package

If you already maintain a Camtrap DP package for your project (with real `deployments`, real `media`, and real timestamps and locations), you usually don't want `zamba`'s placeholder deployment and media rows. Instead, take just the **predicted observations** and append them to your existing `observations` table.

Because `zamba` assigns its own synthetic `mediaID`s (and a placeholder `deploymentID`), the one step that matters is **remapping each predicted observation back to the real `mediaID`/`deploymentID` in your package**, joining on the image file. `zamba`'s `media.csv` records the `filePath`/`fileName` of every image, which is what makes this possible.

```python
import pandas as pd

# zamba prediction package
zamba_media = pd.read_csv("zamba_predictions/media.csv")
zamba_obs = pd.read_csv("zamba_predictions/observations.csv")

# your existing Camtrap DP package
existing_media = pd.read_csv("my_package/media.csv")
existing_obs = pd.read_csv("my_package/observations.csv")

# 1. build a lookup from image file -> your real mediaID + deploymentID
#    fileName is convenient; use filePath if file names are not unique across deployments
file_to_media = existing_media.set_index("fileName")[["mediaID", "deploymentID"]]

# 2. resolve each prediction's file, then remap onto your real IDs
predicted = zamba_obs.copy()
predicted["fileName"] = predicted["mediaID"].map(zamba_media.set_index("mediaID")["fileName"])
resolved = predicted["fileName"].map(file_to_media["mediaID"].to_dict())
predicted["deploymentID"] = predicted["fileName"].map(file_to_media["deploymentID"].to_dict())
predicted["mediaID"] = resolved
predicted["eventID"] = predicted["mediaID"]  # if you key events by media

# 3. drop predictions whose image isn't in your package, and clean up the helper column
unmatched = predicted["mediaID"].isna()
if unmatched.any():
    print(f"Dropping {int(unmatched.sum())} predictions with no matching media in the package")
predicted = predicted[~unmatched].drop(columns="fileName")

# 4. give the appended rows unique observationIDs, then concatenate
predicted["observationID"] = [f"zamba-{i:06d}" for i in range(len(predicted))]
combined = pd.concat([existing_obs, predicted], ignore_index=True)
combined.to_csv("my_package/observations.csv", index=False)
```

Notes:

- Keep `zamba`'s `deployments.csv` and `media.csv` **out** of the merge — you already have the real versions. Only the `observations` rows are appended.
- `zamba` writes `classificationMethod="machine"` and `classifiedBy="zamba"` on every predicted observation, so you can always filter its predictions back out of the combined table.
- If your existing `observations` table lacks the `bbox*` columns, `pandas` adds them (empty for your prior rows); these fields are optional in the spec.
- Make sure the final `observationID` values are unique across the combined table (the `zamba-000000` prefix above keeps them from colliding with your existing IDs).
- After merging, re-validate the package (for example with [`frictionless`](https://framework.frictionlessdata.io/) or the [Camtrap DP tooling](https://camtrap-dp.tdwg.org/#software)) to confirm the appended observations reference valid `mediaID`s and `deploymentID`s.
