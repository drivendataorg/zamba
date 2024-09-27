# `zamba` changelog

## v.2.5.0 (2024-09-27)

* Removed support for Python 3.8 and 3.9 plus other requirement updates ([PR #337](https://github.com/drivendataorg/zamba/pull/337), [PR #335](https://github.com/drivendataorg/zamba/pull/335)). New minimum python version is 3.11.

## v2.4.1 (2024-04-20)

* Bug fixes for docs

## v2.4.0 (2024-04-19)

* Adds experimental image support ([PR #314](https://github.com/drivendataorg/zamba/pull/314))
* Clarifies installation instructions for Linux and Windows operating systems ([PR #299](https://github.com/drivendataorg/zamba/pull/299))

## v2.3.2 (2023-07-17)

* Pin Pydantic to less than v2.0 ([PR #277](https://github.com/drivendataorg/zamba/pull/277))

## v2.3.1 (2023-05-12)

* Code updates for PyTorch Lightning v2.0.0 ([PR #266](https://github.com/drivendataorg/zamba/pull/266), [PR #272](https://github.com/drivendataorg/zamba/pull/272))
* Switch to pyproject.toml-based build and other requirement updates ([PR #254](https://github.com/drivendataorg/zamba/pull/254), [PR #255](https://github.com/drivendataorg/zamba/pull/255), [PR #260](https://github.com/drivendataorg/zamba/pull/260), [PR #262](https://github.com/drivendataorg/zamba/pull/262), [PR #268](https://github.com/drivendataorg/zamba/pull/268))

## v2.3.0 (2022-12-01)

### Model release

* Adds a depth estimation module for predicting the distance between animals and the camera ([PR #247](https://github.com/drivendataorg/zamba/pull/247)). This model comes from one of the winning solutions in the [Deep Chimpact: Depth Estimation for Wildlife Conservation](https://www.drivendata.org/competitions/82/competition-wildlife-video-depth-estimation/) machine learning challenge hosted by DrivenData.

## v2.2.4 (2022-11-10)

* Do not cache videos if the `VIDEO_CACHE_DIR` environment variable is an empty string or zero ([PR #245](https://github.com/drivendataorg/zamba/pull/245))

## v2.2.3 (2022-11-01)

* Fixes Lightning deprecation of DDPPlugin ([PR #244](https://github.com/drivendataorg/zamba/pull/244))

## v2.2.2 (2022-10-04)

* Adds a page to the docs summarizing the performance of the African species classification model on a holdout set ([PR #235](https://github.com/drivendataorg/zamba/pull/235))

## v2.2.1 (2022-09-27)

* Turn off showing local variables in Typer's exception and error handling ([PR #237](https://github.com/drivendataorg/zamba/pull/237))
* Fixes bug where the column order was incorrect for training models when the provided labels are a subset of the model's default labels ([PR #236](https://github.com/drivendataorg/zamba/pull/236))

## v2.2.0 (2022-09-26)

### Model releases and new features

* The default `time_distributed` model (African species classification) has been retrained on over 250,000 videos. This 16x increase in training data significantly improves accuracy. This new version replaces the previous one. ([PR #226](https://github.com/drivendataorg/zamba/pull/226), [PR #232](https://github.com/drivendataorg/zamba/pull/232))
* A new default model option is added: `blank_nonblank`. This model only does blank detection. This binary model can be trained and finetuned in the same way as the species classification models. This model was trained on both African and European data, totaling over 263,000 training videos. ([PR #228](https://github.com/drivendataorg/zamba/pull/228))
* Detect if a user is training in a binary model and preprocess the labels accordingly ([PR #215](https://github.com/drivendataorg/zamba/pull/215))

### Bug fixes and improvements

* Add a validator to ensure that using a model’s default labels is only possible when the species in the provided labels file are a subset of those ([PR #229](https://github.com/drivendataorg/zamba/pull/229))
* Refactor the logic in `instantiate_model` for clarity ([PR #229](https://github.com/drivendataorg/zamba/pull/229))
* Use pqdm to check for missing files in parallel ([PR #224](https://github.com/drivendataorg/zamba/pull/224))
* Set `model_name` based on the provided checkpoint so that user-trained models use the appropriate video loader config ([PR #221](https://github.com/drivendataorg/zamba/pull/221))
* Leave `data_dir` as a relative path ([PR #219](https://github.com/drivendataorg/zamba/pull/219))
* Ensure hparams yaml files get included in the source distribution ([PR #210](https://github.com/drivendataorg/zamba/pull/210))
* Hold back setuptools so mkdocstrings works ([PR #207](https://github.com/drivendataorg/zamba/pull/207))
* Factor out `get_cached_array_path` ([PR #202](https://github.com/drivendataorg/zamba/pull/202/files))

## v2.1.0 (2022-07-15)

- Retrains the time distributed species classification model using the updated MegadetectorLite frame selection ([PR #199](https://github.com/drivendataorg/zamba/pull/199))
- Replaces the MegadetectorLite frame selection model with an improved model trained on significantly more data ([PR #195](https://github.com/drivendataorg/zamba/pull/195))

## v2.0.4 (2022-06-17)

 - Pins `thop` to an earlier version ([PR #191](https://github.com/drivendataorg/zamba/pull/191))
 - Fixes caching so a previously downloaded checkpoint file actually gets used ([PR #190](https://github.com/drivendataorg/zamba/pull/190), [PR #194](https://github.com/drivendataorg/zamba/pull/194))
 - Removes a lightning deprecation warning for DDP ([PR #187](https://github.com/drivendataorg/zamba/pull/187))
 - Ignores extra columns in the user-provided labels or filepaths csv ([PR #186](https://github.com/drivendataorg/zamba/pull/186))

## v2.0.3 (2022-05-06)

Releasing to pick up #179.

 - PR [#179](https://github.com/drivendataorg/zamba/pull/179) removes the DensePose extra from the default dev requirements and tests. Docs are updated to clarify how to install and run tests for DensePose.

## v2.0.2 (2021-12-21)

Releasing to pick up #172.

 - PR [#172](https://github.com/drivendataorg/zamba/pull/172) fixes bug where video loading that uses the YoloX model (all of the built in models) resulted in videos not being able to load.


## v2.0.1 (2021-12-15)

Releasing to pick up #167 and #169.

 - PR [#169](https://github.com/drivendataorg/zamba/pull/169) fixes error in splitting data into train/test/val when only a few videos.
 - PR [#167](https://github.com/drivendataorg/zamba/pull/167) refactors yolox into an `object_detection` module

Other documentation fixes also included.

## v2.0.0 (2021-10-22)

### Previous model: Machine learning competition

The algorithms used by `zamba` v1 were based on the winning solution from the
[Pri-matrix Factorization](https://www.drivendata.org/competitions/49/deep-learning-camera-trap-animals/) machine learning competition, hosted by [DrivenData](https://www.drivendata.org/). Data for the competition was provided by the [Chimp&See project](https://www.chimpandsee.org/#/) and manually labeled by volunteers. The competition had over 300 participants and over 450 submissions throughout the three month challenge. The v1 algorithm was adapted from the winning competition submission, with some aspects changed during development to improve performance.

The core algorithm in `zamba` v1 was a [stacked ensemble](https://en.wikipedia.org/wiki/Ensemble_learning#Stacking) which consisted of a first layer of models that were then combined into a final prediction in a second layer. The first level of the stack consisted of 5 `keras` deep learning models, whose individual predictions were combined in the second level
of the stack to form the final prediction.

In v2, the stacked ensemble algorithm from v1 is replaced with three more powerful [single-model options](models/species-detection.md): `time_distributed`, `slowfast`, and `european`. The new models utilize state-of-the-art image and video classification architectures, and are able to outperform the much more computationally intensive stacked ensemble model.

### New geographies and species

`zamba` v2 incorporates data from Western Europe (Germany). The new data is packaged in the pretrained `european` model, which can predict 11 common European species not present in `zamba` v1.

`zamba` v2 also incorporates new training data from 15 countries in central and west Africa, and adds 12 additional species to the pretrained African models.

### Retraining flexibility

Model training is made available `zamba` v2, so users can finetune a pretrained model using their own data to improve performance for a specific ecology or set of sites. `zamba` v2 also allows users to retrain a model on completely new species labels.
