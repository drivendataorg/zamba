# `zamba` changelog

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
