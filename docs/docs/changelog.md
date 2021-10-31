# `zamba` changelog

## v2 (2021-10-22)

### Previous model: Machine learning competition

The algorithms used by `zamba` v1 were based on the winning solution from the
[Pri-matrix Factorization](https://www.drivendata.org/competitions/49/deep-learning-camera-trap-animals/) machine learning
competition, hosted by [DrivenData](https://www.drivendata.org/). Data for the competition was provided by the [Chimp&See project](https://www.chimpandsee.org/#/) and manually labeled by volunteers. The competition had over 300 participants and over 450 submissions throughout the three month challenge. The v1 algorithm was adapted from the winning competition submission, with some aspects changed during development to improve performance.

The core algorithm in `zamba` v1 was a [stacked ensemble](https://en.wikipedia.org/wiki/Ensemble_learning#Stacking) which consisted of a first layer of models that were then combined into a final prediction in a second layer. The first level of the stack consisted of 5 `keras` deep
learning models, whose individual predictions were combined in the second level
of the stack to form the final prediction.

In v2, the stacked ensemble algorithm from v1 is replaced with three more powerful [single-model options](models/species-detection.md): `time_distributed`, `slowfast`, and `european`. The new models utilize state-of-the-art image and video classification architectures, and are able to outperform the much more computationally intensive stacked ensemble model.

### New geographies and species

`zamba` v2 incorporates data from Western Europe (Germany). The new data is packaged in the pretrained `european` model, which can predict 11 common European species not present in `zamba` v1.

`zamba` v2 also incorporates new training data from 15 countries in central and west Africa, and adds 12 additional species to the pretrained African models.

### Retraining flexibility

Model training is made available `zamba` v2, so users can finetune a pretrained model using their own data to improve performance for a specific ecology or set of sites. `zamba` v2 also allows users to retrain a model on completely new species labels.
