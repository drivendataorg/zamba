# About the Algorithms

The algorithms in `zamba` are designed to identify species of animals that appear in camera trap videos. These classification models were trained using data from the [Chimp&See project](https://www.chimpandsee.org/#/). These videos were labeled by a crowd of volunteers that watched them and identified species that appeared in the clip.

## Machine Learning Competition

The algorithms used by `zamba` are based on the winning solution from the
[Pri-matrix Factorization](https://www.drivendata.org/competitions/49/deep-learning-camera-trap-animals/) machine learning
competition, hosted by [DrivenData](https://www.drivendata.org/). This competition had over 300 participants and over 450 submissions throughout the three month challenge. The algorithm in this software has been adapted from the one that won the machine learning competition. Some aspects
of the solution have been changed during development to improve performance.

## Stacked Ensemble

The core algorithm in `zamba` is a [stacked ensemble](https://en.wikipedia.org/wiki/Ensemble_learning#Stacking) which consists of a first layer of models that are then combined into a final prediction in a second layer. The first level of the stack consists of 5 `keras` deep
learning models, whose individual predictions are combined in the second level
of the stack to form the final prediction.

### First Level Models

The first level of the ensemble is based on fine-tuning a number of well-known deep learning architectures. These architectures were developed for specific tasks, e.g. ImageNet, but are re-trained to identify the species in our camera trap videos. Each of these models is implemented by the `keras` api that is
 shipped _within_ `tensorflow`. They are:

* [`ResNet50`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50)
* [`InceptionV3`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionV3)
* [`Xception`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/Xception)
* [`NASNetMobile`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/NASNetMobile)
* [`InceptionResNetV2`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionResNetV2)

### Second Level Models

Once these models make predictions on the videos, those predictions are then combined and used to train another level of the stack. The second level models used to combine predictions are:
* [`XGBClassifier`](http://xgboost.readthedocs.io/en/latest/python/python_api.html)
* A simple 3 layer neural network model implemented in `keras`.

Out-of-fold predictions from the level 1 models were used to train the level 2
models in order to avoid over-fitting the training set.

The weights for the trained model are included with `zamba`, and they will be downloaded the first time that the software is executed.

**NOTE: `zamba` needs to download these "weights" files for the neural networks
that it uses to make predictions. On first run it will download ~1GB of files
with these weights.** Once these are downloaded, the tool will use the local
versions and will not need to perform this download again.


### Training

While it is possible to re-train the models in `zamba`, this is not currently supported.

### Finetuning

Fine-tuning would enable this model to
predict additional––or entirely different––class labels.

While it would be possible to fine-tune the models in `zamba` to a new set of species on a new set of videos, this is not currently supported.



