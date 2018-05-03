# About the Algorithms

## Winning Performance

The algorithms used by `zamba` are based on the winning solution from the
[Pri-matrix Factorization](https://www.drivendata.org/competitions/49/deep-learning-camera-trap-animals/) machine learning
competition, hosted by [DrivenData](https://www.drivendata.org/). Some aspects
of the solution have been changed during development to improve performance.

## Stacked Ensemble

`zamba` uses a stacked ensemble to determine which species are present in a
video it sees. The first level of the stack consists of 5 `keras` deep
learning models, whose individual predictions are combined in the second level
of the stack to form the final prediction.

### First Level Models

The first level deep learning models are implemented by the `keras` api that is
 shipped _within_ `tensorflow`. They are:

* [`ResNet50`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50)
* [`InceptionV3`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionV3)
* [`Xception`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/Xception)
* [`NASNetMobile`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/NASNetMobile)
* [`InceptionResNetV2`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionResNetV2)

### Second Level Models

The second level models used to combine predictions are:
* [`XGBClassifier`](http://xgboost.readthedocs.io/en/latest/python/python_api.html)
* A simple 3 layer neural network model implemented in `keras`.

### Training

Out-of-fold predictions from the level 1 models were used to train the level 2
models in order to avoid over-fitting the training set.

The trained model is included with `zamba`.

**NOTE: `zamba` needs to download the "weights" files for the neural networks
that it uses to make predictions. On first run it will download ~1GB of files
with these weights.** Once these are downloaded, the tool will use the local
versions and will not need to perform this download again.