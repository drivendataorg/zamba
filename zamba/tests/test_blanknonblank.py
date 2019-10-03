import numpy as np
from zamba.models.blanknonblank import BlankNonBlank


def test_blanknonblank(data_dir):
    n = 10

    bnb = BlankNonBlank()
    n_features = bnb.model.best_estimator_.n_features_

    X = np.random.rand(n, n_features)
    blank = bnb.predict_proba(X)[:, 1]

    assert blank.shape == (n,)
