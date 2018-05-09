import os
from pathlib import Path
import pickle
from multiprocessing.pool import Pool

import numpy as np
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l1

from zamba.models.cnnensemble.src import config, metrics, utils

import logging
logger = logging.getLogger(__file__)

NB_CAT = len(config.CLASSES)


def preprocess_x_histogram(data: np.ndarray):
    """
        Pre-process results of l1 model prediction to be used as an input to l2 model.

        This method is used for xgboost and lgbm models and uses set of per class statistics and histogram

        :param data: np.ndarray, predictions of the single l1 model with the shape of (NB_CLIPS, NB_FRAMES, NB_CLASSES)
        :return: pre-processed X tensor with shape (NB_CLIPS, X_WIDTH)
    """
    rows = []

    for row in data:
        items = [
            np.mean(row, axis=0),
            np.median(row, axis=0),
            np.min(row, axis=0),
            np.max(row, axis=0),
            np.percentile(row, q=10, axis=0),
            np.percentile(row, q=90, axis=0),
        ]
        for col in range(row.shape[1]):
            items.append(np.histogram(row[:, col], bins=10, range=(0.0, 1.0), density=True)[0])
        rows.append(np.hstack(items).flatten())

    return np.array(rows)


def preprocess_x_sorted_bins(data: np.ndarray):
    """
        Pre-process results of l1 model prediction to be used as an input to l2 model.

        This method is used for MLP model.

        Per class predictions are sorted and each bin of 4 values is averaged.

        :param data: np.ndarray, predictions of the single l1 model with the shape of (NB_CLIPS, NB_FRAMES, NB_CLASSES)
        :return: pre-processed X tensor with shape (NB_CLIPS, X_WIDTH)
    """
    rows = []
    for row in data:
        items = []
        for col in range(row.shape[1]):
            sorted = np.sort(row[:, col])
            items.append(sorted.reshape(-1, config.L2_SORTED_BINS_DOWNSAMPLE).mean(axis=1))
        rows.append(np.hstack(items))
    return np.array(rows)


def load_train_data(model_name, fold, preprocess_x):
    """
    Load the out of fold train data from file saved from l1 model prediction

    :param model_name: l1 model name
    :param fold: fold number NOT used for model training
    :param preprocess_x: function used to pre-process model input

    :return: X, y, video_ids
                np.ndarray with pre-processed train data;
                np.ndarray with labels as one hot encoded array
                list of video ids of X and y
    """
    data_path = config.MODEL_DIR / 'output/prediction_train_frames'
    raw_cache_fn = f'{data_path}/{model_name}_{fold}_combined.npz'
    cached = np.load(raw_cache_fn)
    X_raw, y, video_ids = cached['X_raw'], cached['y'], cached['video_ids']
    X = preprocess_x(X_raw)
    return X, y, video_ids


class SecondLevelModel:
    """
    Base class for the second level model, used to combine multiple individual l1 models predictions
    from multiple video frames.
    """
    def __init__(self, l1_model_names, preprocess_l1_model_output):
        """
        :param l1_model_names: list or l1 model names used this model combines results from
        :param preprocess_l1_model_output: function to pre-process l1 models output
        """
        self.l1_model_names = l1_model_names
        self.preprocess_l1_model_output = preprocess_l1_model_output
        self.pool = Pool(config.N_CORES)

    def _predict(self, X):
        """
        Internal, to be implemented by model implementation.

        Predict class probabilities

        :param X: Input vector
        :return: class probabilities as np.ndarray with shape (NB_ITEMS, NB_CLASSES)
        """
        raise NotImplemented()

    def _train(self, X, y):
        """
        Internal, to be implemented by model

        Train model.

        :param X:
        :param y: result as labels (not categorical) TODO: switch to categorical to allow multiple classes on one clip
        :return:
        """
        raise NotImplemented()

    def predict(self, l1_model_results):
        """
        :param l1_model_results: results of l1 model predictions as a dictionary
                             { model_name : ndarray(NB_CLIPS, NB_FRAMES, NB_CLASSES) }
        :return: class probabilities as ndarray with shape (NB_CLIPS, NB_CLASSES)
        """
        X_per_model = self.pool.map(self.preprocess_l1_model_output,
                                    [l1_model_results[model] for model in self.l1_model_names])
        X = np.column_stack(X_per_model)
        return self._predict(X)

    def train(self):
        """
        Train model using saved L1 model out of fold predictions
        """
        X_all_combined = []
        y_all_combined = []

        # list of (model_name, fold) tuples
        models_with_folds = []

        for model_name in self.l1_model_names:
            model_with_folds = []
            for fold in config.TRAIN_FOLDS:
                model_with_folds.append((model_name, fold))
            models_with_folds.append(model_with_folds)

        requests = []
        for model_with_folds in models_with_folds:
            for model_name, fold in model_with_folds:
                requests.append((model_name, fold, self.preprocess_l1_model_output))

        with utils.timeit_context('load all data'):
            results = self.pool.starmap(load_train_data, requests)

        for model_with_folds in models_with_folds:
            X_combined = []
            y_combined = []
            for model_name, fold in model_with_folds:
                X, y, video_ids = results[requests.index((model_name, fold))]
                logger.debug("Load", model_name, fold, X.shape, y.shape)
                X_combined.append(X)
                y_combined.append(y)

            X_all_combined.append(np.row_stack(X_combined))
            y_all_combined.append(np.row_stack(y_combined))

        X = np.column_stack(X_all_combined)
        y = y_all_combined[0]

        logger.debug("Training model", X.shape, y.shape)

        y_cat = np.argmax(y, axis=1)
        self._train(X, y_cat)


class SecondLevelModelXGBoost(SecondLevelModel):
    """
        XGBoost based L2 model implementation
    """
    def __init__(self, combined_model_name, l1_model_names):
        super().__init__(l1_model_names, preprocess_l1_model_output=preprocess_x_histogram)
        self.combined_model_name = combined_model_name
        self.model = None
        self.model_fn = config.MODEL_DIR / f"output/xgb_combined_{self.combined_model_name}.pkl"

    def _predict(self, X):
        if self.model is None:
            self.model = pickle.load(open(self.model_fn.resolve(), "rb"))
        return self.model.predict_proba(X)

    def _train(self, X, y):
        self.model = XGBClassifier(n_estimators=1600, objective='multi:softprob',
                                   learning_rate=0.03, silent=False, n_jobs=config.N_CORES)
        with utils.timeit_context('fit 1600 est'):
            self.model.fit(X, y)  # , eval_set=[(X_test, y_test)], early_stopping_rounds=20, verbose=True)
        pickle.dump(self.model, open(self.model_fn.resolve(), "wb"))


class SecondLevelModelMLP(SecondLevelModel):
    """
        Keras neural network based L2 model implementation
    """
    def __init__(self, combined_model_name, l1_model_names):
        super().__init__(l1_model_names, preprocess_l1_model_output=preprocess_x_sorted_bins)
        self.combined_model_name = combined_model_name
        self.model = None
        self.weights_fn = config.MODEL_DIR / f"output/nn_{self.combined_model_name}_full.pkl"

    def _build_model(self, input_size):
        input_data = Input(shape=(input_size,))
        x = input_data
        x = Dense(2048, activation='relu')(x)
        x = Dropout(0.75)(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(NB_CAT, activation='sigmoid', kernel_regularizer=l1(1e-5))(x)
        model = Model(inputs=input_data, outputs=x)
        return model

    def _predict(self, X):
        if self.model is None:
            self.model = self._build_model(input_size=X.shape[1])
            self.model.load_weights(self.weights_fn.resolve())
            self.model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        return self.model.predict(X)

    def _train(self, X, y):
        self.model = self._build_model(input_size=X.shape[1])
        self.model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()

        batch_size = 64

        def scheduler(epoch):
            if epoch < 32:
                return 1e-3
            if epoch < 48:
                return 4e-4
            if epoch < 80:
                return 1e-4
            return 1e-5

        self.model.fit(X, y,
                       batch_size=batch_size,
                       epochs=80,
                       verbose=1,
                       callbacks=[LearningRateScheduler(schedule=scheduler)])

        self.model.save_weights(self.weights_fn.resolve())


def predict(l1_model_results, profile):
    """
    :param l1_model_results: results of l1 model predictions as a dictionary
                             { model_name : ndarray(NB_CLIPS, NB_FRAMES, NB_CLASSES) }
    :return: class probabilities as ndarray with shape (NB_CLIPS, NB_CLASSES)
    """
    l1_model_names = config.PROFILES[profile]
    xgboost_model = SecondLevelModelXGBoost(l1_model_names=l1_model_names, combined_model_name='preset_'+profile)
    mlp_model = SecondLevelModelMLP(l1_model_names=l1_model_names, combined_model_name='preset_'+profile)

    xgboost_predictions = xgboost_model.predict(l1_model_results)
    mlp_predictions = mlp_model.predict(l1_model_results)

    return xgboost_predictions*0.5 + mlp_predictions*0.5
