from pathlib import Path

import numpy as np
import pandas as pd
import os
import pickle
from zamba.models.cnnensemble.src import utils
# import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from zamba.models.cnnensemble.src import metrics
from zamba.models.cnnensemble.src import config

from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.python.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.python.keras.layers import Input, Lambda
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l1

from multiprocessing.pool import ThreadPool, Pool

NB_CAT = 24


def preprocess_x(data: np.ndarray):
    rows = []
    downsample = 4
    for row in data:
        items = []
        for col in range(row.shape[1]):
            sorted = np.sort(row[:, col])
            items.append(sorted.reshape(-1, downsample).mean(axis=1))
        rows.append(np.hstack(items))
    return np.array(rows)


def load_train_data(model_name, fold, cache_prefix='nn'):
    data_path = config.MODEL_DIR / 'output/prediction_train_frames'
    cache_fn = f'{data_path}/{cache_prefix}_{model_name}_{fold}_cache.npz'
    print(cache_fn, os.path.exists(cache_fn))

    if Path(cache_fn).exists():
        print('loading cache', cache_fn)
        cached = np.load(cache_fn)
        print('loaded cache')
        X, y, video_ids = cached['X'], cached['y'], cached['video_ids']
    else:
        raw_cache_fn = f'{data_path}/{model_name}_{fold}_combined.npz'
        print('loading raw cache', raw_cache_fn, os.path.exists(raw_cache_fn))
        cached = np.load(raw_cache_fn)
        X_raw, y, video_ids = cached['X_raw'], cached['y'], cached['video_ids']
        X = preprocess_x(X_raw)
        np.savez(cache_fn, X=X, y=y, video_ids=video_ids)
    return X, y, video_ids


def load_test_data_uncached(test_path, model_name, fold, video_ids):
    X_raw = []
    for video_id in tqdm(video_ids):
        ds = np.loadtxt(os.path.join(test_path, f'{model_name}_{fold}', video_id+'.csv'), delimiter=',', skiprows=1)
        # 0 col is frame number
        X_raw.append(ds[:, 1:])
    X_raw = np.array(X_raw)
    X = preprocess_x(X_raw)
    return X


def load_test_data(test_path, model_name, fold):
    X_raw = np.load(f'{test_path}/{model_name}_{fold}_combined.npy')
    X = preprocess_x(X_raw)
    return X


def avg_probabilities():
    ds = pd.read_csv(config.TRAINING_SET_LABELS)
    data = ds.as_matrix()[:, 1:]
    return data.mean(axis=0)


def model_nn(input_size):
    input_data = Input(shape=(input_size,))
    x = input_data
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(NB_CAT, activation='sigmoid', kernel_regularizer=l1(1e-5))(x)
    model = Model(inputs=input_data, outputs=x)
    return model


def model_nn_combined(input_size):
    input_data = Input(shape=(input_size,))
    x = input_data
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.75)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(NB_CAT, activation='sigmoid', kernel_regularizer=l1(1e-5))(x)
    model = Model(inputs=input_data, outputs=x)
    return model


def try_train_model_nn(model_name, fold):
    with utils.timeit_context('load data'):
        X, y, video_ids = load_train_data(model_name, fold)

    print(X.shape, y.shape)
    model = model_nn(input_size=X.shape[1])
    model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    batch_size = 64

    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=128,
              verbose=1,
              validation_data=[X_test, y_test],
              callbacks=[ReduceLROnPlateau(factor=0.2, verbose=True, min_lr=1e-6)])

    prediction = model.predict(X_test)

    print(y_test.shape, prediction.shape)
    print(metrics.pri_matrix_loss(y_test, prediction))
    print(metrics.pri_matrix_loss(y_test, np.clip(prediction, 0.001, 0.999)))
    delta = prediction - y_test
    print(np.min(delta), np.max(delta), np.mean(np.abs(delta)), np.sum(np.abs(delta) > 0.5))

    # avg_prob = avg_probabilities()
    # # print(avg_prob)
    # avg_pred = np.repeat([avg_prob], y_test_one_hot.shape[0], axis=0)
    # print(metrics.pri_matrix_loss(y_test_one_hot, avg_pred))
    # print(metrics.pri_matrix_loss(y_test_one_hot, avg_pred*0.1 + prediction*0.9))


def train_model_nn(model_name, fold, load_cache=True):
    with utils.timeit_context('load data'):
        X, y, video_ids = load_train_data(model_name, fold)

    print(X.shape, y.shape)
    model = model_nn(input_size=X.shape[1])
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    batch_size = 64

    def cheduler(epoch):
        if epoch < 32:
            return 1e-3
        if epoch < 48:
            return 4e-4
        if epoch < 80:
            return 1e-4
        return 1e-5

    model.fit(X, y,
              batch_size=batch_size,
              epochs=128,
              verbose=1,
              callbacks=[LearningRateScheduler(schedule=cheduler)])

    model.save_weights(Path(__file__).parent.parent / f"output/nn1_{model_name}_{fold}_full.pkl")


def train_all_single_fold_models():
    for models in config.ALL_MODELS:
        for model_name, fold in models:
            weights_fn = config.MODEL_DIR / f"output/nn1_{model_name}_{fold}_full.pkl"
            print(model_name, fold, weights_fn)
            if weights_fn.exists():
                print('skip existing file')
            else:
                train_model_nn(model_name, fold)


def predict_all_single_fold_models():
    ds = pd.read_csv(config.SUBMISSION_FORMAT)
    classes = list(ds.columns)[1:]

    total_weight = 0.0
    result = np.zeros((ds.shape[0], NB_CAT))

    data_dir = config.MODEL_DIR / 'output/prediction_test_frames/'

    for models in config.ALL_MODELS:
        for model_name, fold in models:
            weights_fn = config.MODEL_DIR / f"output/nn1_{model_name}_{fold}_full.pkl"
            print(model_name, fold, weights_fn)

            with utils.timeit_context('load data'):
                X = load_test_data(data_dir, model_name, fold)
                print(X.shape)

            model = model_nn(input_size=X.shape[1])
            model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
            model.load_weights(weights_fn)

            with utils.timeit_context('predict'):
                prediction = model.predict(X)
                weight = config.MODEL_WEIGHTS[model_name]
                result += prediction * weight
                total_weight += weight

    os.makedirs(Path(__file__).parent.parent / 'submissions', exist_ok=True)
    result /= total_weight

    for clip10 in [5, 4, 3, 2]:
        clip = 10 ** (-clip10)
        for col, cls in enumerate(classes):
            ds[cls] = np.clip(result[:, col] * (1 - clip * 2) + clip, clip, 1.0 - clip)
        ds.to_csv(Path(__file__).parent.parent / f'submissions/submission_single_folds_models_nn_clip_{clip10}.csv',
                  index=False,
                  float_format='%.8f')


def train_model_nn_combined_folds(combined_model_name, model_with_folds, load_cache=True):
    X_combined = []
    y_combined = []

    for model_name, fold in model_with_folds:
        with utils.timeit_context('load data'):
            X, y, video_ids = load_train_data(model_name, fold)
            X_combined.append(X)
            y_combined.append(y)

    X = np.row_stack(X_combined)
    y = np.row_stack(y_combined)

    print(X.shape, y.shape)
    model = model_nn(input_size=X.shape[1])
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    batch_size = 64

    def cheduler(epoch):
        if epoch < 32:
            return 1e-3
        if epoch < 48:
            return 4e-4
        if epoch < 80:
            return 1e-4
        return 1e-5

    model.fit(X, y,
              batch_size=batch_size,
              epochs=128,
              verbose=1,
              callbacks=[LearningRateScheduler(schedule=cheduler)])

    model.save_weights(Path(__file__).parent.parent / f"output/nn_combined_folds_{combined_model_name}.pkl")


def train_all_models_nn_combined(combined_model_name, models_with_folds):
    X_all_combined = []
    y_all_combined = []

    for model_with_folds in models_with_folds:
        X_combined = []
        y_combined = []
        for model_name, fold in model_with_folds:
            with utils.timeit_context('load data'):
                X, y, video_ids = load_train_data(model_name, fold)
                X_combined.append(X)
                y_combined.append(y)

        X_all_combined.append(np.row_stack(X_combined))
        y_all_combined.append(np.row_stack(y_combined))

    X = np.column_stack(X_all_combined)
    y = y_all_combined[0]

    print(X.shape, y.shape)
    model = model_nn_combined(input_size=X.shape[1])
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    batch_size = 64

    def cheduler(epoch):
        if epoch < 32:
            return 1e-3
        if epoch < 48:
            return 4e-4
        if epoch < 80:
            return 1e-4
        return 1e-5

    model.fit(X, y,
              batch_size=batch_size,
              epochs=80,
              verbose=1,
              callbacks=[LearningRateScheduler(schedule=cheduler)])

    model.save_weights(Path(__file__).parent.parent / f"output/nn_{combined_model_name}_full.pkl")


def try_train_all_models_nn_combined(models_with_folds):
    X_all_combined = []
    y_all_combined = []

    for model_with_folds in models_with_folds:
        X_combined = []
        y_combined = []
        for model_name, fold in model_with_folds:
            with utils.timeit_context('load data'):
                X, y, video_ids = load_train_data(model_name, fold)
                X_combined.append(X)
                y_combined.append(y)

        X_all_combined.append(np.row_stack(X_combined))
        y_all_combined.append(np.row_stack(y_combined))

    X = np.column_stack(X_all_combined)
    y = y_all_combined[0]

    print(X.shape, y.shape)
    model = model_nn_combined(input_size=X.shape[1])
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    batch_size = 64

    def cheduler(epoch):
        if epoch < 32:
            return 1e-3
        if epoch < 48:
            return 4e-4
        if epoch < 80:
            return 1e-4
        return 1e-5

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=128,
              verbose=1,
              validation_data=[X_test, y_test],
              callbacks=[LearningRateScheduler(schedule=cheduler)])

    prediction = model.predict(X_test)

    print(y_test.shape, prediction.shape)
    print(metrics.pri_matrix_loss(y_test, prediction))
    print(metrics.pri_matrix_loss(y_test, np.clip(prediction, 0.001, 0.999)))
    print(metrics.pri_matrix_loss(y_test, np.clip(prediction, 0.0001, 0.9999)))
    delta = prediction - y_test
    print(np.min(delta), np.max(delta), np.mean(np.abs(delta)), np.sum(np.abs(delta) > 0.5))


def predict_on_test_combined(combined_model_name, models_with_folds):
    ds = pd.read_csv(config.SUBMISSION_FORMAT)
    classes = list(ds.columns)[1:]
    folds = [1, 2, 3, 4]

    X_combined = {fold: [] for fold in folds}
    for model_with_folds in models_with_folds:
        for data_model_name, data_fold in model_with_folds:
            data_dir = config.MODEL_DIR / f'output/prediction_test_frames/'
            with utils.timeit_context('load data'):
                X_combined[data_fold].append(load_test_data(data_dir, data_model_name, data_fold))
                # print(X_combined[-1].shape)
    pickle.dump(X_combined, open(Path(__file__).parent.parent / f"output/X_combined_{combined_model_name}.pkl", "wb"))

    # X_combined = pickle.load(open(Path(__file__).parent.parent / f"output/X_combined_{combined_model_name}.pkl", 'rb'))

    model = model_nn_combined(input_size=np.column_stack(X_combined[1]).shape[1])
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.load_weights(Path(__file__).parent.parent / f"output/nn_{combined_model_name}_full.pkl")

    predictions = []
    with utils.timeit_context('predict'):
        for fold in [1, 2, 3, 4]:
            X = np.column_stack(X_combined[fold])
            predictions.append(model.predict(X))

    prediction = np.mean(np.array(predictions).astype(np.float64), axis=0)
    os.makedirs(Path(__file__).parent.parent / 'submissions', exist_ok=True)

    for clip10 in [5, 4, 3, 2]:
        clip = 10 ** (-clip10)
        for col, cls in enumerate(classes):
            ds[cls] = np.clip(prediction[:, col]*(1-clip*2)+clip, clip, 1.0-clip)
        ds.to_csv(Path(__file__).parent.parent / f'submissions/submission_combined_models_nn_{combined_model_name}_clip_{clip10}.csv',
                  index=False,
                  float_format='%.8f')


def predict_on_test(model_name, fold, data_model_name=None, data_fold=None):
    ds = pd.read_csv(config.SUBMISSION_FORMAT)
    classes = list(ds.columns)[1:]

    if data_model_name is None:
        data_model_name = model_name

    if data_fold is None:
        data_fold = fold

    with utils.timeit_context('load data'):
        X = load_test_data(Path(__file__).parent.parent / 'output/prediction_test_frames/', data_model_name, data_fold)
        print(X.shape)

    model = model_nn(input_size=X.shape[1])
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.load_weights(Path(__file__).parent.parent / f"output/nn1_{model_name}_{fold}_full.pkl")

    with utils.timeit_context('predict'):
        prediction = model.predict(X)

    for col, cls in enumerate(classes):
        ds[cls] = np.clip(prediction[:, col], 0.001, 0.999)
    os.makedirs(Path(__file__).parent.parent / 'submissions', exist_ok=True)
    ds.to_csv(Path(__file__).parent.parent / f'submissions/submission_one_model_nn_{model_name}_{data_fold}.csv', index=False, float_format='%.7f')


def check_corr(sub1, sub2):
    print(sub1, sub2)
    s1 = pd.read_csv(Path(Path(__file__).parent.parent / 'submissions/', sub1))
    s2 = pd.read_csv(Path(Path(__file__).parent.parent / 'submissions/', sub2))
    for col in s1.columns[1:]:
        print(col, s1[col].corr(s2[col]))

    print('mean ', sub1, sub2, 'sub2-sub1')
    for col in s1.columns[1:]:
        print('{:20}  {:.6} {:.6} {:.6}'.format(col, s1[col].mean(), s2[col].mean(), s2[col].mean() - s1[col].mean()))


def combine_submissions1():
    for clip10 in [5, 4]:
        sources = [
            (f'submission_combined_models_nn_combined_extra_dr075_clip_{clip10}.csv', 4.0),
            (f'submission_combined_folds_models_nn_clip_{clip10}.csv', 1.0),
            (f'submission_single_folds_models_nn_clip_{clip10}.csv', 1.0),

            (f'submission_combined_models_xgboost_2k_extra_clip_{clip10}.csv', 4.0),
            (f'submission_combined_folds_models_xgboost_clip_{clip10}.csv', 1.0),
            (f'submission_single_folds_models_xgboost_clip_{clip10}.csv', 1.0),
        ]
        total_weight = sum([s[1] for s in sources])
        ds = pd.read_csv(config.SUBMISSION_FORMAT)
        for src_fn, weight in sources:
            pth = Path(Path(__file__).parent.parent / 'submissions/', src_fn)
            src = pd.read_csv(pth)
            for col in ds.columns[1:]:
                ds[col] += src[col]*weight/total_weight
        pth = config.MODEL_DIR / f'submissions/submission_59_avg_xgb_nn_all_4_1_1_clip_{clip10}.csv'
        ds.to_csv(pth, index=False, float_format='%.8f')


def combine_submissions2():
    for clip10 in [5, 4, 3]:
        sources = [
            (f'submission_combined_models_nn_combined_extra_dr075_clip_{clip10}.csv', 4.0),
            (f'submission_combined_folds_models_nn_clip_{clip10}.csv', 1.0),
            (f'submission_single_folds_models_nn_clip_{clip10}.csv', 1.0),

            (f'submission_combined_models_xgboost_2k_extra_clip_{clip10}.csv', 4.0),
            (f'submission_combined_folds_models_xgboost_clip_{clip10}.csv', 1.0),
            (f'submission_single_folds_models_xgboost_clip_{clip10}.csv', 1.0),

        ]
        total_weight = sum([s[1] for s in sources])
        ds = pd.read_csv(config.SUBMISSION_FORMAT)
        for src_fn, weight in sources:
            check_corr('submission_combined_models_xgboost_2k_extra_clip_4.csv', src_fn)
            pth = Path(Path(__file__).parent.parent / 'submissions/', src_fn)
            src = pd.read_csv(pth)
            for col in ds.columns[1:]:
                ds[col] += src[col]*weight/total_weight
        pth = config.MODEL_DIR / f'submissions/submission_60_avg_xgb_nn_lgb_all_4_1_1_clip_{clip10}.csv'
        ds.to_csv(pth, index=False, float_format='%.8f')
        return ds


def train_combined_folds_models():
    for models in config.ALL_MODELS:
        combined_model_name = models[0][0] + '_combined'
        print('*' * 64)
        print(combined_model_name)
        print('*' * 64)
        train_model_nn_combined_folds(combined_model_name, models, load_cache=True)


def predict_combined_folds_models():
    ds = pd.read_csv(config.SUBMISSION_FORMAT)
    classes = list(ds.columns)[1:]

    total_weight = 0.0
    result = np.zeros((ds.shape[0], NB_CAT))

    data_dir = config.MODEL_DIR / 'output/prediction_test_frames/'
    pool = ThreadPool(8)

    for models in config.ALL_MODELS:
        combined_model_name = models[0][0] + '_combined'

        def load_data(request):
            model_name, fold = request
            return load_test_data(data_dir, model_name, fold)

        with utils.timeit_context('load 4 folds data'):
            X_for_folds = pool.map(load_data, models)

        model = model_nn(input_size=X_for_folds[0].shape[1])
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        model.load_weights(Path(__file__).parent.parent / f"output/nn_combined_folds_{combined_model_name}.pkl")

        for (model_name, fold), X in zip(models, X_for_folds):
            with utils.timeit_context('predict'):
                prediction = model.predict(X)
                weight = config.MODEL_WEIGHTS[model_name]
                result += prediction*weight
                total_weight += weight

    os.makedirs(Path(__file__).parent.parent / 'submissions', exist_ok=True)
    result /= total_weight

    for clip10 in [5, 4, 3, 2]:
        clip = 10 ** (-clip10)
        for col, cls in enumerate(classes):
            ds[cls] = np.clip(result[:, col] * (1 - clip * 2) + clip, clip, 1.0 - clip)
        ds.to_csv(Path(__file__).parent.parent / f'submissions/submission_combined_folds_models_nn_clip_{clip10}.csv',
                  index=False,
                  float_format='%.8f')


def predict_unused_clips(data_model_name, data_fold, combined_model_name):
    ds = pd.read_csv(config.SUBMISSION_FORMAT)
    classes = list(ds.columns)[1:]

    data_dir = config.MODEL_DIR / f'output/prediction_unused_frames/'
    video_ids = [fn[:-4] for fn in os.listdir(data_dir) if fn.endswith('.csv')]

    with utils.timeit_context('load data'):
        X = load_test_data_uncached(data_dir, data_model_name, data_fold, video_ids)

    model = model_nn(input_size=X.shape[1])
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.load_weights(Path(__file__).parent.parent / f"output/nn1_{combined_model_name}_0_full.pkl")

    with utils.timeit_context('predict'):
        prediction = model.predict(X)

    ds = pd.DataFrame(data={'filename': video_ids})

    for col, cls in enumerate(classes):
        ds[cls] = prediction[:, col]  # np.clip(prediction[:, col], 0.001, 0.999)
    # os.makedirs(Path(__file__).parent.parent / 'submissions', exist_ok=True)
    ds.to_csv(Path(__file__).parent.parent / f'output/prediction_unused_frames/{data_model_name}_{data_fold}.csv', index=False, float_format='%.7f')


def main():
    with utils.timeit_context('predict nn model'):
        # train_all_models_nn_combined('combined_extra_dr075', config.ALL_MODELS)
        predict_on_test_combined('combined_extra_dr075', config.ALL_MODELS)

        #  train_combined_folds_models()
        predict_combined_folds_models()

        # train_all_single_fold_models()
        predict_all_single_fold_models()

    combine_submissions1()
    preds2 = combine_submissions2()
    return preds2
