import os
from pathlib import Path
import pickle
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from zamba.models.cnnensemble.src import config, metrics, utils


NB_CAT = 24
N_CORES = 8

def preprocess_x(data: np.ndarray):
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


def load_train_data(model_name, fold, cache_prefix='xgb'):
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


def load_test_data(test_path, model_name, fold):
    X_raw = np.load(f'{test_path}/{model_name}_{fold}_combined.npy')
    X = preprocess_x(X_raw)
    return X


def load_test_data_from_std_path(model_name, fold):
    test_path = config.MODEL_DIR / 'output/prediction_test_frames'
    X_raw = np.load(f'{test_path}/{model_name}_{fold}_combined.npy')
    print('preprocess', model_name, fold)
    X = preprocess_x(X_raw)
    return X


def avg_probabilities():
    ds = pd.read_csv(config.TRAINING_SET_LABELS)
    data = ds.as_matrix()[:, 1:]
    return data.mean(axis=0)


def try_train_model_xgboost(model_name, fold):
    with utils.timeit_context('load data'):
        X, y, video_ids = load_train_data(model_name, fold)

    y_cat = np.argmax(y, axis=1)
    print(X.shape, y.shape)
    print(np.unique(y_cat))

    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.25, random_state=42)

    model = XGBClassifier(n_estimators=500, objective='multi:softprob', silent=True)
    model.fit(X, y_cat, eval_set=[(X_test, y_test)], early_stopping_rounds=20, verbose=True)

    prediction = model.predict_proba(X_test)
    if prediction.shape[1] == 23: # insert mission lion col
        prediction = np.insert(prediction, obj=12, values=0.0, axis=1)
    print(model.score(X_test, y_test))

    y_test_one_hot = np.eye(24)[y_test]
    print(y_test.shape, prediction.shape, y_test_one_hot.shape)
    print(metrics.pri_matrix_loss(y_test_one_hot, prediction))
    print(metrics.pri_matrix_loss(y_test_one_hot, np.clip(prediction, 0.001, 0.999)))
    delta = prediction - y_test_one_hot
    print(np.min(delta), np.max(delta), np.mean(np.abs(delta)), np.sum(np.abs(delta) > 0.5))

    avg_prob = avg_probabilities()
    avg_pred = np.repeat([avg_prob], y_test_one_hot.shape[0], axis=0)
    print(metrics.pri_matrix_loss(y_test_one_hot, avg_pred))
    print(metrics.pri_matrix_loss(y_test_one_hot, avg_pred*0.1 + prediction*0.9))


def model_xgboost(model_name, fold):
    with utils.timeit_context('load data'):
        X, y, video_ids = load_train_data(model_name, fold)

    y_cat = np.argmax(y, axis=1)
    print(X.shape, y.shape)
    print(np.unique(y_cat))

    model = XGBClassifier(n_estimators=400, objective='multi:softprob', learning_rate=0.1, silent=True)
    model.fit(X, y_cat)
    pickle.dump(model, open(Path(__file__).parent.parent / f"output/xgb_{model_name}_{fold}_full.pkl", "wb"))


def predict_on_test(model_name, fold, use_cache=False):
    model = pickle.load(open(Path(__file__).parent.parent / f"output/xgb_{model_name}_{fold}_full.pkl", "rb"))
    print(model)
    ds = pd.read_csv(config.SUBMISSION_FORMAT)
    classes = list(ds.columns)[1:]
    print(classes)

    data_dir = config.MODEL_DIR / f'output/prediction_test_frames'
    with utils.timeit_context('load data'):
        cache_fn = config.MODEL_DIR / f'output/prediction_test_frames/{model_name}_{fold}_cache.npy'
        if use_cache:
            X = np.load(cache_fn)
        else:
            X = load_test_data(data_dir, model_name, fold)
            np.save(cache_fn, X)
        print(X.shape)
    with utils.timeit_context('predict'):
        prediction = model.predict_proba(X)

    if prediction.shape[1] == 23:
        prediction = np.insert(prediction, obj=12, values=0.0, axis=1)

    for col, cls in enumerate(classes):
        ds[cls] = np.clip(prediction[:, col], 0.001, 0.999)
    os.makedirs(Path(__file__).parent.parent / 'submissions', exist_ok=True)
    ds.to_csv(Path(__file__).parent.parent / f'submissions/submission_one_model_{model_name}_{fold}.csv', index=False, float_format='%.7f')


def train_all_models_xgboost_combined(combined_model_name, models_with_folds):
    X_all_combined = []
    y_all_combined = []

    requests = []
    results = []
    for model_with_folds in models_with_folds:
        for model_name, fold in model_with_folds:
            requests.append((model_name, fold))
            # results.append(load_one_model(requests[-1]))

    pool = Pool(N_CORES)
    with utils.timeit_context('load all data'):
        results = pool.starmap(load_train_data, requests)

    for model_with_folds in models_with_folds:
        X_combined = []
        y_combined = []
        for model_name, fold in model_with_folds:
            X, y, video_ids = results[requests.index((model_name, fold))]
            print(model_name, fold, X.shape)
            X_combined.append(X)
            y_combined.append(y)

        X_all_combined.append(np.row_stack(X_combined))
        y_all_combined.append(np.row_stack(y_combined))

    X = np.column_stack(X_all_combined)
    y = y_all_combined[0]

    print(X.shape, y.shape)

    y_cat = np.argmax(y, axis=1)
    print(X.shape, y.shape)
    print(np.unique(y_cat))

    model = XGBClassifier(n_estimators=1600, objective='multi:softprob', learning_rate=0.03, silent=False)
    with utils.timeit_context('fit 1600 est'):
        model.fit(X, y_cat)  # , eval_set=[(X_test, y_test)], early_stopping_rounds=20, verbose=True)
    pickle.dump(model, open(Path(__file__).parent.parent / f"output/xgb_combined_{combined_model_name}.pkl", "wb"))


def load_test_data_one_model(request):
    data_dir, model_name, fold = request
    return fold, load_test_data(data_dir, model_name, fold)


def predict_on_test_combined(combined_model_name, models_with_folds):
    ds = pd.read_csv(config.SUBMISSION_FORMAT)
    classes = list(ds.columns)[1:]
    folds = [1, 2, 3, 4]

    X_combined = {fold: [] for fold in folds}
    requests = []

    for model_with_folds in models_with_folds:
        for data_model_name, data_fold in model_with_folds:
            data_dir = config.MODEL_DIR / f'output/prediction_test_frames'
            with utils.timeit_context('load data'):
                requests.append((data_dir, data_model_name, data_fold))
    pool = Pool(N_CORES)
    results = pool.map(load_test_data_one_model, requests)
    for data_fold, X in results:
        X_combined[data_fold].append(X)
    pickle.dump(X_combined, open(Path(Path(__file__).parent.parent, f"output/X_combined_xgb_{combined_model_name}.pkl").resolve(), "wb"))

    to_open = config.MODEL_DIR / f"output/xgb_combined_{combined_model_name}.pkl"
    model = pickle.load(open(to_open.resolve(), "rb"))
    print(model)

    predictions = []
    with utils.timeit_context('predict'):
        for fold in [1, 2, 3, 4]:
            X = np.column_stack(X_combined[fold])
            predictions.append(model.predict_proba(X))
            print('prediction', predictions[-1].shape)
            
    prediction = np.mean(np.array(predictions).astype(np.float64), axis=0)
    os.makedirs(Path(__file__).parent.parent / 'submissions', exist_ok=True)
    print('predictions', prediction.shape)

    for clip10 in [5, 4, 3, 2]:
        clip = 10 ** (-clip10)
        for col, cls in enumerate(classes):
            ds[cls] = np.clip(prediction[:, col]*(1-clip*2)+clip, clip, 1.0-clip)
        ds.to_csv(Path(__file__).parent.parent / f'submissions/submission_combined_models_xgboost_{combined_model_name}_clip_{clip10}.csv',
                  index=False,
                  float_format='%.8f')


def train_model_xgboost_combined_folds(combined_model_name, model_with_folds):
    X_combined = []
    y_combined = []

    for model_name, fold in model_with_folds:
        with utils.timeit_context('load data'):
            X, y, video_ids = load_train_data(model_name, fold)
            X_combined.append(X)
            y_combined.append(y)

    X = np.row_stack(X_combined)
    y = np.row_stack(y_combined)

    y_cat = np.argmax(y, axis=1)
    print(X.shape, y.shape)
    print(np.unique(y_cat))

    model = XGBClassifier(n_estimators=500, objective='multi:softprob', learning_rate=0.1, silent=True)
    with utils.timeit_context('fit 500 est'):
        model.fit(X, y_cat)
    pickle.dump(model, open(Path(__file__).parent.parent / f"output/xgb_combined_folds_{combined_model_name}.pkl", "wb"))


def train_combined_folds_models():
    for models in config.ALL_MODELS:
        combined_model_name = models[0][0] + '_combined'
        print('*' * 64)
        print(combined_model_name)
        print('*' * 64)
        train_model_xgboost_combined_folds(combined_model_name, models)


def predict_combined_folds_models():
    ds = pd.read_csv(config.SUBMISSION_FORMAT)
    classes = list(ds.columns)[1:]

    total_weight = 0.0
    result = np.zeros((ds.shape[0], NB_CAT))

    pool = Pool(N_CORES)

    for models in config.ALL_MODELS:
        combined_model_name = models[0][0] + '_combined'

        with utils.timeit_context('load 4 folds data'):
            X_for_folds = pool.starmap(load_test_data_from_std_path, models)

        model = pickle.load(open(Path(__file__).parent.parent / f"output/xgb_combined_folds_{combined_model_name}.pkl", "rb"))

        for (model_name, fold), X in zip(models, X_for_folds):
            with utils.timeit_context('predict'):
                prediction = model.predict_proba(X)
                weight = config.MODEL_WEIGHTS[model_name]
                result += prediction*weight
                total_weight += weight

    os.makedirs(Path(__file__).parent.parent / 'submissions', exist_ok=True)
    result /= total_weight

    for clip10 in [5, 4, 3, 2]:
        clip = 10 ** (-clip10)
        for col, cls in enumerate(classes):
            ds[cls] = np.clip(result[:, col] * (1 - clip * 2) + clip, clip, 1.0 - clip)
        ds.to_csv(Path(__file__).parent.parent / f'submissions/submission_combined_folds_models_xgboost_clip_{clip10}.csv',
                  index=False,
                  float_format='%.8f')


def train_all_single_fold_models():
    for models in config.ALL_MODELS:
        for model_name, fold in models:
            weights_fn = config.MODEL_DIR / "output" / f"xgb_{model_name}_{fold}_full.pkl"
            print(model_name, fold, weights_fn)
            if weights_fn.exists():
                print('skip existing file')
            else:
                with utils.timeit_context('train'):
                    model_xgboost(model_name, fold)


def predict_all_single_fold_models():
    ds = pd.read_csv(config.SUBMISSION_FORMAT)
    classes = list(ds.columns)[1:]

    total_weight = 0.0
    result = np.zeros((ds.shape[0], NB_CAT))

    requests = []
    for model_with_folds in config.ALL_MODELS:
        for model_name, fold in model_with_folds:
            requests.append((model_name, fold))
    pool = Pool(N_CORES)
    with utils.timeit_context('load all data'):
        results = pool.starmap(load_test_data_from_std_path, requests)

    for models in config.ALL_MODELS:
        for model_name, fold in models:
            model = pickle.load(open(Path(__file__).parent.parent / f"output/xgb_{model_name}_{fold}_full.pkl", "rb"))
            print(model_name, fold, model)

            with utils.timeit_context('load data'):
                X = results[requests.index((model_name, fold))]
                print(X.shape)

            with utils.timeit_context('predict'):
                prediction = model.predict_proba(X)
                if prediction.shape[1] == 23:
                    prediction = np.insert(prediction, obj=12, values=0.0, axis=1)
                weight = config.MODEL_WEIGHTS[model_name]
                result += prediction * weight
                total_weight += weight

    os.makedirs(Path(__file__).parent.parent / 'submissions', exist_ok=True)
    result /= total_weight

    for clip10 in [5, 4, 3, 2]:
        clip = 10 ** (-clip10)
        for col, cls in enumerate(classes):
            ds[cls] = np.clip(result[:, col] * (1 - clip * 2) + clip, clip, 1.0 - clip)
        ds.to_csv(Path(__file__).parent.parent / f'submissions/submission_single_folds_models_xgboost_clip_{clip10}.csv',
                  index=False,
                  float_format='%.8f')


def check_corr(sub1, sub2):
    print(sub1, sub2)
    s1 = pd.read_csv(Path(__file__).parent.parent / 'submissions/' + sub1)
    s2 = pd.read_csv(Path(__file__).parent.parent / 'submissions/' + sub2)
    for col in s1.columns[1:]:
        print(col, s1[col].corr(s2[col]))

    print('mean ', sub1, sub2, 'sub2-sub1')
    for col in s1.columns[1:]:
        print('{:20}  {:.6} {:.6} {:.6}'.format(col, s1[col].mean(), s2[col].mean(), s2[col].mean() - s1[col].mean()))


def main():
    with utils.timeit_context('train xgboost model'):
        predict_on_test_combined("2k_extra", models_with_folds=config.ALL_MODELS)
        predict_combined_folds_models()
        predict_all_single_fold_models()
