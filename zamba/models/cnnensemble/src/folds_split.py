import pandas as pd
import numpy as np

from zamba.models.cnnensemble.src import config


def group_aware_split_to_folds(df, classes):
    """
    Split to folds in a group aware way

    Split policy:
    independent folds selection for each class

    distribute grouped items to fold, starting from smallest group to be added to the currently smallest fold
    to maintain approximate fold sizes. The biggest group of each class is distributed between folds.

    The black frames are distributed between folds randomly

    :param df:
    :param classes:
    :return:
    """
    fold_file_names = [list() for _ in config.TRAIN_FOLDS]
    groups = df['group__id'].unique()
    df['fold'] = 0

    def min_len_idx(arr):
        return np.argmin([len(x) for x in arr])

    # iterate from more common classes to less common
    # to reduce effect of multiple labels per video changing split of rare classes
    for cls_count, cls in sorted([(np.sum(df[c] > 0), c) for c in classes], reverse=True):
        cls_ds = df[df[cls] > 0]

        cls_fold_file_names = [[] for _ in config.TRAIN_FOLDS]

        file_name_by_group = [list(cls_ds[cls_ds['group__id'] == group].filename) for group in groups]
        file_name_by_group = sorted(file_name_by_group, key=len, reverse=True)

        # special case for blank category, distribute it randomly between folds
        # to avoid only non blank frames from some group in any fold
        if cls == 'blank':
            file_name_by_group = [sum(file_name_by_group, [])]

        # distribute grouped items to fold, starting from smallest group
        # to be added to the currently smallest fold
        while len(file_name_by_group) > 1:
            cls_fold_file_names[min_len_idx(cls_fold_file_names)] += file_name_by_group.pop()

        # split the last group to smaller chunks and distribute between folds in the same way
        assert len(file_name_by_group) == 1
        file_name_by_group = file_name_by_group[0]
        np.random.shuffle(file_name_by_group)
        chunk_size = max(1, len(file_name_by_group) // 128)

        while len(file_name_by_group) > 0:
            cls_fold_file_names[min_len_idx(cls_fold_file_names)] += file_name_by_group[:chunk_size]
            file_name_by_group = file_name_by_group[chunk_size:]

        # add smallest cls fold items to largest total fold items and so on
        src_idxs = np.argsort([len(items) for items in cls_fold_file_names])
        dst_idxs = np.argsort([len(items) for items in fold_file_names])[::-1]

        # print(cls, [len(l) for l in cls_fold_file_names])

        for src_idx, dst_idx in zip(src_idxs, dst_idxs):
            fold_file_names[dst_idx] += cls_fold_file_names[src_idx]
            df.loc[df['filename'].isin(set(cls_fold_file_names[src_idx])), 'fold'] = config.TRAIN_FOLDS[dst_idx]

    return df


def _join_groups(df, group_names_df):
    group_names_df['obfu_id'] = group_names_df['obfu_id'].astype(str) + '.mp4'
    group_names_df = group_names_df.set_index('obfu_id', drop=True)
    df = df.join(group_names_df[['group__id']], on='filename')
    return df


def prepare_group_aware_split_to_folds(df, group_names_df):
    return group_aware_split_to_folds(_join_groups(df, group_names_df), config.CLASSES)


def test_split_to_folds_is_balanced():
    training_set_labels_ds_full = pd.read_csv(config.TRAINING_SET_LABELS)
    group_names_df = pd.read_csv(config.TRAINING_GROUPS, low_memory=False)
    df = prepare_group_aware_split_to_folds(training_set_labels_ds_full, group_names_df)

    print_groups(df)

    for cls in config.CLASSES:
        cls_ds = df[df[cls] > 0]
        print(cls)
        for fold in [0] + config.TRAIN_FOLDS:
            print(fold, np.sum(cls_ds.fold == fold))

        sample_counts = [np.sum(cls_ds.fold == fold) for fold in config.TRAIN_FOLDS]
        large_fold = max(sample_counts)
        small_fold = min(sample_counts)

        assert large_fold - small_fold < max([2, large_fold//2])


def print_groups(training_set_labels_ds_full):

    for group in training_set_labels_ds_full['group__id'].unique():
        group_ds = training_set_labels_ds_full[training_set_labels_ds_full['group__id'] == group]
        print(group, len(group_ds))
        for cls in config.CLASSES:
            print(f'   {cls} ', len(group_ds[group_ds[cls] > 0]))
    print()

    for cls in config.CLASSES:
        cls_ds = training_set_labels_ds_full[training_set_labels_ds_full[cls] > 0]
        print(cls, len(cls_ds))

        for group in training_set_labels_ds_full['group__id'].unique():
            if group is None:
                print(f' n {group} ', len(cls_ds[cls_ds['group__id'].null]))
            else:
                print(f'   {group} ', len(cls_ds[cls_ds['group__id'] == group]))

