import itertools
from string import ascii_letters

import numpy as np
import pandas as pd

from zamba.data.metadata import create_site_specific_splits, one_hot_to_labels


def test_site_specific_splits():
    group = pd.Series(
        list(
            itertools.chain.from_iterable(
                [element] * n for element, n in zip(ascii_letters[:20], range(20, 0, -1))
            )
        )
    )

    group_counts = group.value_counts()
    split = create_site_specific_splits(group, {"train": 3, "val": 1, "holdout": 1})
    assert split.value_counts().to_dict() == {
        "train": group_counts.iloc[0::5].sum()
        + group_counts.iloc[3::5].sum()
        + group_counts.iloc[4::5].sum(),
        "val": group_counts.iloc[1::5].sum(),
        "holdout": group_counts.iloc[2::5].sum(),
    }


def test_site_specific_splits_with_nulls():
    group = pd.Series(
        list(
            itertools.chain.from_iterable(
                [element] * n
                for element, n in zip([None] + list(ascii_letters[:20]), range(20, 0, -1))
            )
        )
    )
    group_counts = group.value_counts()

    split = create_site_specific_splits(
        group, {"train": 3, "val": 1, "holdout": 1}, random_state=2345
    )
    notnull_split = split[group.notnull()]
    assert notnull_split.value_counts().to_dict() == {
        "train": group_counts.iloc[0::5].sum()
        + group_counts.iloc[3::5].sum()
        + group_counts.iloc[4::5].sum(),
        "val": group_counts.iloc[1::5].sum(),
        "holdout": group_counts.iloc[2::5].sum(),
    }

    null_split = split[group.isnull()]
    null_split_group_counts = null_split.value_counts()
    assert null_split_group_counts.to_dict() == {"train": 9, "holdout": 7, "val": 4}


def test_one_hot_to_labels():
    values = np.eye(10)  # All rows have at least one species
    values[:5, 5:] = np.eye(5)  # First five rows have two species

    one_hot = pd.DataFrame(
        values,
        columns=[f"species_{name}" for name in ascii_letters[:10]],
        index=[f"data/{i}" for i in range(10)],
    )
    one_hot["extra_column"] = 1

    assert one_hot_to_labels(one_hot).to_dict(orient="records") == [
        {"filepath": "data/0", "label": "a"},
        {"filepath": "data/0", "label": "f"},
        {"filepath": "data/1", "label": "b"},
        {"filepath": "data/1", "label": "g"},
        {"filepath": "data/2", "label": "c"},
        {"filepath": "data/2", "label": "h"},
        {"filepath": "data/3", "label": "d"},
        {"filepath": "data/3", "label": "i"},
        {"filepath": "data/4", "label": "e"},
        {"filepath": "data/4", "label": "j"},
        {"filepath": "data/5", "label": "f"},
        {"filepath": "data/6", "label": "g"},
        {"filepath": "data/7", "label": "h"},
        {"filepath": "data/8", "label": "i"},
        {"filepath": "data/9", "label": "j"},
    ]
