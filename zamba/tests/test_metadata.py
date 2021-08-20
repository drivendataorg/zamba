import itertools
from string import ascii_letters

import pandas as pd

from zamba_algorithms.data.metadata import create_site_specific_splits


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
