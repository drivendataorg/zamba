import itertools
from uuid import uuid4

from loguru import logger
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union


def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # From https://docs.python.org/3/library/itertools.html#recipes
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = itertools.cycle(itertools.islice(nexts, num_active))


def create_site_specific_splits(
    site: pd.Series,
    proportions: Dict[str, int],
    random_state: Optional[Union[int, np.random.mtrand.RandomState]] = 989,
):
    """Splits sites into distinct groups whose sizes roughly matching the given proportions. Null
    sites are randomly assigned to groups using the provided proportions.

    Args:
        site (pd.Series): A series of sites, one element per observation,
        proportions (dict): A dict whose keys are the resulting groups and whose values are the
            rough proportion of data in each group.
        seed (int): Seed for random split of null sites.

    Example:
        Split data into groups where each site is in one and only one group with roughly 50-25-25
        train-val-holdout proportions.

        >>> create_site_specific_splits(site, proportions={"train": 2, "val": 1, "holdout": 1})

    Returns:
        pd.Series: A series containing the resulting split, one element per observation.

    """

    assignments = {}
    sites = site.value_counts(dropna=True).sort_values(ascending=False).index
    n_subgroups = sum(proportions.values())
    for i, subset in enumerate(
        roundrobin(*([subset] * proportions[subset] for subset in proportions))
    ):
        for group in sites[i::n_subgroups]:
            assignments[group] = subset

    # Divide null sites among the groups
    null_sites = site.isnull()
    if null_sites.sum() > 0:
        logger.debug(f"{null_sites.sum():,} null sites randomly assigned to groups.")
        null_groups = []
        for group, group_proportion in proportions.items():
            null_group = f"{group}-{uuid4()}"
            null_groups.append(null_group)
            assignments[null_group] = group

        rng = (
            np.random.RandomState(random_state) if isinstance(random_state, int) else random_state
        )
        site = site.copy()
        site.loc[null_sites] = rng.choice(
            null_groups,
            p=np.asarray(list(proportions.values())) / sum(proportions.values()),
            size=null_sites.sum(),
            replace=True,
        )

    return site.replace(assignments)


def one_hot_to_labels(
    one_hot: pd.DataFrame, column_prefix: Optional[str] = r"species_"
) -> pd.DataFrame:
    if column_prefix:
        one_hot = one_hot.filter(regex=column_prefix)
        # remove prefix
        one_hot.columns = [c.split(column_prefix, 1)[1] for c in one_hot.columns]

    one_hot.index = one_hot.index.rename("filepath")
    one_hot.columns = one_hot.columns.rename("label")

    labels = (one_hot == 1).stack()
    labels = labels[labels]
    return labels.reset_index().drop(0, axis=1)
