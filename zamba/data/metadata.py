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
    labels: pd.DataFrame = None,  # <--- Add this line
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

    # Calculate target ratios from the proportions dict (e.g., {'train': 2, 'val': 1} -> {train: 0.66, val: 0.33})
    target_names = list(proportions.keys())
    total_parts = sum(proportions.values())
    target_ratios = {k: v / total_parts for k, v in proportions.items()}
    
    assignments = {}

    if labels is None:
        # FALLBACK: Original Round-Robin logic if no labels are provided
        sites = site.value_counts(dropna=True).sort_values(ascending=False).index
        n_subgroups = sum(proportions.values())
        for i, subset in enumerate(
            roundrobin(*([subset] * proportions[subset] for subset in proportions))
        ):
            for group in sites[i::n_subgroups]:
                assignments[group] = subset
    else:
        # FIX: Label-Aware Iterative Stratification
        # 1. Aggregate species counts per site
        site_labels = labels.groupby(site).sum()
        
        # 2. Rank species by rarity (least frequent globally is prioritized)
        label_rarity = site_labels.sum().sort_values().index
        
        remaining_sites = set(site_labels.index)
        split_counts = pd.DataFrame(0, index=target_names, columns=site_labels.columns)

        # 3. Greedy Assignment: Process rarest species first
        for label in label_rarity:
            # Find sites containing this rare species that aren't assigned yet
            relevant_sites = [s for s in remaining_sites if site_labels.loc[s, label] > 0]
            
            for s in relevant_sites:
                best_split = None
                max_diff = -float('inf')
                
                # Assign to the split that is furthest behind its target for this specific label
                for split in target_names:
                    current_total_for_label = split_counts.sum(axis=0)[label]
                    target_count = current_total_for_label * target_ratios[split]
                    diff = target_count - split_counts.loc[split, label]
                    
                    if diff > max_diff:
                        max_diff = diff
                        best_split = split
                
                assignments[s] = best_split
                split_counts.loc[best_split] += site_labels.loc[s]
                remaining_sites.remove(s)

        # 4. Cleanup: Assign any remaining sites (those with no labels) to the first split
        for s in list(remaining_sites):
            assignments[s] = target_names[0]

    # Map the site-level assignments back to every row in the original Series
    return site.map(assignments)

    


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
