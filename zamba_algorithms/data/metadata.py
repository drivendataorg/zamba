import itertools
import os
import sys
from uuid import uuid4

from loguru import logger
import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import Any, Dict, Optional, Union
from cloudpathlib import S3Path

from zamba_algorithms.data.config import (
    EUROPEAN_ZAMBA_LABELS,
    NEW_ZAMBA_LABELS,
    ORIGINAL_ZAMBA_LABELS,
)

s3p = S3Path("s3://drivendata-client-zamba")
PROCESSED_DIR = s3p / "data" / "processed"
INTERIM_DIR = s3p / "data" / "interim"

logger.remove()
log_level = os.environ["LOGURU_LEVEL"] if "LOGURU_LEVEL" in os.environ else "INFO"
logger.add(sys.stderr, level=log_level)


def clean_vernacular_name(x):
    if pd.isnull(x):
        return np.nan
    else:
        return x.lower().replace("-", " ").replace("_", " ").strip()


def vernacular_name_to_label(vernacular_name: str):
    if pd.isnull(vernacular_name):
        return np.nan
    else:
        return vernacular_name.lower().replace(" ", "_").strip()


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


class LoadMetadataConfig(BaseModel):
    zamba_label: str
    subset: Optional[str] = None
    subset_kwargs: Dict[str, Any] = dict()
    include_local_path: bool = True
    min_filesize_mb: Optional[float] = 0.5


def load_metadata(
    zamba_label: str = "original",
    subset: Optional[str] = None,
    subset_kwargs: Dict[str, Any] = dict(),
    include_local_path: bool = True,
    min_filesize_mb: Optional[float] = 0.5,
    seed: int = 5023,
):
    """Loads dataframe of video metadata including filepaths, location info, and zamba labels.
    Upon loading 'data/processed/unified_metadata.csv', zamba labels are one hot encoded and
    collapsed to one row per video.

    Args:
        zamba_label (str, optional): Zamba categories to use as labels. Defaults to "original".
            "New" breaks larger categories such as 'other_primate' into smaller subcategories.
        subset (str, optional): If "dev", return metadata file for development subset with "split"
            column indicating train, validation, and holdout split. If "half", return 66k
            metadata file that has the same val and holdout splits as the development set but uses
            the rest of the data for the train set, excluding sites used in val and holdout.
            Defaults to None.
        subset_kwargs (dict, optional): Optional parameters to pass to create_development_subset.
            Example use: {'n_blanks_per_country': 10, 'n_per_species': 5}. Defaults to dict().
        min_filesize_mb (float, optional): Filesize in megabytes of minimum filesize to include.
        seed (int): Random seed used for site-specific split creation in the half and full sets,
            and for sampling half of the full set.

    Returns:
        pd.DataFrame: Metadata dataframe
    """
    label_vars = ("zamba_label_original", "zamba_label_new", "european_zamba_label")
    with (PROCESSED_DIR / "unified_metadata.csv").open("r") as f:
        df = pd.read_csv(f, low_memory=False, parse_dates=["datetime"])

    if zamba_label == "original":
        zamba_var = "zamba_label_original"
        zamba_categories = ORIGINAL_ZAMBA_LABELS + ["unidentified"]

    elif zamba_label == "new":
        zamba_var = "zamba_label_new"
        zamba_categories = NEW_ZAMBA_LABELS + ["unidentified"]

    elif zamba_label == "european":
        zamba_var = "european_zamba_label"
        zamba_categories = EUROPEAN_ZAMBA_LABELS
        subset = "european"

        european_species_counts = df.loc[df.researcher == "hjalmar", zamba_var].value_counts()
        df = df.loc[
            df[zamba_var].isin(european_species_counts[european_species_counts >= 50].index)
        ]

    df.drop(columns=[column for column in label_vars if column != zamba_var], axis=1, inplace=True)

    # exclude files smaller than some minimum filesize
    if min_filesize_mb:
        df = df.loc[df.filesize_mb > min_filesize_mb]

    # remove nulls
    # for original: removes videos that don't fit into original categories (e.g. giraffe)
    # for new: removes videos that don't fit into new categories (e.g. non-spefic "other primate")
    df = df[df[zamba_var].notnull()]

    # pass in categories so we get columns for all possible species
    df[zamba_var] = pd.Categorical(df[zamba_var], categories=zamba_categories)

    # get dummies for zamba label
    ohe_df = pd.get_dummies(df.set_index("filepath")[zamba_var], prefix="species")
    species_cols = ohe_df.filter(regex="species_").columns

    # collapse labels to one row per video
    single_row_zamba_labels = ohe_df.groupby(ohe_df.index, sort=False).max()
    assert df.filepath.nunique() == len(single_row_zamba_labels)

    # get number of species in each video
    species_per_file = df.groupby("filepath")[zamba_var].nunique()
    # get number of OHE zamba labels per row
    labels_per_file = single_row_zamba_labels.sum(axis=1)
    # check these are the same (effectively captured multiple species per video)
    assert (species_per_file.loc[labels_per_file.index] == labels_per_file).all()

    # set unidentified videos to nan
    # this should only apply where ONLY species is unidentified (e.g not duiker + unidentified)
    unidentified_mask = (single_row_zamba_labels.species_unidentified == 1) & (
        single_row_zamba_labels[species_cols].sum(axis=1) == 1
    )
    single_row_zamba_labels.loc[unidentified_mask, species_cols] = np.nan
    single_row_zamba_labels.drop("species_unidentified", axis=1, inplace=True)

    # now collapse other metadata columns to single row
    single_row_metadata = df.drop(
        [
            "number_obj_detected",
            "comment",
            "genus",
            "species",
            "language_vernacular_name",
            zamba_var,
        ],
        axis=1,
    ).drop_duplicates()

    # should have one row per filepath for metadata df and OHE zamba label df
    assert np.all(single_row_metadata.filepath == single_row_zamba_labels.index)

    # join zamba labels to other metadata columns
    # both are in the same order but we merge on filepath for extra security
    metadata = single_row_metadata.join(single_row_zamba_labels, on="filepath")

    # TESTS
    # check we have one row per video
    assert len(metadata) == df.filepath.nunique() == metadata.filepath.nunique()

    # check we have expected number of species columns
    # minus 1 b/c we have dropped "unidentified" after converting to nans
    final_species_cols = metadata.filter(regex="species_").columns
    assert len(final_species_cols) == (len(zamba_categories) - 1)

    # check there are no other species labels for a blank video
    assert np.all(metadata[metadata.species_blank == 1][final_species_cols].sum(axis=1) == 1)

    # check species columns are either entirely null or entirely filled in
    assert np.all(
        metadata[final_species_cols].isnull().sum(axis=1).isin([0, len(final_species_cols)])
    )

    # check videos where only animal was unidentified species have nans for all species columns
    # use nunique for cases where there's a video with two rows, each of which are unidentified (different comments)
    single_species_vids = (
        df.groupby("filepath")
        .language_vernacular_name.nunique(dropna=False)
        .pipe(lambda x: x[x == 1])
        .index
    )
    assert (
        df[
            df.filepath.isin(single_species_vids) & (df[zamba_var] == "unidentified")
        ].filepath.nunique()
        == metadata[final_species_cols].isnull().all(axis=1).sum()
    )
    metadata = metadata.reset_index(drop=True)

    if include_local_path:
        metadata["local_path"] = metadata.filepath.apply(lambda x: S3Path(x).key)

    # drop bad videos
    bad_files = pd.read_csv(INTERIM_DIR / "failed_videos.csv")
    metadata = metadata[~metadata.local_path.isin(bad_files.local_path)]

    if subset == "dev":
        metadata = create_development_dataset(metadata, **subset_kwargs)

    elif subset == "european":
        metadata["split"] = create_site_specific_splits(
            metadata.n_transect, {"train": 2, "val": 1, "holdout": 1}
        )
        metadata = metadata[~metadata.filter(regex="species_").isnull().any(axis=1)]

    elif subset in ["full", "half"]:
        for location, location_metadata in metadata.groupby("location"):
            logger.debug(location)
            metadata.loc[location_metadata.index, "split"] = create_site_specific_splits(
                location_metadata.n_transect,
                proportions={"train": 8, "val": 1, "holdout": 1},
                random_state=seed,
            )

        # remove unidentified videos
        metadata = metadata[~metadata.filter(regex="species_").isnull().any(axis=1)]

        if subset == "half":
            metadata = metadata.sample(frac=0.5, random_state=seed)

    return metadata


def create_development_dataset(
    df, n_blanks_per_country=150, n_per_species=300, splits=True, seed=555
):
    """Subset metadata dataframe to development set. By default, this yields around 2k videos.

    Args:
        df pd.DataFrame): Metadata dataframe
        n_blanks_per_country (int, optional): Number of blanks to select per country. Defaults to 100.
        n_per_species (int, optional): Number of videos to select per species. Defaults to 100. This
        is split as evenly as possible across countries.

    Returns:
        pd.DataFrame: Metadata for selected subset.
    """
    rng = np.random.RandomState(seed)
    species_cols = list(df.filter(regex="species_").columns)

    # equal number of blanks per country (for countries with blanks)
    blanks = list(
        df[df.species_blank == 1]
        .groupby("country", group_keys=False)
        .sample(n_blanks_per_country, random_state=seed)
        .filepath
    )

    # draw a sample of n_per_species split as evenly as possible across countries
    all_species = []
    for col in np.setdiff1d(species_cols, ["species_blank"]):

        # dataframe for that species
        all_possible = df[df[col] == 1]

        # if have less than n_per_species samples total, keep all
        if len(all_possible) < n_per_species:
            all_species.extend(all_possible.filepath)

        # otherwise sample species from each country
        else:
            # number of countries with that species
            n = all_possible.country.nunique()
            # samples to draw per country
            s = np.int(n_per_species / n)

            selected_filepaths = (
                all_possible.groupby("country")
                .apply(lambda x: x.sample(s, random_state=seed) if len(x) >= s else x)
                .filepath.tolist()
            )

            # if n_per_species couldn't be split equally across countries, fill out the reamining number
            if (len(selected_filepaths) < n_per_species) & (len(all_possible) > n_per_species):

                remaning_n = n_per_species - len(selected_filepaths)

                selected_filepaths.extend(
                    rng.choice(
                        np.setdiff1d(all_possible.filepath, selected_filepaths),
                        size=remaning_n,
                        replace=False,
                    )
                )

                assert len(selected_filepaths) == n_per_species

            all_species.extend(selected_filepaths)

    dev_data = df[df.filepath.isin(blanks + all_species)].copy()

    # add weights at country-species level to enable weighted metrics
    def _weight_lookup(x, species_cols, weights_dict):
        weight = 0
        species = x[species_cols][x == 1].index
        for s in species:
            weight += weights_dict[s][x.country]
        return weight

    # proportion of all animal observations (not of all videos)
    original_species_proportions = (
        df.groupby(["country"])[species_cols].sum() / df[species_cols].sum().sum()
    )
    subset_species_proportions = (
        dev_data.groupby(["country"])[species_cols].sum() / dev_data[species_cols].sum().sum()
    )
    weights = (original_species_proportions / subset_species_proportions).to_dict()

    dev_data["weight"] = dev_data.apply(
        _weight_lookup, species_cols=species_cols, weights_dict=weights, axis=1
    )
    assert np.isclose(dev_data.weight.sum(), dev_data[species_cols].sum().sum())

    # A "split" column indicating site-aware train, test, and holdout sets
    if splits:
        return create_dev_set_train_test_split(dev_data, seed=seed)

    else:
        return dev_data


def create_dev_set_train_test_split(dev_data, seed):
    """Adds column to metadata indicating train/val/holdout set. Where possible, data is split by site
    so a transect is either all train or all test.

    Args:
        dev_data (pd.DataFrame): Development metadata.

    Returns:
        pd.DataFrame: Metadata with "split" column.
    """
    dfs = []
    # within location (national parks), split on transect where available
    for location in dev_data.location.unique():
        location_df = dev_data[dev_data.location == location].copy()

        location_df["split"] = create_site_specific_splits(
            location_df.n_transect, {"train": 1, "val": 1, "holdout": 1}
        )
        dfs.append(location_df)

    final_df = pd.concat(dfs)
    assert len(final_df) == len(dev_data)
    # each transect is entirely in one split
    assert np.all(final_df.groupby(["location", "n_transect"])["split"].nunique() == 1)
    return final_df


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
        train-val-holdout proportions::

        $ create_site_specific_splits(site, proportions={"train": 2, "val": 1, "holdout": 1})

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