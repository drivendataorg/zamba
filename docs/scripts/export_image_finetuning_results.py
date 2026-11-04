"""Export a compact snapshot of the image fine-tuning experiments for the docs.

Run from the zamba repository root:

    ../zamba-finetune-experiments/.venv/bin/python \
        docs/scripts/export_image_finetuning_results.py \
        --experiments-dir ../zamba-finetune-experiments

The source experiment repository is intentionally not a documentation build
dependency. The generated JSON and CSV files are committed with the docs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pandas as pd

CLASS_CONDITIONS = ["finetune-lila", "balanced-aug-lila"]
SEEN_LABELS = {
    "species": "species seen",
    "genus": "genus seen",
    "family": "family/order seen",
    "order": "family/order seen",
    "unseen": "unseen",
}

RUN_COLUMNS = [
    "dataset",
    "condition",
    "tier",
    "base_model",
    "regime",
    "overall_accuracy",
    "macro_f1",
    "macro_accuracy_all_classes",
    "macro_accuracy_reliable_classes",
    "n_classes",
    "n_reliable_classes",
    "n_test_images",
    "n_scored_crops",
    "n_no_detection",
    "scored_on",
]

CLASS_COLUMNS = [
    "dataset",
    "label",
    "condition",
    "tier",
    "accuracy",
    "ci_low",
    "ci_high",
    "support",
    "train_n",
    "val_n",
    "test_n",
    "n_loc",
    "det_rate",
    "lila_train_seen",
    "overlap",
    "tax_level",
    "quality",
    "genuine",
    "reliable",
    "det_ok",
    "apparent_size",
    "md_miss_rate",
]

CONDITION_LABELS = {
    "finetune-lila": "Fine-tune lila.science",
    "finetune-speciesnet": "Fine-tune SpeciesNet",
    "scratch-lila-arch": "ImageNet start, lila architecture",
    "scratch-speciesnet-arch": "ImageNet start, SpeciesNet architecture",
    "linearprobe-lila": "Linear probe, lila.science",
    "linearprobe-speciesnet": "Linear probe, SpeciesNet",
    "balanced-aug-lila": "Balanced sampling + augmentation, lila.science",
    "linearprobe-bioclip": "Linear probe, BioCLIP",
    "zeroshot-pe-core-g": "Zero-shot PE-Core-G",
    "linearprobe-dinov3-cls": "Linear probe, DINOv3 CLS",
    "linearprobe-dinov3-meanpatch": "Linear probe, DINOv3 mean patch",
    "linearprobe-dinov3-concat": "Linear probe, DINOv3 concatenated",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/docs/assets/data"),
    )
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="Export from a dirty experiment worktree and record source_dirty=true.",
    )
    return parser.parse_args()


def records(df: pd.DataFrame) -> list[dict]:
    """Return JSON-safe records with stable precision."""
    out = df.copy()
    float_columns = out.select_dtypes(include="float").columns
    out[float_columns] = out[float_columns].round(6)
    return json.loads(out.to_json(orient="records"))


def source_revision(experiments_dir: Path, allow_dirty: bool = False) -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(experiments_dir), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(experiments_dir), "status", "--porcelain"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Could not inspect source repository: {experiments_dir}") from exc

    dirty = bool(status)
    if dirty and not allow_dirty:
        raise RuntimeError(
            "Experiment source worktree is dirty. Commit or stash its changes, or pass "
            "--allow-dirty-source to mark the snapshot explicitly."
        )
    return commit, dirty


def load_taxonomy_metadata(C, P) -> pd.DataFrame:
    """Load label rank and quality without depending on the report's private helpers."""
    taxonomy_files = [P.TAXONOMY_CSV, C.META_DIR / "camtrapdp-taxonomy-mapping.csv"]
    taxonomy_frames = [pd.read_csv(path) for path in taxonomy_files if path.exists()]
    if not taxonomy_frames:
        raise FileNotFoundError("No taxonomy mapping was found in the experiment repository.")

    taxonomy = pd.concat(taxonomy_frames, ignore_index=True)
    taxonomy["query"] = taxonomy["query"].astype(str).str.lower().str.strip()
    rows = []
    for slug, dataset_name in C.DATASETS.items():
        dataset_taxonomy = taxonomy[taxonomy["dataset_name"] == dataset_name].drop_duplicates(
            "query"
        )
        rank = dataset_taxonomy.set_index("query")["taxonomy_level"].to_dict()
        quality = P.label_quality(slug)
        labels = set(rank) | set(quality)
        rows.extend(
            {
                "dataset": slug,
                "label_key": str(label).lower().strip(),
                "tax_level": rank.get(str(label).lower().strip()),
                "quality": quality.get(label),
            }
            for label in labels
        )
    return pd.DataFrame(rows)


def load_location_counts(C) -> pd.DataFrame:
    rows = []
    for slug in C.DATASETS:
        catalog_path = C.CATALOG_DIR / f"{slug}.parquet"
        if not catalog_path.exists():
            continue
        catalog = pd.read_parquet(catalog_path, columns=["label", "location"])
        catalog["label_key"] = catalog["label"].astype(str).str.lower().str.strip()
        counts = catalog.groupby("label_key").location.nunique().rename("n_loc").reset_index()
        counts["dataset"] = slug
        rows.append(counts[["dataset", "label_key", "n_loc"]])
    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["dataset", "label_key", "n_loc"])
    )


def load_detection_rates(C) -> pd.DataFrame:
    rows = []
    for slug in C.DATASETS:
        detection_path = C.DETECT_DIR / f"{slug}.parquet"
        selection_path = C.SPLIT_DIR / f"{slug}_selection.parquet"
        if not (detection_path.exists() and selection_path.exists()):
            continue
        detections = pd.read_parquet(detection_path, columns=["file_name", "has_detection"])
        selection = pd.read_parquet(selection_path, columns=["file_name", "label"])
        rates = (
            selection.merge(detections, on="file_name", how="inner")
            .groupby("label")
            .has_detection.mean()
            .rename("det_rate")
            .reset_index()
        )
        rates["label_key"] = rates["label"].astype(str).str.lower().str.strip()
        rates["dataset"] = slug
        rows.append(rates[["dataset", "label_key", "det_rate"]])
    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["dataset", "label_key", "det_rate"])
    )


def load_class_results(C, P, R) -> pd.DataFrame:
    """Build the documented class table from stable experiment data APIs and files."""
    per_class = R.load_per_class()
    if per_class.empty:
        return per_class

    frames = []
    for slug in per_class.dataset.unique():
        provenance = P.class_provenance(slug)[["label", "lila_train_seen"]]
        counts = R.per_class_counts(slug)
        subset = (
            per_class[per_class.dataset == slug]
            .merge(provenance, on="label", how="left")
            .merge(counts, on="label", how="left")
        )
        subset["train_n"] = [
            row.get(f"train_n_{int(tier)}")
            for row, tier in zip(subset.to_dict("records"), subset.tier)
        ]
        frames.append(subset)

    results = pd.concat(frames, ignore_index=True)
    results["overlap"] = results["lila_train_seen"].map(SEEN_LABELS).fillna("unseen")
    results["label_key"] = results.label.astype(str).str.lower().str.strip()
    results = results.merge(load_taxonomy_metadata(C, P), on=["dataset", "label_key"], how="left")
    results = results.merge(load_location_counts(C), on=["dataset", "label_key"], how="left")
    results = results.merge(load_detection_rates(C), on=["dataset", "label_key"], how="left")
    results["genuine"] = (results.quality == "specific") & results.tax_level.isin(
        ["species", "subspecies", "variety"]
    )
    results["reliable"] = (results.support >= C.RELIABLE_HOLDOUT_PER_CLASS) & ~results[
        "low_confidence"
    ].fillna(True).astype(bool)
    results["det_ok"] = results.det_rate.fillna(1.0) >= 0.85
    return results


def validate_export(
    runs: pd.DataFrame, class_results: pd.DataFrame, datasets: pd.DataFrame, C
) -> None:
    """Fail before publishing an incomplete or internally inconsistent snapshot."""
    run_key = ["dataset", "condition", "tier"]
    if runs.empty or class_results.empty or datasets.empty:
        raise ValueError("The documentation export cannot contain an empty result table.")
    if runs.duplicated(run_key).any():
        raise ValueError("Run results contain duplicate dataset/condition/tier rows.")
    for metric in ["overall_accuracy", "macro_accuracy_reliable_classes"]:
        if runs[metric].isna().any() or not runs[metric].between(0, 1).all():
            raise ValueError(f"Run metric {metric} contains missing or out-of-range values.")

    required_conditions = {
        "finetune-lila",
        "finetune-speciesnet",
        "scratch-lila-arch",
        "balanced-aug-lila",
    }
    run_keys = set(runs[run_key].itertuples(index=False, name=None))
    missing = [
        (dataset, condition, tier)
        for dataset in C.DATASETS
        for condition in required_conditions
        for tier in C.TRAIN_TIERS
        if (dataset, condition, tier) not in run_keys
    ]
    if missing:
        raise ValueError(f"Missing required run rows, beginning with: {missing[:3]}")

    paired = class_results[
        class_results.condition.isin(CLASS_CONDITIONS) & (class_results.quality == "specific")
    ]
    if paired.accuracy.isna().any() or not paired.accuracy.between(0, 1).all():
        raise ValueError("Paired class results contain missing or out-of-range accuracy.")
    pair_counts = paired.groupby(["dataset", "label", "tier"]).condition.nunique()
    if not (pair_counts == len(CLASS_CONDITIONS)).all():
        raise ValueError("Some baseline/combined class-result pairs are incomplete.")


def main() -> None:
    args = parse_args()
    experiments_dir = args.experiments_dir.resolve()
    commit, source_dirty = source_revision(experiments_dir, args.allow_dirty_source)
    sys.path.insert(0, str(experiments_dir / "src"))

    from zexp import breakdowns as B  # noqa: PLC0415
    from zexp import config as C  # noqa: PLC0415
    from zexp import provenance as P  # noqa: PLC0415
    from zexp import results as R  # noqa: PLC0415

    runs = R.load_runs()
    runs["regime"] = runs["condition"].map(
        lambda value: (
            "scratch"
            if value.startswith("scratch")
            else (
                "linear probe"
                if value.startswith("linearprobe")
                else ("zero shot" if value.startswith("zeroshot") else "fine-tune")
            )
        )
    )
    runs = runs[RUN_COLUMNS].sort_values(["dataset", "condition", "tier"])

    class_results = load_class_results(C, P, R)
    covariates = B.all_class_covariates()[["dataset", "label", "apparent_size", "md_miss_rate"]]
    class_results = class_results[class_results["condition"].isin(CLASS_CONDITIONS)].merge(
        covariates, on=["dataset", "label"], how="left"
    )
    class_results = class_results[CLASS_COLUMNS].sort_values(
        ["dataset", "condition", "tier", "label"]
    )

    datasets = R.load_dataset_stats().rename(columns={"n_classes": "catalog_classes"})
    provenance = pd.DataFrame([P.dataset_provenance(slug) for slug in C.DATASETS]).rename(
        columns={"n_classes": "provenance_classes"}
    )
    datasets = datasets.merge(
        provenance,
        left_on="slug",
        right_on="slug",
        how="left",
    ).sort_values("slug")
    if "location_source" in datasets:
        datasets["location_source"] = datasets["location_source"].map(
            lambda value: (
                json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
            )
        )

    validate_export(runs, class_results, datasets, C)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs.to_csv(output_dir / "image-finetuning-runs.csv", index=False)
    class_results.to_csv(output_dir / "image-finetuning-class-results.csv", index=False)
    datasets.to_csv(output_dir / "image-finetuning-datasets.csv", index=False)

    payload = {
        "schema_version": 1,
        "source_commit": commit,
        "source_dirty": source_dirty,
        "methods": {
            "zamba_version": version("zamba"),
            "training_tiers": C.TRAIN_TIERS,
            "holdout_per_class_target": C.HOLDOUT_PER_CLASS,
            "reliable_holdout_minimum": C.RELIABLE_HOLDOUT_PER_CLASS,
            "minimum_locations_per_class": C.MIN_LOCATIONS_PER_CLASS,
            "location_split_fractions": C.LOCATION_SPLIT_FRACTIONS,
            "maximum_epochs": C.MAX_EPOCHS,
            "early_stopping_patience": C.EARLY_STOPPING_PATIENCE,
            "seed": C.SEED,
            "optimizer": "AdamW",
            "weight_decay": C.WEIGHT_DECAY,
            "warmup_epochs": C.WARMUP_EPOCHS,
            "scheduler": "linear warmup then cosine decay",
            "stochastic_weight_averaging": False,
            "batch_sizes": {
                "lila.science": C.BATCH_SIZE,
                "speciesnet": C.SPECIESNET_BATCH_SIZE,
            },
            "learning_rates": {
                condition: C.lr_for(model_name, C.condition_mode(condition))
                for condition, model_name, _ in C.CONDITIONS
                if model_name in {"lila.science", "speciesnet"}
            },
        },
        "condition_labels": CONDITION_LABELS,
        "dataset_labels": C.DATASETS,
        "runs": records(runs),
        "class_results": records(class_results),
        "datasets": records(datasets),
    }
    (output_dir / "image-finetuning-results.json").write_text(
        json.dumps(payload, separators=(",", ":")) + "\n"
    )

    print(
        f"Exported {len(runs)} runs, {len(class_results)} class results, "
        f"and {len(datasets)} datasets to {output_dir}."
    )


if __name__ == "__main__":
    main()
