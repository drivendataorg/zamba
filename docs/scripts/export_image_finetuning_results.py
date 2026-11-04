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
from pathlib import Path

import pandas as pd

CLASS_CONDITIONS = ["finetune-lila", "balanced-aug-lila"]

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
    "balanced-aug-lila": "Balanced sampling, lila.science",
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
    return parser.parse_args()


def records(df: pd.DataFrame) -> list[dict]:
    """Return JSON-safe records with stable precision."""
    out = df.copy()
    float_columns = out.select_dtypes(include="float").columns
    out[float_columns] = out[float_columns].round(6)
    return json.loads(out.to_json(orient="records"))


def source_commit(experiments_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(experiments_dir), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    args = parse_args()
    experiments_dir = args.experiments_dir.resolve()
    sys.path.insert(0, str(experiments_dir / "src"))

    from zexp import breakdowns as B  # noqa: PLC0415
    from zexp import config as C  # noqa: PLC0415
    from zexp import provenance as P  # noqa: PLC0415
    from zexp import report as Report  # noqa: PLC0415
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

    class_results = Report._familiarity_data()
    covariates = B.all_class_covariates()[["dataset", "label", "apparent_size", "md_miss_rate"]]
    class_results = class_results[class_results["condition"].isin(CLASS_CONDITIONS)].merge(
        covariates, on=["dataset", "label"], how="left"
    )
    class_results = class_results[CLASS_COLUMNS].sort_values(
        ["dataset", "condition", "tier", "label"]
    )

    datasets = R.load_dataset_stats()
    provenance = pd.DataFrame([P.dataset_provenance(slug) for slug in C.DATASETS])
    datasets = datasets.merge(
        provenance,
        left_on="slug",
        right_on="slug",
        how="left",
    ).sort_values("slug")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs.to_csv(output_dir / "image-finetuning-runs.csv", index=False)
    class_results.to_csv(output_dir / "image-finetuning-class-results.csv", index=False)
    datasets.to_csv(output_dir / "image-finetuning-datasets.csv", index=False)

    payload = {
        "schema_version": 1,
        "source_commit": source_commit(experiments_dir),
        "methods": {
            "training_tiers": C.TRAIN_TIERS,
            "holdout_per_class_target": C.HOLDOUT_PER_CLASS,
            "reliable_holdout_minimum": C.RELIABLE_HOLDOUT_PER_CLASS,
            "minimum_locations_per_class": C.MIN_LOCATIONS_PER_CLASS,
            "location_split_fractions": C.LOCATION_SPLIT_FRACTIONS,
            "maximum_epochs": C.MAX_EPOCHS,
            "early_stopping_patience": C.EARLY_STOPPING_PATIENCE,
            "seed": C.SEED,
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
