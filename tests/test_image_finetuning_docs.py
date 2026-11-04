import csv
import importlib.util
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "docs" / "docs" / "assets" / "data"
RESULTS_PAGE = ROOT / "docs" / "docs" / "image-finetuning-results.md"
GUIDE_PAGE = ROOT / "docs" / "docs" / "images-finetune-guide.md"
EXPORTER_PATH = ROOT / "docs" / "scripts" / "export_image_finetuning_results.py"


def load_snapshot():
    return json.loads((DATA_DIR / "image-finetuning-results.json").read_text())


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def quantile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def load_exporter():
    spec = importlib.util.spec_from_file_location("image_finetuning_exporter", EXPORTER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snapshot_schema_and_required_comparisons():
    data = load_snapshot()
    assert data["schema_version"] == 1
    assert re.fullmatch(r"[0-9a-f]{40}", data["source_commit"])
    assert data["source_dirty"] is False
    assert data["methods"]["training_tiers"] == [10, 100, 1000, 10000]
    assert data["methods"]["zamba_version"] == "2.7.2"
    assert data["methods"]["seed"] == 42

    datasets = set(data["dataset_labels"])
    tiers = set(data["methods"]["training_tiers"])
    assert len(datasets) == 8
    assert len(data["runs"]) == 264

    run_keys = [(row["dataset"], row["condition"], row["tier"]) for row in data["runs"]]
    assert len(run_keys) == len(set(run_keys))
    required_conditions = {
        "finetune-lila",
        "finetune-speciesnet",
        "scratch-lila-arch",
        "balanced-aug-lila",
    }
    assert {
        (dataset, condition, tier)
        for dataset in datasets
        for condition in required_conditions
        for tier in tiers
    }.issubset(run_keys)

    for row in data["runs"]:
        assert row["dataset"] in datasets
        assert row["tier"] in tiers
        assert 0 <= row["overall_accuracy"] <= 1
        assert 0 <= row["macro_accuracy_reliable_classes"] <= 1

    distinct_evaluations = {
        (
            row["dataset"],
            row["condition"],
            None if row["condition"] == "zeroshot-pe-core-g" else row["tier"],
        )
        for row in data["runs"]
    }
    assert len(distinct_evaluations) == 258

    paired_conditions = defaultdict(set)
    for row in data["class_results"]:
        if row["quality"] == "specific" and row["condition"] in {
            "finetune-lila",
            "balanced-aug-lila",
        }:
            assert 0 <= row["accuracy"] <= 1
            paired_conditions[(row["dataset"], row["label"], row["tier"])].add(row["condition"])
    assert paired_conditions
    assert all(
        conditions == {"finetune-lila", "balanced-aug-lila"}
        for conditions in paired_conditions.values()
    )


def test_published_takeaways_match_snapshot():
    data = load_snapshot()
    results_page = RESULTS_PAGE.read_text()
    guide_page = GUIDE_PAGE.read_text()

    def run_metric(condition, tier):
        return mean(
            row["macro_accuracy_reliable_classes"]
            for row in data["runs"]
            if row["condition"] == condition and row["tier"] == tier
        )

    finetune_10 = run_metric("finetune-lila", 10)
    finetune_100 = run_metric("finetune-lila", 100)
    scratch_10 = run_metric("scratch-lila-arch", 10)
    assert f"averaged {finetune_10:.2f} reliable-class macro accuracy" in results_page
    assert f"and {finetune_100:.2f} at 100" in results_page
    assert f"averaged about {finetune_10:.2f} reliable-class macro accuracy" in guide_page
    assert f"about {scratch_10:.2f} macro accuracy" in guide_page
    tier_10 = [
        row["macro_accuracy_reliable_classes"]
        for row in data["runs"]
        if row["condition"] == "finetune-lila" and row["tier"] == 10
    ]
    tier_100 = [
        row["macro_accuracy_reliable_classes"]
        for row in data["runs"]
        if row["condition"] == "finetune-lila" and row["tier"] == 100
    ]
    assert (
        f"ranged from about {min(tier_10):.2f} to {max(tier_10):.2f} at 10 images and "
        f"{min(tier_100):.2f} to {max(tier_100):.2f} at 100"
    ) in guide_page

    comparisons = []
    for dataset in data["dataset_labels"]:
        for tier in data["methods"]["training_tiers"]:
            values = {
                row["condition"]: row["macro_accuracy_reliable_classes"]
                for row in data["runs"]
                if row["dataset"] == dataset and row["tier"] == tier
            }
            comparisons.append(values["finetune-lila"] > values["finetune-speciesnet"])
    assert f"{sum(comparisons)} of {len(comparisons)}" in results_page
    assert f"{sum(comparisons)} of {len(comparisons)}" in guide_page

    rows = [
        row
        for row in data["class_results"]
        if row["tier"] == 10000 and row["quality"] == "specific"
    ]
    by_key = defaultdict(dict)
    for row in rows:
        by_key[(row["dataset"], row["label"])][row["condition"]] = row
    rare_changes = []
    for conditions in by_key.values():
        baseline = conditions["finetune-lila"]
        balanced = conditions["balanced-aug-lila"]
        if baseline["train_n"] < 200:
            rare_changes.append(balanced["accuracy"] - baseline["accuracy"])
    assert f"gained +{mean(rare_changes):.2f}" in results_page

    location_rows = [
        row
        for row in data["class_results"]
        if row["condition"] == "finetune-lila"
        and row["tier"] == 10000
        and row["genuine"]
        and row["reliable"]
        and row["det_ok"]
        and row["n_loc"] is not None
    ]
    low_cutoff = quantile([row["n_loc"] for row in location_rows], 1 / 3)
    high_cutoff = quantile([row["n_loc"] for row in location_rows], 2 / 3)
    tier_10_by_class = {
        (row["dataset"], row["label"]): row["accuracy"]
        for row in data["class_results"]
        if row["condition"] == "finetune-lila"
        and row["tier"] == 10
        and row["genuine"]
        and row["reliable"]
        and row["det_ok"]
    }
    few_accuracy = mean(
        tier_10_by_class[(row["dataset"], row["label"])]
        for row in location_rows
        if row["n_loc"] <= low_cutoff
    )
    many_accuracy = mean(
        tier_10_by_class[(row["dataset"], row["label"])]
        for row in location_rows
        if row["n_loc"] > high_cutoff
    )
    assert f"group averaged {many_accuracy:.2f} accuracy" in results_page
    assert f"lowest group averaged {few_accuracy:.2f}" in results_page

    assert data["source_commit"][:12] in results_page
    assert "10–100 images per class is enough" not in results_page
    assert "More locations beat more images" not in results_page


def test_download_assets_are_present_and_portable():
    expected = [
        DATA_DIR / "image-finetuning-runs.csv",
        DATA_DIR / "image-finetuning-class-results.csv",
        DATA_DIR / "image-finetuning-datasets.csv",
    ]
    assert all(path.is_file() for path in expected)

    with expected[-1].open(newline="") as file:
        reader = csv.DictReader(file)
        assert "catalog_classes" in reader.fieldnames
        assert "provenance_classes" in reader.fieldnames
        assert "n_classes_x" not in reader.fieldnames
        assert "n_classes_y" not in reader.fieldnames
        rows = list(reader)
    assert rows
    assert all(isinstance(json.loads(row["location_source"]), dict) for row in rows)


def test_exporter_rejects_dirty_source_by_default(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "docs-test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Docs test"], cwd=tmp_path, check=True)
    source_file = tmp_path / "source.txt"
    source_file.write_text("clean\n")
    subprocess.run(["git", "add", "source.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    exporter = load_exporter()
    commit, dirty = exporter.source_revision(tmp_path)
    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    assert dirty is False

    source_file.write_text("dirty\n")
    with pytest.raises(RuntimeError, match="worktree is dirty"):
        exporter.source_revision(tmp_path)
    assert exporter.source_revision(tmp_path, allow_dirty=True) == (commit, True)
