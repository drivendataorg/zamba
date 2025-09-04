"""Tests for filepath handling in the image code path"""

import json
from pathlib import Path
import shutil

import pandas as pd
import pytest
import torch
from typer.testing import CliRunner

from zamba.image_cli import app as image_app
from zamba.images.data import ImageClassificationDataModule
from conftest import ASSETS_DIR

runner = CliRunner()

TEST_IMAGES_DIR = ASSETS_DIR / "images"
ENA24_SMALL_DIR = TEST_IMAGES_DIR / "ena24_small"


def create_csv_with_relative_paths(files, output_path):
    """Create a CSV file with relative paths to the given files."""
    pd.DataFrame({"filepath": [f.name for f in files]}).to_csv(output_path, index=False)


def create_csv_with_absolute_paths(files, output_path):
    """Create a CSV file with absolute paths to the given files."""
    pd.DataFrame({"filepath": [str(f.absolute()) for f in files]}).to_csv(output_path, index=False)


def create_csv_with_mixed_paths(files, output_path):
    """Create a CSV file with a mix of relative and absolute paths."""
    filepaths = []
    for i, f in enumerate(files):
        if i % 2 == 0:
            filepaths.append(f.name)  # Relative path
        else:
            filepaths.append(str(f.absolute()))  # Absolute path

    pd.DataFrame({"filepath": filepaths}).to_csv(output_path, index=False)


def create_labels_csv(files, labels_df, output_path):
    """Create a labels CSV file with the given files and labels."""
    # Extract just the filename part to match with the labels dataframe
    filenames = [f.name for f in files]

    # Create a new dataframe with just the files we're using
    filtered_labels = labels_df[labels_df["filepath"].isin(filenames)].copy()

    # Update the filepath column with the appropriate path format
    filtered_labels["filepath"] = filenames

    filtered_labels.to_csv(output_path, index=False)


@pytest.fixture
def ena24_dataset_setup(tmp_path):
    """Set up a temporary directory with the ena24_small dataset."""

    # Create a data directory for test images
    data_dir = tmp_path / "images"
    data_dir.mkdir()

    # Copy all image files from ena24_small to our temp directory
    image_files = []
    for img_path in ENA24_SMALL_DIR.glob("*.jpg"):
        shutil.copy(img_path, data_dir)
        image_files.append(data_dir / img_path.name)

    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()

    # Create CSVs with different path formats for prediction
    relative_csv = csv_dir / "relative_paths.csv"
    create_csv_with_relative_paths(image_files, relative_csv)

    absolute_csv = csv_dir / "absolute_paths.csv"
    create_csv_with_absolute_paths(image_files, absolute_csv)

    mixed_csv = csv_dir / "mixed_paths.csv"
    create_csv_with_mixed_paths(image_files, mixed_csv)

    original_labels_csv = ENA24_SMALL_DIR / "labels.csv"
    original_labels_df = pd.read_csv(original_labels_csv)
    relative_labels = csv_dir / "relative_labels.csv"
    create_labels_csv(image_files, original_labels_df, relative_labels)

    absolute_labels = csv_dir / "absolute_labels.csv"
    absolute_labels_df = original_labels_df.copy()
    absolute_labels_df["filepath"] = [
        str((data_dir / f).absolute()) for f in original_labels_df["filepath"]
    ]
    absolute_labels_df.to_csv(absolute_labels, index=False)

    mixed_labels = csv_dir / "mixed_labels.csv"
    mixed_labels_df = original_labels_df.copy()
    for i, filepath in enumerate(mixed_labels_df["filepath"]):
        if i % 2 == 0:
            mixed_labels_df.loc[i, "filepath"] = filepath  # Relative path
        else:
            mixed_labels_df.loc[i, "filepath"] = str(
                (data_dir / filepath).absolute()
            )  # Absolute path
    mixed_labels_df.to_csv(mixed_labels, index=False)

    # Create output directories
    save_dir = tmp_path / "output"
    save_dir.mkdir()

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    original_annotations_json = ENA24_SMALL_DIR / "coco_annotations.json"
    coco_file = csv_dir / "coco_annotations.json"
    shutil.copy(original_annotations_json, coco_file)

    # The megadetector annotations have been validated with
    # https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/validate_batch_results.py
    original_md_annotations_json = ENA24_SMALL_DIR / "md_annotations.json"
    md_file = csv_dir / "md_annotations.json"
    shutil.copy(original_md_annotations_json, md_file)

    return {
        "data_dir": data_dir,
        "csv_dir": csv_dir,
        "save_dir": save_dir,
        "checkpoint_dir": checkpoint_dir,
        "cache_dir": cache_dir,
        "files": image_files,
        "labels_df": original_labels_df,
        "relative_csv": relative_csv,
        "absolute_csv": absolute_csv,
        "mixed_csv": mixed_csv,
        "relative_labels": relative_labels,
        "absolute_labels": absolute_labels,
        "mixed_labels": mixed_labels,
        "coco_file": coco_file,
        "md_file": md_file,
    }


@pytest.mark.parametrize("input_format", ["relative_csv", "absolute_csv", "mixed_csv"])
@pytest.mark.network
def test_image_cli_with_paths(mocker, ena24_dataset_setup, input_format):
    """Test that the image CLI can handle a CSV with the specified path format."""
    predict_mock = mocker.patch("zamba.images.manager.ZambaImagesManager.predict")

    result = runner.invoke(
        image_app,
        [
            "predict",
            "--data-dir",
            str(ena24_dataset_setup["data_dir"]),
            "--filepaths",
            str(ena24_dataset_setup[input_format]),
            "--save-dir",
            str(ena24_dataset_setup["save_dir"]),
            "--yes",
        ],
    )

    assert result.exit_code == 0
    predict_mock.assert_called_once()

    # Check that the config passed to the predict function has the correct filepaths
    predict_config = predict_mock.call_args[0][0]
    assert len(predict_config.filepaths) == len(ena24_dataset_setup["files"])
    for file_path in predict_config.filepaths["filepath"]:
        assert Path(file_path).exists()


@pytest.mark.parametrize("input_format", ["relative_labels", "absolute_labels", "mixed_labels"])
def test_train_with_different_path_formats(mocker, ena24_dataset_setup, input_format):
    """Test that the image CLI can handle training with different label path formats."""
    train_mock = mocker.patch("zamba.images.manager.ZambaImagesManager.train")

    result = runner.invoke(
        image_app,
        [
            "train",
            "--data-dir",
            str(ena24_dataset_setup["data_dir"]),
            "--labels",
            str(ena24_dataset_setup[input_format]),
            "--save-dir",
            str(ena24_dataset_setup["save_dir"]),
            "--cache-dir",
            str(ena24_dataset_setup["cache_dir"]),
            "--checkpoint-path",
            str(ena24_dataset_setup["checkpoint_dir"]),
            "--from-scratch",  # Train from scratch to avoid needing a checkpoint
            "--max-epochs",
            "1",  # Just 1 epoch for testing
            "--yes",
        ],
    )

    assert result.exit_code == 0
    train_mock.assert_called_once()

    train_config = train_mock.call_args[0][0]

    processed_images = set(Path(p).name for p in train_config.labels["filepath"].tolist())
    expected_images = {f.name for f in ena24_dataset_setup["files"]}

    assert len(processed_images) > 0
    assert processed_images == expected_images


@pytest.mark.network
def test_image_cli_file_discovery(mocker, ena24_dataset_setup):
    """Test that the image CLI can discover files in a directory when no CSV is provided."""
    predict_mock = mocker.patch("zamba.images.manager.ZambaImagesManager.predict")

    result = runner.invoke(
        image_app,
        [
            "predict",
            "--data-dir",
            str(ena24_dataset_setup["data_dir"]),
            "--save-dir",
            str(ena24_dataset_setup["save_dir"]),
            "--yes",
        ],
    )

    assert result.exit_code == 0
    predict_mock.assert_called_once()

    # Check that all files in the directory were found
    predict_config = predict_mock.call_args[0][0]
    assert len(predict_config.filepaths) == len(ena24_dataset_setup["files"])


def test_train_with_coco_labels(mocker, ena24_dataset_setup):
    """Test training with COCO format JSON labels."""
    train_mock = mocker.patch("zamba.images.manager.ZambaImagesManager.train")

    result = runner.invoke(
        image_app,
        [
            "train",
            "--data-dir",
            str(ena24_dataset_setup["data_dir"]),
            "--labels",
            str(ena24_dataset_setup["coco_file"]),
            "--save-dir",
            str(ena24_dataset_setup["save_dir"]),
            "--cache-dir",
            str(ena24_dataset_setup["cache_dir"]),
            "--checkpoint-path",
            str(ena24_dataset_setup["checkpoint_dir"]),
            "--from-scratch",  # Train from scratch to avoid needing a checkpoint
            "--max-epochs",
            "1",  # Just 1 epoch for testing
            "--yes",
        ],
    )

    assert result.exit_code == 0
    train_mock.assert_called_once()

    train_config = train_mock.call_args[0][0]
    assert len(train_config.labels) == 21
    assert set(["x1", "y1", "x2", "y2"]).issubset(train_config.labels.columns)

    # Test that x2 is calculated correctly from x1 and width (1 and 111)
    assert train_config.labels.loc[0, "x2"] == 112
    # Test that y2 is calculated correctly from x1 and height (11 and 10)
    assert train_config.labels.loc[0, "y2"] == 21


def test_train_with_md_labels(mocker, ena24_dataset_setup):
    """Test training with COCO format JSON labels."""
    train_mock = mocker.patch("zamba.images.manager.ZambaImagesManager.train")

    result = runner.invoke(
        image_app,
        [
            "train",
            "--data-dir",
            str(ena24_dataset_setup["data_dir"]),
            "--labels",
            str(ena24_dataset_setup["md_file"]),
            "--labels-format",
            "megadetector",
            "--save-dir",
            str(ena24_dataset_setup["save_dir"]),
            "--cache-dir",
            str(ena24_dataset_setup["cache_dir"]),
            "--checkpoint-path",
            str(ena24_dataset_setup["checkpoint_dir"]),
            "--from-scratch",  # Train from scratch to avoid needing a checkpoint
            "--max-epochs",
            "1",  # Just 1 epoch for testing
            "--yes",
        ],
    )

    assert result.exit_code == 0
    train_mock.assert_called_once()

    config = train_mock.call_args[0][0]
    assert len(config.labels) == 21
    assert set(["x1", "y1", "x2", "y2"]).issubset(config.labels.columns)

    # MD bounds are relative
    assert config.labels.loc[0, "x1"] < 1.0

    data = ImageClassificationDataModule(
        data_dir=config.data_dir,
        cache_dir=config.cache_dir,
        annotations=config.labels,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        detection_threshold=config.detections_threshold,
        crop_images=config.crop_images,
    )

    assert data.annotations.loc[0, "x1"] == 1920 * 0.35  # 672
    assert data.annotations.loc[0, "y1"] == 1080 * 0.35  # 378

    # some buffer for floating point precision
    assert (
        abs(data.annotations.loc[0, "x2"] - ((1920 * 0.35) + (1920 * 0.3))) <= 1
    )  # 672 + 576 = 1248
    assert (
        abs(data.annotations.loc[0, "y2"] - ((1080 * 0.35) + (1080 * 0.3))) <= 1
    )  # 378 + 324 = 702

    # make sure bounding boxes are absolute
    assert (
        (
            (data.annotations.loc[:, ["x1", "y1", "x2", "y2"]] == 0)
            | (data.annotations.loc[:, ["x1", "y1", "x2", "y2"]] >= 1.0)
        )
        .all()
        .all()
    )


@pytest.mark.network
def test_image_cli_csv_output(mocker, ena24_dataset_setup):
    """Test that the image CLI can output predictions in CSV format."""

    mock_detector = mocker.MagicMock()
    mock_detector.generate_detections_one_image.return_value = {
        "detections": [
            {
                "category": "1",  # Animal
                "conf": 0.9,
                "bbox": [0.1, 0.2, 0.3, 0.4],
            }
        ]
    }
    mocker.patch("megadetector.detection.run_detector.load_detector", return_value=mock_detector)

    mock_classifier = mocker.MagicMock()
    mock_classifier.species = ["animal", "blank"]

    # Mock classification result
    mock_classification_result = torch.tensor([[0.8, 0.2]])
    mock_classifier.return_value = mock_classification_result

    mocker.patch("zamba.images.manager.instantiate_model", return_value=mock_classifier)

    # Run the CLI command
    result = runner.invoke(
        image_app,
        [
            "predict",
            "--data-dir",
            str(ena24_dataset_setup["data_dir"]),
            "--save-dir",
            str(ena24_dataset_setup["save_dir"]),
            "--results-file-format",
            "csv",
            "--yes",
        ],
    )

    assert result.exit_code == 0

    # Check that the CSV output file was created
    output_csv = ena24_dataset_setup["save_dir"] / "zamba_predictions.csv"
    assert output_csv.exists()

    # Check the contents of the CSV file
    predictions = pd.read_csv(output_csv)

    # Check that required columns exist
    assert "filepath" in predictions.columns
    assert "detection_category" in predictions.columns
    assert "detection_conf" in predictions.columns
    assert "x1" in predictions.columns and "y1" in predictions.columns
    assert "x2" in predictions.columns and "y2" in predictions.columns

    # Check that our species columns exist
    for species in ["animal", "blank"]:
        assert species in predictions.columns


@pytest.mark.network
def test_image_cli_megadetector_output(mocker, ena24_dataset_setup):
    """Test that the image CLI can output predictions in MegaDetector JSON format."""
    # Mock detector
    mock_detector = mocker.MagicMock()
    mock_detector.generate_detections_one_image.return_value = {
        "detections": [
            {
                "category": "1",  # Animal
                "conf": 0.9,
                "bbox": [0.1, 0.2, 0.3, 0.4],
            }
        ]
    }
    mocker.patch("megadetector.detection.run_detector.load_detector", return_value=mock_detector)

    mock_classifier = mocker.MagicMock()
    mock_classifier.species = ["animal", "blank"]

    mock_classification_result = torch.tensor([[0.8, 0.2]])
    mock_classifier.return_value = mock_classification_result

    mocker.patch("zamba.images.manager.instantiate_model", return_value=mock_classifier)

    result = runner.invoke(
        image_app,
        [
            "predict",
            "--data-dir",
            str(ena24_dataset_setup["data_dir"]),
            "--save-dir",
            str(ena24_dataset_setup["save_dir"]),
            "--results-file-format",
            "megadetector",
            "--yes",
        ],
    )

    assert result.exit_code == 0

    output_json = ena24_dataset_setup["save_dir"] / "zamba_predictions.json"
    assert output_json.exists()

    # Check the contents of the JSON file
    with open(output_json, "r") as f:
        predictions = json.load(f)

    assert "images" in predictions
    assert "detection_categories" in predictions
    assert "info" in predictions

    first_image = predictions["images"][0]
    assert "file" in first_image
    assert "detections" in first_image

    detection = first_image["detections"][0]
    assert "category" in detection
    assert "conf" in detection
    assert "bbox" in detection

    classification = detection["classifications"][0]
    assert len(classification) == 2
