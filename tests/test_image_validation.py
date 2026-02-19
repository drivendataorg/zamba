import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from PIL import Image

from zamba.images.config import ImageClassificationPredictConfig, ImageClassificationTrainingConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def valid_image_path(temp_dir):
    """Create a valid image file."""
    image_path = temp_dir / "valid_image.jpg"
    img = Image.new('RGB', (100, 100), color='red')
    img.save(image_path)
    return image_path


@pytest.fixture
def invalid_image_path(temp_dir):
    """Create an invalid image file with corrupt data."""
    image_path = temp_dir / "invalid_image.jpg"
    with open(image_path, 'wb') as f:
        f.write(b"This is not a valid image file")
    return image_path


@pytest.fixture
def labels_csv_valid(temp_dir, valid_image_path):
    """Create a labels CSV with only valid images."""
    labels_path = temp_dir / "labels_valid.csv"
    df = pd.DataFrame({
        'filepath': [valid_image_path.name],
        'label': ['cat']
    })
    df.to_csv(labels_path, index=False)
    return labels_path


@pytest.fixture
def labels_csv_mixed(temp_dir, valid_image_path, invalid_image_path):
    """Create a labels CSV with both valid and invalid images."""
    labels_path = temp_dir / "labels_mixed.csv"
    df = pd.DataFrame({
        'filepath': [valid_image_path.name, invalid_image_path.name],
        'label': ['cat', 'dog']
    })
    df.to_csv(labels_path, index=False)
    return labels_path


@pytest.fixture
def filepaths_df_valid(valid_image_path):
    """Create a filepaths DataFrame with only valid images."""
    return pd.DataFrame({
        'filepath': [str(valid_image_path)]
    })


@pytest.fixture
def filepaths_df_mixed(valid_image_path, invalid_image_path):
    """Create a filepaths DataFrame with both valid and invalid images."""
    return pd.DataFrame({
        'filepath': [str(valid_image_path), str(invalid_image_path)]
    })


class TestImageValidation:
    """Test the image validation functionality."""
    
    def test_predict_config_validate_images_disabled(self, temp_dir, filepaths_df_mixed):
        """Test that validation is disabled by default."""
        config = ImageClassificationPredictConfig(
            data_dir=temp_dir,
            filepaths=filepaths_df_mixed,
            validate_images=False
        )
        
        # Should include both valid and invalid images
        assert len(config.filepaths) == 2
        assert config.validate_images is False
    
    def test_predict_config_validate_images_enabled(self, temp_dir, filepaths_df_mixed, valid_image_path):
        """Test that validation filters out invalid images when enabled."""
        config = ImageClassificationPredictConfig(
            data_dir=temp_dir,
            filepaths=filepaths_df_mixed,
            validate_images=True
        )
        
        # Should only include valid images
        assert len(config.filepaths) == 1
        assert str(valid_image_path) in config.filepaths['filepath'].values
        assert config.validate_images is True
    
    def test_predict_config_validate_images_all_valid(self, temp_dir, filepaths_df_valid):
        """Test that validation works correctly with all valid images."""
        config = ImageClassificationPredictConfig(
            data_dir=temp_dir,
            filepaths=filepaths_df_valid,
            validate_images=True
        )
        
        # Should include the valid image
        assert len(config.filepaths) == 1
    
    def test_train_config_validate_images_disabled(self, temp_dir, labels_csv_mixed):
        """Test that training validation is disabled by default."""
        config = ImageClassificationTrainingConfig(
            data_dir=temp_dir,
            labels=labels_csv_mixed,
            validate_images=False
        )
        
        # Should include both valid and invalid images
        assert len(config.labels) == 2
        assert config.validate_images is False
    
    def test_train_config_validate_images_enabled(self, temp_dir, labels_csv_mixed):
        """Test that training validation filters out invalid images when enabled."""
        config = ImageClassificationTrainingConfig(
            data_dir=temp_dir,
            labels=labels_csv_mixed,
            validate_images=True
        )
        
        # Should only include valid images
        assert len(config.labels) == 1
        assert config.validate_images is True
    
    def test_train_config_validate_images_all_valid(self, temp_dir, labels_csv_valid):
        """Test that training validation works correctly with all valid images."""
        config = ImageClassificationTrainingConfig(
            data_dir=temp_dir,
            labels=labels_csv_valid,
            validate_images=True
        )
        
        # Should include the valid image
        assert len(config.labels) == 1
    
    def test_validate_filepath_with_pil_valid(self, valid_image_path):
        """Test the PIL validation function with valid image."""
        result = ImageClassificationTrainingConfig._validate_filepath_with_pil(
            (0, valid_image_path)
        )
        
        assert result[0] == 0  # index
        assert result[1] == valid_image_path  # path
        assert result[2] is True  # valid
    
    def test_validate_filepath_with_pil_invalid(self, invalid_image_path):
        """Test the PIL validation function with invalid image."""
        result = ImageClassificationTrainingConfig._validate_filepath_with_pil(
            (0, invalid_image_path)
        )
        
        assert result[0] == 0  # index
        assert result[1] == invalid_image_path  # path
        assert result[2] is False  # invalid
    
    def test_validate_filepath_with_pil_nonexistent(self, temp_dir):
        """Test the PIL validation function with non-existent file."""
        nonexistent_path = temp_dir / "nonexistent.jpg"
        result = ImageClassificationTrainingConfig._validate_filepath_with_pil(
            (0, nonexistent_path)
        )
        
        assert result[0] == 0  # index
        assert result[1] == nonexistent_path  # path
        assert result[2] is False  # invalid
    
    def test_validate_filepath_with_pil_empty_file(self, temp_dir):
        """Test the PIL validation function with empty file."""
        empty_path = temp_dir / "empty.jpg"
        empty_path.touch()  # Create empty file
        
        result = ImageClassificationTrainingConfig._validate_filepath_with_pil(
            (0, empty_path)
        )
        
        assert result[0] == 0  # index
        assert result[1] == empty_path  # path
        assert result[2] is False  # invalid
    
    @patch('zamba.images.config.logger')
    def test_training_validation_logging(self, mock_logger, temp_dir, labels_csv_mixed):
        """Test that appropriate logging messages are generated."""
        config = ImageClassificationTrainingConfig(
            data_dir=temp_dir,
            labels=labels_csv_mixed,
            validate_images=True
        )
        
        # Check that the validation info message was logged
        mock_logger.info.assert_any_call(
            "Validating image files exist and can be opened with PIL"
        )
        
        # Check that the warning message was logged
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "cannot be opened with PIL" in warning_call
        assert "ignoring those files" in warning_call
    
    @patch('zamba.images.config.logger')
    def test_prediction_validation_logging(self, mock_logger, temp_dir, filepaths_df_mixed):
        """Test that appropriate logging messages are generated for prediction."""
        config = ImageClassificationPredictConfig(
            data_dir=temp_dir,
            filepaths=filepaths_df_mixed,
            validate_images=True
        )
        
        # Check that the validation info message was logged
        mock_logger.info.assert_any_call(
            "Validating image files can be opened with PIL"
        )
        
        # Check that the warning message was logged
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "cannot be opened with PIL" in warning_call
        assert "ignoring those files" in warning_call