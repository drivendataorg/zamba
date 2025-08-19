import tempfile
from pathlib import Path
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from zamba.onnx_converter import (
    ONNXConverter,
    validate_onnx_model,
    test_onnx_inference,
)
from zamba.models.config import ModelEnum
from zamba.models.utils import RegionEnum


class TestONNXConverter:
    """Test the ONNX converter functionality."""
    
    @pytest.fixture
    def converter(self):
        """Create an ONNX converter instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            converter = ONNXConverter(
                model_cache_dir=Path(temp_dir),
                weight_download_region=RegionEnum("us"),
            )
            yield converter
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_converter_initialization(self, converter):
        """Test ONNX converter initialization."""
        assert converter.model_cache_dir.exists()
        assert converter.weight_download_region == RegionEnum("us")
    
    @patch('zamba.onnx_converter.get_model_checkpoint_filename')
    @patch('zamba.onnx_converter.instantiate_model')
    def test_convert_video_model(self, mock_instantiate_model, mock_get_checkpoint, converter, temp_dir):
        """Test video model conversion."""
        # Mock the model loading
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_instantiate_model.return_value = mock_model
        
        # Mock checkpoint path
        mock_checkpoint = Path("/fake/checkpoint.ckpt")
        mock_get_checkpoint.return_value = mock_checkpoint
        
        # Mock torch.onnx.export
        with patch('torch.onnx.export') as mock_export:
            output_path = temp_dir / "test_video_model.onnx"
            
            result = converter.convert_video_model(
                model_name=ModelEnum.time_distributed,
                output_path=output_path,
                input_shape=(1, 16, 3, 224, 224),
            )
            
            # Verify the model was loaded and converted
            mock_get_checkpoint.assert_called_once_with(ModelEnum.time_distributed)
            mock_instantiate_model.assert_called_once_with(checkpoint=mock_checkpoint)
            mock_model.eval.assert_called_once()
            mock_export.assert_called_once()
            
            assert result == output_path
    
    @patch('zamba.onnx_converter.instantiate_image_model')
    def test_convert_image_model(self, mock_instantiate_image_model, converter, temp_dir):
        """Test image model conversion."""
        # Mock the model loading
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_instantiate_image_model.return_value = mock_model
        
        # Mock torch.onnx.export
        with patch('torch.onnx.export') as mock_export:
            checkpoint_path = Path("/fake/image_checkpoint.ckpt")
            output_path = temp_dir / "test_image_model.onnx"
            
            result = converter.convert_image_model(
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                input_shape=(1, 3, 224, 224),
            )
            
            # Verify the model was loaded and converted
            mock_instantiate_image_model.assert_called_once_with(checkpoint=checkpoint_path)
            mock_model.eval.assert_called_once()
            mock_export.assert_called_once()
            
            assert result == output_path
    
    @patch('zamba.onnx_converter.DepthEstimationManager')
    @patch('torch.jit.load')
    def test_convert_depth_model(self, mock_jit_load, mock_depth_manager_class, converter, temp_dir):
        """Test depth model conversion."""
        # Mock the depth manager
        mock_depth_manager = Mock()
        mock_depth_manager.model_weights = Path("/fake/depth_model.pt")
        mock_depth_manager_class.return_value = mock_depth_manager
        
        # Mock the TorchScript model
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_jit_load.return_value = mock_model
        
        # Mock torch.onnx.export
        with patch('torch.onnx.export') as mock_export:
            output_path = temp_dir / "test_depth_model.onnx"
            
            result = converter.convert_depth_model(
                output_path=output_path,
                input_shape=(1, 15, 270, 480),
            )
            
            # Verify the depth manager was created and model was loaded
            mock_depth_manager_class.assert_called_once()
            mock_jit_load.assert_called_once_with(mock_depth_manager.model_weights, map_location='cpu')
            mock_model.eval.assert_called_once()
            mock_export.assert_called_once()
            
            assert result == output_path
    
    @patch('zamba.onnx_converter.get_model_checkpoint_filename')
    @patch('zamba.onnx_converter.instantiate_model')
    @patch('zamba.onnx_converter.DepthEstimationManager')
    @patch('torch.jit.load')
    def test_convert_all_models(
        self,
        mock_jit_load,
        mock_depth_manager_class,
        mock_instantiate_model,
        mock_get_checkpoint,
        converter,
        temp_dir,
    ):
        """Test converting all models."""
        # Mock video model conversion
        mock_video_model = Mock()
        mock_video_model.eval.return_value = None
        mock_instantiate_model.return_value = mock_video_model
        mock_get_checkpoint.return_value = Path("/fake/checkpoint.ckpt")
        
        # Mock depth model conversion
        mock_depth_manager = Mock()
        mock_depth_manager.model_weights = Path("/fake/depth_model.pt")
        mock_depth_manager_class.return_value = mock_depth_manager
        
        mock_depth_model = Mock()
        mock_depth_model.eval.return_value = None
        mock_jit_load.return_value = mock_depth_model
        
        # Mock torch.onnx.export
        with patch('torch.onnx.export') as mock_export:
            result = converter.convert_all_models(
                output_dir=temp_dir,
                include_video=True,
                include_image=False,  # Skip image models for this test
                include_depth=True,
            )
            
            # Verify models were converted
            assert len(result) == 5  # 4 video models + 1 depth model
            assert "video_time_distributed" in result
            assert "video_slowfast" in result
            assert "video_european" in result
            assert "video_blank_nonblank" in result
            assert "depth" in result
            
            # Verify export was called for each model
            assert mock_export.call_count == 5
    
    def test_convert_all_models_with_exceptions(self, converter, temp_dir):
        """Test that exceptions during conversion are handled gracefully."""
        with patch('zamba.onnx_converter.get_model_checkpoint_filename', side_effect=Exception("Test error")):
            result = converter.convert_all_models(
                output_dir=temp_dir,
                include_video=True,
                include_image=False,
                include_depth=False,
            )
            
            # Should return empty dict when all conversions fail
            assert result == {}


class TestONNXValidation:
    """Test ONNX model validation functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_validate_onnx_model_success(self, temp_dir):
        """Test successful ONNX model validation."""
        # Create a simple PyTorch model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        pytorch_model = SimpleModel()
        pytorch_model.eval()
        
        # Create a dummy ONNX model
        dummy_input = torch.randn(1, 10)
        onnx_path = temp_dir / "test_model.onnx"
        
        # Export to ONNX
        torch.onnx.export(
            pytorch_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
        )
        
        # Test validation
        result = validate_onnx_model(
            onnx_path=onnx_path,
            pytorch_model=pytorch_model,
            input_shape=(1, 10),
        )
        
        assert result is True
    
    def test_validate_onnx_model_failure(self, temp_dir):
        """Test ONNX model validation failure."""
        # Create a simple PyTorch model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        pytorch_model = SimpleModel()
        pytorch_model.eval()
        
        # Create a different ONNX model (different weights)
        class DifferentModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
                # Initialize with different weights
                torch.nn.init.ones_(self.linear.weight)
                torch.nn.init.zeros_(self.linear.bias)
            
            def forward(self, x):
                return self.linear(x)
        
        different_model = DifferentModel()
        different_model.eval()
        
        dummy_input = torch.randn(1, 10)
        onnx_path = temp_dir / "test_model.onnx"
        
        # Export different model to ONNX
        torch.onnx.export(
            different_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
        )
        
        # Test validation should fail
        result = validate_onnx_model(
            onnx_path=onnx_path,
            pytorch_model=pytorch_model,
            input_shape=(1, 10),
        )
        
        assert result is False


class TestONNXPerformance:
    """Test ONNX model performance testing functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_test_onnx_inference(self, temp_dir):
        """Test ONNX inference performance testing."""
        # Create a simple PyTorch model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        pytorch_model = SimpleModel()
        pytorch_model.eval()
        
        # Create ONNX model
        dummy_input = torch.randn(1, 10)
        onnx_path = temp_dir / "test_model.onnx"
        
        torch.onnx.export(
            pytorch_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
        )
        
        # Test performance
        stats = test_onnx_inference(
            onnx_path=onnx_path,
            input_shape=(1, 10),
            num_runs=5,
        )
        
        # Verify stats structure
        assert "mean_time" in stats
        assert "std_time" in stats
        assert "min_time" in stats
        assert "max_time" in stats
        
        # Verify stats are reasonable
        assert stats["mean_time"] > 0
        assert stats["std_time"] >= 0
        assert stats["min_time"] > 0
        assert stats["max_time"] > 0
        assert stats["min_time"] <= stats["mean_time"] <= stats["max_time"]


class TestONNXCLI:
    """Test the ONNX conversion CLI command."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @patch('zamba.cli.ONNXConverter')
    def test_convert_to_onnx_cli(self, mock_converter_class, temp_dir, runner):
        """Test the convert_to_onnx CLI command."""
        # Mock the converter
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter
        
        # Mock conversion results
        mock_converted_models = {
            "video_time_distributed": temp_dir / "time_distributed.onnx",
            "video_slowfast": temp_dir / "slowfast.onnx",
            "depth": temp_dir / "depth.onnx",
        }
        mock_converter.convert_all_models.return_value = mock_converted_models
        
        # Mock validation and performance testing
        with patch('zamba.cli.validate_onnx_model', return_value=True) as mock_validate:
            with patch('zamba.cli.test_onnx_inference') as mock_performance:
                # Run CLI command
                result = runner.invoke(
                    "convert-to-onnx",
                    [
                        "--output-dir", str(temp_dir),
                        "--include-video",
                        "--include-depth",
                        "--no-include-image",
                        "--validate",
                        "--test-performance",
                        "--yes",
                    ]
                )
                
                # Verify command executed successfully
                assert result.exit_code == 0
                
                # Verify converter was called
                mock_converter_class.assert_called_once()
                mock_converter.convert_all_models.assert_called_once()
                
                # Verify validation was called for each model
                assert mock_validate.call_count == 3
                
                # Verify performance testing was called for each model
                assert mock_performance.call_count == 3
    
    @patch('zamba.cli.ONNXConverter')
    def test_convert_to_onnx_cli_no_validation(self, mock_converter_class, temp_dir, runner):
        """Test the convert_to_onnx CLI command without validation."""
        # Mock the converter
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter
        
        # Mock conversion results
        mock_converted_models = {
            "video_time_distributed": temp_dir / "time_distributed.onnx",
        }
        mock_converter.convert_all_models.return_value = mock_converted_models
        
        # Run CLI command without validation
        result = runner.invoke(
            "convert-to-onnx",
            [
                "--output-dir", str(temp_dir),
                "--include-video",
                "--no-include-image",
                "--no-include-depth",
                "--yes",
            ]
        )
        
        # Verify command executed successfully
        assert result.exit_code == 0
        
        # Verify converter was called
        mock_converter_class.assert_called_once()
        mock_converter.convert_all_models.assert_called_once()


@pytest.fixture
def runner():
    """Create a CLI runner for testing."""
    from typer.testing import CliRunner
    from zamba.cli import app
    return CliRunner(mix_stderr=False) 