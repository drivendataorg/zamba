import tempfile
from pathlib import Path
import pytest
import torch
import numpy as np

from zamba.onnx_converter import ONNXConverter, validate_onnx_model, test_onnx_inference
from zamba.models.config import ModelEnum
from zamba.models.utils import RegionEnum


@pytest.mark.integration
class TestONNXIntegration:
    """Integration tests for ONNX conversion with real models."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def converter(self):
        """Create an ONNX converter instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            converter = ONNXConverter(
                model_cache_dir=Path(temp_dir),
                weight_download_region=RegionEnum("us"),
            )
            yield converter
    
    def test_simple_model_conversion(self, temp_dir):
        """Test conversion of a simple PyTorch model to ONNX."""
        # Create a simple model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)
                self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.fc = torch.nn.Linear(16, 10)
            
            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        model = SimpleModel()
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Export to ONNX
        onnx_path = temp_dir / "simple_model.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
        )
        
        # Validate the ONNX model
        assert onnx_path.exists()
        
        # Test validation
        result = validate_onnx_model(
            onnx_path=onnx_path,
            pytorch_model=model,
            input_shape=(1, 3, 224, 224),
        )
        
        assert result is True
        
        # Test performance
        stats = test_onnx_inference(
            onnx_path=onnx_path,
            input_shape=(1, 3, 224, 224),
            num_runs=5,
        )
        
        assert "mean_time" in stats
        assert stats["mean_time"] > 0
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available for GPU testing"
    )
    def test_gpu_model_conversion(self, temp_dir):
        """Test conversion of a model that can run on GPU."""
        # Create a simple model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)
                self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.fc = torch.nn.Linear(16, 10)
            
            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        model = SimpleModel()
        model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
        
        # Export to ONNX
        onnx_path = temp_dir / "gpu_model.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
        )
        
        # Validate the ONNX model
        assert onnx_path.exists()
        
        # Move model back to CPU for validation
        model = model.cpu()
        
        # Test validation
        result = validate_onnx_model(
            onnx_path=onnx_path,
            pytorch_model=model,
            input_shape=(1, 3, 224, 224),
        )
        
        assert result is True
    
    def test_dynamic_batch_size(self, temp_dir):
        """Test ONNX conversion with dynamic batch size."""
        # Create a simple model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)
                self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.fc = torch.nn.Linear(16, 10)
            
            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        model = SimpleModel()
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Export to ONNX with dynamic batch size
        onnx_path = temp_dir / "dynamic_batch_model.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
        )
        
        # Validate the ONNX model
        assert onnx_path.exists()
        
        # Test validation with different batch sizes
        for batch_size in [1, 2, 4]:
            result = validate_onnx_model(
                onnx_path=onnx_path,
                pytorch_model=model,
                input_shape=(batch_size, 3, 224, 224),
            )
            assert result is True
    
    def test_video_model_shape(self, temp_dir):
        """Test conversion with video model input shape."""
        # Create a simple video model (time distributed)
        class SimpleVideoModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv3d(3, 16, (3, 3, 3), padding=(1, 1, 1))
                self.pool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
                self.fc = torch.nn.Linear(16, 10)
            
            def forward(self, x):
                # x shape: (batch, frames, channels, height, width)
                # Convert to (batch, channels, frames, height, width)
                x = x.permute(0, 2, 1, 3, 4)
                x = torch.relu(self.conv(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        model = SimpleVideoModel()
        model.eval()
        
        # Create dummy input for video model
        dummy_input = torch.randn(1, 16, 3, 224, 224)  # (batch, frames, channels, height, width)
        
        # Export to ONNX
        onnx_path = temp_dir / "video_model.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
        )
        
        # Validate the ONNX model
        assert onnx_path.exists()
        
        # Test validation
        result = validate_onnx_model(
            onnx_path=onnx_path,
            pytorch_model=model,
            input_shape=(1, 16, 3, 224, 224),
        )
        
        assert result is True
        
        # Test performance
        stats = test_onnx_inference(
            onnx_path=onnx_path,
            input_shape=(1, 16, 3, 224, 224),
            num_runs=3,
        )
        
        assert "mean_time" in stats
        assert stats["mean_time"] > 0


@pytest.mark.slow
class TestONNXSlowIntegration:
    """Slow integration tests that may take longer to run."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_large_model_conversion(self, temp_dir):
        """Test conversion of a larger model."""
        # Create a larger model
        class LargeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.features = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 64, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(64, 64, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2, 2),
                    torch.nn.Conv2d(64, 128, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(128, 128, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2, 2),
                    torch.nn.Conv2d(128, 256, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(256, 256, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.classifier = torch.nn.Linear(256, 1000)
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        model = LargeModel()
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Export to ONNX
        onnx_path = temp_dir / "large_model.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
        )
        
        # Validate the ONNX model
        assert onnx_path.exists()
        
        # Test validation
        result = validate_onnx_model(
            onnx_path=onnx_path,
            pytorch_model=model,
            input_shape=(1, 3, 224, 224),
        )
        
        assert result is True
        
        # Test performance with multiple runs
        stats = test_onnx_inference(
            onnx_path=onnx_path,
            input_shape=(1, 3, 224, 224),
            num_runs=10,
        )
        
        assert "mean_time" in stats
        assert stats["mean_time"] > 0
        assert stats["std_time"] >= 0 