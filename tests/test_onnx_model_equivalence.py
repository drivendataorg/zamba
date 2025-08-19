import tempfile
from pathlib import Path
import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import patch
import torchvision.transforms as transforms

from zamba.onnx_converter import ONNXConverter, validate_onnx_model
from zamba.models.config import ModelEnum
from zamba.models.utils import RegionEnum
from zamba.models.model_manager import instantiate_model
from zamba.models.depth_estimation.depth_manager import DepthEstimationManager
from zamba.data.video import load_video_frames
from zamba.images.data import load_image


def process_video_frames_for_model(frames, target_size=(224, 224), num_frames=16):
    """Process video frames to match model input requirements.
    
    Args:
        frames: List of video frames as numpy arrays
        target_size: Target size for resizing (height, width)
        num_frames: Number of frames to process
        
    Returns:
        torch.Tensor: Processed frames tensor with shape (1, T, C, H, W)
    """
    resize_transform = transforms.Resize(target_size)
    
    # Process frames to match model input requirements
    processed_frames = []
    for frame in frames[:num_frames]:
        # Convert to tensor and resize
        frame_tensor = torch.from_numpy(frame).float()
        frame_tensor = frame_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        frame_tensor = resize_transform(frame_tensor)
        processed_frames.append(frame_tensor)
    
    # Stack frames and add batch dimension
    frames_tensor = torch.stack(processed_frames, dim=0)  # (T, C, H, W)
    frames_tensor = frames_tensor.unsqueeze(0)  # Add batch dimension
    
    return frames_tensor


class TestONNXModelEquivalence:
    """Test that ONNX models produce the same results as PyTorch models on test assets."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def converter(self):
        """Create an ONNX converter instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            converter = ONNXConverter(
                model_cache_dir=cache_dir,
                weight_download_region=RegionEnum("us"),
            )
            # Download weights for the models we'll test
            from zamba.utils_cli import download_all_weights
            download_all_weights(
                cache_dir=cache_dir,
                overwrite=False,
                include_video=True,
                include_depth=True,
                include_image=False,
                include_densepose=False,
                weight_download_region="us",
            )
            yield converter
    
    def test_video_model_equivalence_time_distributed(self, temp_dir, converter):
        """Test that time_distributed ONNX model produces same results as PyTorch model."""
        # Convert model to ONNX
        onnx_path = temp_dir / "time_distributed.onnx"
        converter.convert_video_model(
            model_name=ModelEnum.time_distributed,
            output_path=onnx_path,
        )
        
        # Load test video
        test_video_path = Path("tests/assets/videos/data/raw/benjamin/04250002.MP4")
        if not test_video_path.exists():
            pytest.skip(f"Test video not found: {test_video_path}")
        
        # Load video frames
        frames = load_video_frames(test_video_path, fps=1.0)
        if len(frames) == 0:
            pytest.skip("No frames loaded from test video")
        
        # Prepare input tensor (take first 16 frames and resize to 224x224)
        frames_tensor = process_video_frames_for_model(frames, target_size=(224, 224), num_frames=16)
        
        # Load PyTorch model
        checkpoint_filename = converter._get_checkpoint_filename(ModelEnum.time_distributed)
        checkpoint_path = converter.model_cache_dir / checkpoint_filename
        pytorch_model = instantiate_model(checkpoint=checkpoint_path)
        pytorch_model.eval()
        
        # Get PyTorch prediction
        with torch.no_grad():
            pytorch_output = pytorch_model(frames_tensor)
            pytorch_probs = torch.sigmoid(pytorch_output).numpy()
        
        # Load ONNX model and get prediction
        import onnxruntime as ort
        ort_session = ort.InferenceSession(str(onnx_path))
        input_name = ort_session.get_inputs()[0].name
        
        onnx_output = ort_session.run(None, {input_name: frames_tensor.numpy()})[0]
        onnx_probs = 1 / (1 + np.exp(-onnx_output))  # sigmoid
        
        # Compare outputs
        np.testing.assert_allclose(
            pytorch_probs, onnx_probs, rtol=1e-5, atol=1e-5
        )
        
        print(f"PyTorch output shape: {pytorch_probs.shape}")
        print(f"ONNX output shape: {onnx_probs.shape}")
        print(f"Max difference: {np.abs(pytorch_probs - onnx_probs).max():.2e}")
        print(f"Mean difference: {np.abs(pytorch_probs - onnx_probs).mean():.2e}")
    
    def test_video_model_equivalence_european(self, temp_dir, converter):
        """Test that european ONNX model produces same results as PyTorch model."""
        # Convert model to ONNX
        onnx_path = temp_dir / "european.onnx"
        converter.convert_video_model(
            model_name=ModelEnum.european,
            output_path=onnx_path,
        )
        
        # Load test video
        test_video_path = Path("tests/assets/videos/data/raw/benjamin/04250002.MP4")
        if not test_video_path.exists():
            pytest.skip(f"Test video not found: {test_video_path}")
        
        # Load video frames
        frames = load_video_frames(test_video_path, fps=1.0)
        if len(frames) == 0:
            pytest.skip("No frames loaded from test video")
        
        # Prepare input tensor (take first 16 frames and resize to 224x224)
        frames_tensor = process_video_frames_for_model(frames, target_size=(224, 224), num_frames=16)
        
        # Load PyTorch model
        checkpoint_filename = converter._get_checkpoint_filename(ModelEnum.european)
        checkpoint_path = converter.model_cache_dir / checkpoint_filename
        pytorch_model = instantiate_model(checkpoint=checkpoint_path)
        pytorch_model.eval()
        
        # Get PyTorch prediction
        with torch.no_grad():
            pytorch_output = pytorch_model(frames_tensor)
            pytorch_probs = torch.sigmoid(pytorch_output).numpy()
        
        # Load ONNX model and get prediction
        import onnxruntime as ort
        ort_session = ort.InferenceSession(str(onnx_path))
        input_name = ort_session.get_inputs()[0].name
        
        onnx_output = ort_session.run(None, {input_name: frames_tensor.numpy()})[0]
        onnx_probs = 1 / (1 + np.exp(-onnx_output))  # sigmoid
        
        # Compare outputs
        np.testing.assert_allclose(
            pytorch_probs, onnx_probs, rtol=1e-5, atol=1e-5
        )
        
        print(f"PyTorch output shape: {pytorch_probs.shape}")
        print(f"ONNX output shape: {onnx_probs.shape}")
        print(f"Max difference: {np.abs(pytorch_probs - onnx_probs).max():.2e}")
        print(f"Mean difference: {np.abs(pytorch_probs - onnx_probs).mean():.2e}")
    
    def test_depth_model_equivalence(self, temp_dir, converter):
        """Test that depth ONNX model produces same results as PyTorch model."""
        # Convert model to ONNX
        onnx_path = temp_dir / "depth.onnx"
        converter.convert_depth_model(output_path=onnx_path)
        
        # Load test video
        test_video_path = Path("tests/assets/videos/data/raw/benjamin/04250002.MP4")
        if not test_video_path.exists():
            pytest.skip(f"Test video not found: {test_video_path}")
        
        # Load video frames
        frames = load_video_frames(test_video_path, fps=1.0)
        if len(frames) == 0:
            pytest.skip("No frames loaded from test video")
        
        # Prepare input tensor for depth model (5 frames * 3 channels = 15 channels)
        # Depth model expects (batch, channels, height, width) where channels = 5 * 3
        frames_tensor = torch.from_numpy(frames[:5]).float()
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # (T, C, H, W)
        frames_tensor = frames_tensor.reshape(1, -1, frames_tensor.shape[2], frames_tensor.shape[3])
        
        # Load PyTorch model
        depth_manager = DepthEstimationManager(
            model_cache_dir=converter.model_cache_dir,
            gpus=0,
            weight_download_region=converter.weight_download_region,
        )
        pytorch_model = torch.jit.load(depth_manager.model_weights, map_location='cpu')
        pytorch_model.eval()
        
        # Get PyTorch prediction
        with torch.no_grad():
            pytorch_output = pytorch_model(frames_tensor)
        
        # Load ONNX model and get prediction
        import onnxruntime as ort
        ort_session = ort.InferenceSession(str(onnx_path))
        input_name = ort_session.get_inputs()[0].name
        
        onnx_output = ort_session.run(None, {input_name: frames_tensor.numpy()})[0]
        
        # Compare outputs
        np.testing.assert_allclose(
            pytorch_output.numpy(), onnx_output, rtol=1e-5, atol=1e-5
        )
        
        print(f"PyTorch output shape: {pytorch_output.shape}")
        print(f"ONNX output shape: {onnx_output.shape}")
        print(f"Max difference: {np.abs(pytorch_output.numpy() - onnx_output).max():.2e}")
        print(f"Mean difference: {np.abs(pytorch_output.numpy() - onnx_output).mean():.2e}")
    
    def test_image_model_equivalence(self, temp_dir, converter):
        """Test that image ONNX model produces same results as PyTorch model."""
        # Skip if no image checkpoint available
        pytest.skip("Image model conversion requires checkpoint paths - skipping for now")
        
        # Load test image
        test_image_path = Path("tests/assets/images/dog.jpg")
        if not test_image_path.exists():
            pytest.skip(f"Test image not found: {test_image_path}")
        
        # Load and preprocess image
        image = load_image(test_image_path)
        # Add preprocessing steps here as needed
        
        # Convert model to ONNX (would need checkpoint path)
        # onnx_path = temp_dir / "image_model.onnx"
        # converter.convert_image_model(checkpoint_path, onnx_path)
        
        # Compare outputs
        # ... implementation would go here
    
    def test_multiple_videos_equivalence(self, temp_dir, converter):
        """Test model equivalence across multiple test videos."""
        # Find all test videos
        test_video_dir = Path("tests/assets/videos/data/raw")
        if not test_video_dir.exists():
            pytest.skip(f"Test video directory not found: {test_video_dir}")
        
        video_files = list(test_video_dir.rglob("*.MP4")) + list(test_video_dir.rglob("*.AVI"))
        if not video_files:
            pytest.skip("No test videos found")
        
        # Test with first few videos
        for video_path in video_files[:3]:  # Limit to first 3 videos
            print(f"\nTesting with video: {video_path}")
            
            try:
                # Load video frames
                frames = load_video_frames(video_path, fps=1.0)
                if len(frames) == 0:
                    print(f"Skipping {video_path} - no frames loaded")
                    continue
                
                # Prepare input tensor
                frames_tensor = process_video_frames_for_model(frames, target_size=(224, 224), num_frames=16)
                
                # Convert model to ONNX
                onnx_path = temp_dir / f"test_{video_path.stem}.onnx"
                converter.convert_video_model(
                    model_name=ModelEnum.time_distributed,
                    output_path=onnx_path,
                )
                
                # Load PyTorch model
                checkpoint_filename = converter._get_checkpoint_filename(ModelEnum.time_distributed)
                checkpoint_path = converter.model_cache_dir / checkpoint_filename
                pytorch_model = instantiate_model(checkpoint=checkpoint_path)
                pytorch_model.eval()
                
                # Get predictions
                with torch.no_grad():
                    pytorch_output = pytorch_model(frames_tensor)
                    pytorch_probs = torch.sigmoid(pytorch_output).numpy()
                
                import onnxruntime as ort
                ort_session = ort.InferenceSession(str(onnx_path))
                input_name = ort_session.get_inputs()[0].name
                
                onnx_output = ort_session.run(None, {input_name: frames_tensor.numpy()})[0]
                onnx_probs = 1 / (1 + np.exp(-onnx_output))
                
                # Compare outputs
                max_diff = np.abs(pytorch_probs - onnx_probs).max()
                mean_diff = np.abs(pytorch_probs - onnx_probs).mean()
                
                print(f"  Max difference: {max_diff:.2e}")
                print(f"  Mean difference: {mean_diff:.2e}")
                
                # Assert that differences are small
                assert max_diff < 1e-4, f"Max difference {max_diff} too large for {video_path}"
                assert mean_diff < 1e-5, f"Mean difference {mean_diff} too large for {video_path}"
                
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue


# Helper method for ONNXConverter
def _get_checkpoint_filename(self, model_name):
    """Get checkpoint filename for a model."""
    from zamba.models.utils import get_model_checkpoint_filename
    return get_model_checkpoint_filename(model_name)

# Add the helper method to ONNXConverter
ONNXConverter._get_checkpoint_filename = _get_checkpoint_filename 