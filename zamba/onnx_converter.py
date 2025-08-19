import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import torch
from loguru import logger

from zamba.models.model_manager import instantiate_model
from zamba.models.depth_estimation.depth_manager import DepthEstimationManager
from zamba.images.manager import instantiate_model as instantiate_image_model
from zamba.models.config import ModelEnum
from zamba.models.utils import get_model_checkpoint_filename, RegionEnum
from zamba.utils_cli import download_all_weights


class ONNXConverter:
    """Converts PyTorch models to ONNX format for optimized inference."""
    
    def __init__(
        self,
        model_cache_dir: Optional[Path] = None,
        weight_download_region: RegionEnum = RegionEnum("us"),
    ):
        """Initialize the ONNX converter.
        
        Args:
            model_cache_dir: Directory for model weights cache
            weight_download_region: Region for downloading weights
        """
        self.model_cache_dir = model_cache_dir or Path.home() / ".cache" / "zamba"
        self.weight_download_region = weight_download_region
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _ensure_weights_downloaded(self, include_video=True, include_image=True, include_depth=True):
        """Download all required model weights before conversion."""
        download_all_weights(
            cache_dir=self.model_cache_dir,
            overwrite=False,
            include_video=include_video,
            include_image=include_image,
            include_depth=include_depth,
            include_densepose=False,  # Skip densepose for ONNX conversion
            weight_download_region=self.weight_download_region.value,
        )

    def convert_video_model(
        self,
        model_name: ModelEnum,
        output_path: Path,
        input_shape: Tuple[int, int, int, int] = (1, 16, 3, 224, 224),
        opset_version: int = 11,
        dynamic_axes: Optional[Dict] = None,
    ) -> Path:
        """Convert a video classification model to ONNX format.
        
        Args:
            model_name: Name of the model to convert
            output_path: Path to save the ONNX model
            input_shape: Input tensor shape (batch, frames, channels, height, width)
            opset_version: ONNX opset version to use
            dynamic_axes: Dynamic axes configuration for variable batch sizes
            
        Returns:
            Path to the converted ONNX model
        """
        logger.info(f"Converting video model {model_name} to ONNX format")
        
        # Get checkpoint filename and construct full path in cache directory
        checkpoint_filename = get_model_checkpoint_filename(model_name)
        checkpoint_path = self.model_cache_dir / checkpoint_filename
        
        # Load model
        model = instantiate_model(checkpoint=checkpoint_path)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Set up dynamic axes if not provided
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
        )
        
        logger.info(f"Video model converted and saved to {output_path}")
        return output_path
    
    def convert_image_model(
        self,
        checkpoint_path: Path,
        output_path: Path,
        input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
        opset_version: int = 11,
        dynamic_axes: Optional[Dict] = None,
    ) -> Path:
        """Convert an image classification model to ONNX format.
        
        Args:
            checkpoint_path: Path to the PyTorch checkpoint
            output_path: Path to save the ONNX model
            input_shape: Input tensor shape (batch, channels, height, width)
            opset_version: ONNX opset version to use
            dynamic_axes: Dynamic axes configuration for variable batch sizes
            
        Returns:
            Path to the converted ONNX model
        """
        logger.info(f"Converting image model from {checkpoint_path} to ONNX format")
        
        # Load model
        model = instantiate_image_model(checkpoint=checkpoint_path)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Set up dynamic axes if not provided
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
        )
        
        logger.info(f"Image model converted and saved to {output_path}")
        return output_path
    
    def convert_depth_model(
        self,
        output_path: Path,
        input_shape: Tuple[int, int, int, int] = (1, 15, 270, 480),  # 5 frames * 3 channels
        opset_version: int = 11,
        dynamic_axes: Optional[Dict] = None,
    ) -> Path:
        """Convert the depth estimation model to ONNX format.
        
        Args:
            output_path: Path to save the ONNX model
            input_shape: Input tensor shape (batch, channels, height, width)
            opset_version: ONNX opset version to use
            dynamic_axes: Dynamic axes configuration for variable batch sizes
            
        Returns:
            Path to the converted ONNX model
        """
        logger.info("Converting depth model to ONNX format")
        
        # Initialize depth manager to get model weights path
        depth_manager = DepthEstimationManager(
            model_cache_dir=self.model_cache_dir,
            gpus=0,  # Use CPU for conversion
            weight_download_region=self.weight_download_region,
        )
        
        # Load the TorchScript model
        model = torch.jit.load(depth_manager.model_weights, map_location='cpu')
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Set up dynamic axes if not provided
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
        )
        
        logger.info(f"Depth model converted and saved to {output_path}")
        return output_path
    
    def convert_all_models(
        self,
        output_dir: Path,
        include_video: bool = True,
        include_image: bool = True,
        include_depth: bool = True,
        opset_version: int = 11,
    ) -> Dict[str, Path]:
        """Convert all available models to ONNX format.
        
        Args:
            output_dir: Directory to save all converted models
            include_video: Whether to convert video models
            include_image: Whether to convert image models
            include_depth: Whether to convert depth model
            opset_version: ONNX opset version to use
            
        Returns:
            Dictionary mapping model names to their ONNX file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        # Ensure all required weights are downloaded before conversion
        self._ensure_weights_downloaded(include_video=include_video, include_image=include_image, include_depth=include_depth)
        converted_models = {}
        
        # Convert video models
        if include_video:
            video_models = [
                ModelEnum.time_distributed,
                ModelEnum.slowfast,
                ModelEnum.european,
                ModelEnum.blank_nonblank,
            ]
            
            for model_name in video_models:
                try:
                    output_path = output_dir / f"{model_name.value}.onnx"
                    self.convert_video_model(
                        model_name=model_name,
                        output_path=output_path,
                        opset_version=opset_version,
                    )
                    converted_models[f"video_{model_name.value}"] = output_path
                except Exception as e:
                    logger.error(f"Failed to convert video model {model_name}: {e}")
        
        # Convert image models (would need checkpoint paths)
        if include_image:
            logger.warning("Image model conversion requires checkpoint paths. Skipping for now.")
        
        # Convert depth model
        if include_depth:
            try:
                output_path = output_dir / "depth.onnx"
                self.convert_depth_model(
                    output_path=output_path,
                    opset_version=opset_version,
                )
                converted_models["depth"] = output_path
            except Exception as e:
                logger.error(f"Failed to convert depth model: {e}")
        
        logger.info(f"Converted {len(converted_models)} models to ONNX format")
        return converted_models


def validate_onnx_model(
    onnx_path: Path,
    pytorch_model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> bool:
    """Validate that an ONNX model produces the same outputs as the original PyTorch model.
    
    Args:
        onnx_path: Path to the ONNX model
        pytorch_model: Original PyTorch model
        input_shape: Shape of input tensor for testing
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        
    Returns:
        True if models produce matching outputs, False otherwise
    """
    logger.info(f"Validating ONNX model: {onnx_path}")
    
    # Load ONNX model
    ort_session = ort.InferenceSession(str(onnx_path))
    
    # Create test input
    test_input = torch.randn(input_shape)
    
    # Get PyTorch output
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).numpy()
    
    # Get ONNX output
    onnx_input = {ort_session.get_inputs()[0].name: test_input.numpy()}
    onnx_output = ort_session.run(None, onnx_input)[0]
    
    # Compare outputs
    try:
        np.testing.assert_allclose(
            pytorch_output, onnx_output, rtol=rtol, atol=atol
        )
        logger.info("ONNX model validation passed")
        return True
    except AssertionError as e:
        logger.error(f"ONNX model validation failed: {e}")
        return False


def test_onnx_inference(
    onnx_path: Path,
    input_shape: Tuple[int, ...],
    num_runs: int = 10,
) -> Dict[str, float]:
    """Test ONNX model inference performance.
    
    Args:
        onnx_path: Path to the ONNX model
        input_shape: Shape of input tensor for testing
        num_runs: Number of inference runs to average
        
    Returns:
        Dictionary with timing statistics
    """
    logger.info(f"Testing ONNX inference: {onnx_path}")
    
    # Load ONNX model
    ort_session = ort.InferenceSession(str(onnx_path))
    
    # Create test input
    test_input = torch.randn(input_shape).numpy()
    onnx_input = {ort_session.get_inputs()[0].name: test_input}
    
    # Warm up
    for _ in range(5):
        ort_session.run(None, onnx_input)
    
    # Time inference
    import time
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        ort_session.run(None, onnx_input)
        end_time = time.time()
        times.append(end_time - start_time)
    
    stats = {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
    }
    
    logger.info(f"ONNX inference stats: {stats}")
    return stats 