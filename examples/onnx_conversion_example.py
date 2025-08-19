#!/usr/bin/env python3
"""
Example script demonstrating ONNX conversion functionality in Zamba.

This script shows how to:
1. Convert PyTorch models to ONNX format
2. Validate converted models
3. Test inference performance
4. Use ONNX models for inference

Run this script with:
    python examples/onnx_conversion_example.py
"""

import tempfile
from pathlib import Path
import torch
import numpy as np

from zamba.onnx_converter import ONNXConverter, validate_onnx_model, test_onnx_inference
from zamba.models.config import ModelEnum
from zamba.models.utils import RegionEnum


def create_simple_model():
    """Create a simple PyTorch model for demonstration."""
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
    
    return SimpleModel()


def main():
    """Main function demonstrating ONNX conversion."""
    print("=== Zamba ONNX Conversion Example ===\n")
    
    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        output_dir = temp_path / "onnx_models"
        
        print(f"1. Creating ONNX converter...")
        converter = ONNXConverter(
            model_cache_dir=temp_path / "cache",
            weight_download_region=RegionEnum("us"),
        )
        print(f"   Converter initialized with cache dir: {converter.model_cache_dir}")
        
        print(f"\n2. Creating a simple PyTorch model...")
        model = create_simple_model()
        model.eval()
        print(f"   Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        print(f"\n3. Converting model to ONNX...")
        onnx_path = output_dir / "simple_model.onnx"
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Export to ONNX with dynamic batch size
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
        print(f"   Model exported to: {onnx_path}")
        
        print(f"\n4. Validating ONNX model...")
        is_valid = validate_onnx_model(
            onnx_path=onnx_path,
            pytorch_model=model,
            input_shape=(1, 3, 224, 224),
        )
        print(f"   Validation: {'PASSED' if is_valid else 'FAILED'}")
        
        print(f"\n5. Testing ONNX inference performance...")
        stats = test_onnx_inference(
            onnx_path=onnx_path,
            input_shape=(1, 3, 224, 224),
            num_runs=10,
        )
        print(f"   Mean inference time: {stats['mean_time']:.4f}s")
        print(f"   Standard deviation: {stats['std_time']:.4f}s")
        print(f"   Min time: {stats['min_time']:.4f}s")
        print(f"   Max time: {stats['max_time']:.4f}s")
        
        print(f"\n6. Demonstrating ONNX inference...")
        import onnxruntime as ort
        
        # Load ONNX model
        session = ort.InferenceSession(str(onnx_path))
        
        # Prepare input
        input_name = session.get_inputs()[0].name
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Run inference
        output = session.run(None, {input_name: input_data})[0]
        print(f"   ONNX output shape: {output.shape}")
        print(f"   ONNX output sample: {output[0, :5]}")  # First 5 values
        
        # Compare with PyTorch
        with torch.no_grad():
            pytorch_output = model(torch.from_numpy(input_data)).numpy()
        print(f"   PyTorch output sample: {pytorch_output[0, :5]}")  # First 5 values
        
        # Check if outputs match
        diff = np.abs(output - pytorch_output).max()
        print(f"   Max difference: {diff:.2e}")
        print(f"   Outputs match: {diff < 1e-5}")
        
        print(f"\n7. Testing with different batch sizes...")
        for batch_size in [1, 2, 4]:
            input_data = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
            output = session.run(None, {input_name: input_data})[0]
            print(f"   Batch size {batch_size}: output shape {output.shape}")
        
        print(f"\n=== Example completed successfully! ===")
        print(f"ONNX model saved to: {onnx_path}")
        print(f"You can now use this ONNX model with ONNX Runtime for optimized inference.")


if __name__ == "__main__":
    main() 