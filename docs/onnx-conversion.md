# ONNX Model Conversion

Zamba provides functionality to convert PyTorch models to ONNX format for optimized inference. This allows you to use ONNX Runtime for faster inference and better deployment options.

## Overview

The ONNX conversion feature supports:
- **Video classification models**: time_distributed, slowfast, european, blank_nonblank
- **Image classification models**: Any custom trained image model
- **Depth estimation models**: The depth estimation model

Note: DensePose models are not supported for ONNX conversion.

## CLI Usage

### Convert All Models

Convert all available models to ONNX format:

```bash
zamba convert-to-onnx --output-dir ./onnx_models --yes
```

### Convert Specific Model Types

Convert only video models:
```bash
zamba convert-to-onnx --output-dir ./onnx_models --include-video --no-include-image --no-include-depth --yes
```

Convert only depth model:
```bash
zamba convert-to-onnx --output-dir ./onnx_models --no-include-video --no-include-image --include-depth --yes
```

### Validate Converted Models

Validate that converted ONNX models produce the same outputs as original PyTorch models:

```bash
zamba convert-to-onnx --output-dir ./onnx_models --validate --yes
```

### Test Performance

Test ONNX model inference performance:

```bash
zamba convert-to-onnx --output-dir ./onnx_models --test-performance --yes
```

### Full Example

Convert all models with validation and performance testing:

```bash
zamba convert-to-onnx \
    --output-dir ./onnx_models \
    --include-video \
    --include-image \
    --include-depth \
    --validate \
    --test-performance \
    --opset-version 11 \
    --yes
```

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output-dir` | Directory to save converted ONNX models | `./onnx_models` |
| `--include-video` | Whether to convert video models | `True` |
| `--include-image` | Whether to convert image models | `True` |
| `--include-depth` | Whether to convert depth model | `True` |
| `--opset-version` | ONNX opset version to use | `11` |
| `--model-cache-dir` | Directory for model weights cache | Default cache |
| `--weight-download-region` | Server region for downloading weights | `us` |
| `--validate` | Validate converted models against original PyTorch models | `False` |
| `--test-performance` | Test ONNX model inference performance | `False` |
| `--yes` | Skip confirmation and proceed with conversion | `False` |

## Python API Usage

### Basic Conversion

```python
from zamba.onnx_converter import ONNXConverter
from pathlib import Path

# Initialize converter
converter = ONNXConverter(
    model_cache_dir=Path("./cache"),
    weight_download_region="us"
)

# Convert all models
converted_models = converter.convert_all_models(
    output_dir=Path("./onnx_models"),
    include_video=True,
    include_image=True,
    include_depth=True,
    opset_version=11
)

print(f"Converted {len(converted_models)} models")
```

### Convert Specific Model Types

```python
# Convert only video models
video_models = converter.convert_all_models(
    output_dir=Path("./onnx_models"),
    include_video=True,
    include_image=False,
    include_depth=False
)

# Convert specific video model
converter.convert_video_model(
    model_name="time_distributed",
    output_path=Path("./time_distributed.onnx"),
    input_shape=(1, 16, 3, 224, 224)
)

# Convert depth model
converter.convert_depth_model(
    output_path=Path("./depth.onnx"),
    input_shape=(1, 15, 270, 480)
)
```

### Validate Models

```python
from zamba.onnx_converter import validate_onnx_model
from zamba.models.model_manager import instantiate_model

# Load original PyTorch model
pytorch_model = instantiate_model(checkpoint=Path("./model.ckpt"))

# Validate ONNX model
is_valid = validate_onnx_model(
    onnx_path=Path("./model.onnx"),
    pytorch_model=pytorch_model,
    input_shape=(1, 16, 3, 224, 224),
    rtol=1e-5,
    atol=1e-5
)

print(f"Model validation: {'PASSED' if is_valid else 'FAILED'}")
```

### Test Performance

```python
from zamba.onnx_converter import test_onnx_inference

# Test ONNX model performance
stats = test_onnx_inference(
    onnx_path=Path("./model.onnx"),
    input_shape=(1, 16, 3, 224, 224),
    num_runs=10
)

print(f"Mean inference time: {stats['mean_time']:.4f}s")
print(f"Standard deviation: {stats['std_time']:.4f}s")
print(f"Min time: {stats['min_time']:.4f}s")
print(f"Max time: {stats['max_time']:.4f}s")
```

## Model Input Shapes

### Video Models
- **time_distributed**: `(batch_size, 16, 3, 224, 224)`
- **slowfast**: `(batch_size, 32, 3, 224, 224)`
- **european**: `(batch_size, 16, 3, 224, 224)`
- **blank_nonblank**: `(batch_size, 16, 3, 224, 224)`

### Image Models
- **Default**: `(batch_size, 3, 224, 224)`
- **Custom**: Depends on training configuration

### Depth Model
- **depth**: `(batch_size, 15, 270, 480)` (5 frames × 3 channels)

## Using ONNX Models

### With ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("model.onnx")

# Prepare input
input_name = session.get_inputs()[0].name
input_data = np.random.randn(1, 16, 3, 224, 224).astype(np.float32)

# Run inference
output = session.run(None, {input_name: input_data})[0]
print(f"Output shape: {output.shape}")
```

### With TensorRT (if available)

```python
import tensorrt as trt
import numpy as np

# Load ONNX model and convert to TensorRT
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open("model.onnx", "rb") as model:
    parser.parse(model.read())

# Build engine
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GB
engine = builder.build_engine(network, config)

# Save engine
with open("model.trt", "wb") as f:
    f.write(engine.serialize())
```

## Performance Comparison

ONNX models typically provide:
- **Faster inference**: 1.2x to 2x speedup compared to PyTorch
- **Smaller memory footprint**: Optimized for inference
- **Better deployment options**: Support for various inference engines
- **Cross-platform compatibility**: Run on different hardware and platforms

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Use smaller batch sizes or convert on CPU
2. **ONNX export errors**: Check model compatibility and opset version
3. **Validation failures**: Ensure input shapes match exactly
4. **Performance issues**: Try different ONNX Runtime providers

### Debug Mode

Enable debug logging to see detailed conversion information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Model Compatibility

Not all PyTorch operations are supported in ONNX. Common issues include:
- Dynamic control flow (if/else, loops)
- Custom operations
- Complex tensor operations

For unsupported operations, consider:
- Simplifying the model architecture
- Using a different opset version
- Implementing custom ONNX operators

## Examples

### Complete Workflow

```python
from pathlib import Path
from zamba.onnx_converter import ONNXConverter, validate_onnx_model, test_onnx_inference

# 1. Convert models
converter = ONNXConverter()
converted_models = converter.convert_all_models(
    output_dir=Path("./onnx_models"),
    include_video=True,
    include_depth=True
)

# 2. Validate models
for model_name, onnx_path in converted_models.items():
    print(f"Validating {model_name}...")
    # Load corresponding PyTorch model and validate
    # (implementation depends on model type)

# 3. Test performance
for model_name, onnx_path in converted_models.items():
    print(f"Testing performance of {model_name}...")
    if "video_" in model_name:
        input_shape = (1, 16, 3, 224, 224)
    elif model_name == "depth":
        input_shape = (1, 15, 270, 480)
    else:
        input_shape = (1, 3, 224, 224)
    
    stats = test_onnx_inference(onnx_path, input_shape)
    print(f"  Mean time: {stats['mean_time']:.4f}s")

print("ONNX conversion workflow completed!")
```

This documentation provides a comprehensive guide to using the ONNX conversion functionality in Zamba. The feature enables optimized inference and better deployment options for all supported model types. 