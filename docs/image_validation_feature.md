# Image Validation Feature

## Overview

This feature adds a new CLI option `--validate-images` that enables PIL-based image validation to filter out corrupt or invalid image files during processing.

## Usage

### Command Line Interface

For image prediction:
```bash
zamba image predict --data-dir /path/to/images --validate-images
```

For image training:
```bash
zamba image train --data-dir /path/to/images --labels /path/to/labels.csv --validate-images
```

### Python API

For prediction:
```python
from zamba.images.config import ImageClassificationPredictConfig

config = ImageClassificationPredictConfig(
    data_dir="/path/to/images",
    validate_images=True
)
```

For training:
```python
from zamba.images.config import ImageClassificationTrainingConfig

config = ImageClassificationTrainingConfig(
    data_dir="/path/to/images",
    labels="/path/to/labels.csv",
    validate_images=True
)
```

## Behavior

### Default Behavior (validate_images=False)

- Only checks if files exist and have non-zero size
- Processes all files that pass basic existence checks
- Faster processing as no image decoding is performed

### With Validation Enabled (validate_images=True)

- Performs basic existence and size checks
- Attempts to open each image with PIL (Python Imaging Library)
- Filters out images that cannot be opened or decoded
- Logs warning messages about filtered files
- Continues processing with only valid images

## Implementation Details

### Training Configuration

The `ImageClassificationTrainingConfig` class includes:
- New `validate_images: bool = False` parameter
- Enhanced `validate_image_files()` method that uses PIL validation when enabled
- New `_validate_filepath_with_pil()` static method for PIL-based validation

### Prediction Configuration

The `ImageClassificationPredictConfig` class includes:
- New `validate_images: bool = False` parameter
- New `validate_image_files_predict()` method for PIL-based validation
- Validates filepaths DataFrame after it's loaded from CSV or directory

### Logging

When invalid images are found, the system logs:
- Info message: "Validating image files exist and can be opened with PIL"
- Warning message: "X image files cannot be opened with PIL; ignoring those files"

## Error Handling

The validation gracefully handles:
- Non-existent files
- Empty files
- Corrupt image files
- Files with wrong extensions
- Any PIL-related exceptions

Invalid files are simply excluded from processing without stopping the entire operation.

## Performance Considerations

- Enabling validation adds processing time as each image must be opened
- For large datasets, consider the trade-off between validation time and processing reliability
- Validation is performed in parallel using process_map for training
- For prediction, validation is performed sequentially but only on explicitly provided files

## Testing

The implementation includes comprehensive tests covering:
- Valid image handling
- Invalid image filtering
- Mixed valid/invalid scenarios
- Logging behavior
- CLI option parsing
- Configuration class behavior

## Migration

This is a backward-compatible change:
- Default behavior remains unchanged (validation disabled)
- Existing code will continue to work without modification
- Only users who explicitly enable the option will see the new behavior