# User Tutorial: Training a Model on Labaled Videos

## Basic examples

### CLI

### python

### Training

To train a model based on the videos in `vids_to_classify` and the labels in `example_labels.csv`:

```python
from zamba.models.model_manager import train_model
from zamba.models.config import TrainConfig, VideoLoaderConfig

train_config = TrainConfig(labels='example_labels.csv', data_directory='vids_to_classify/')
video_loader_config = VideoLoaderConfig()

train_model(train_config=train_config, video_loader_config=video_loader_config)
```

<!-- TODO: where will the model be saved?><!-->

## CLI tutorial

## Python tutorial