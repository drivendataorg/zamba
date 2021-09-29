# Debugging

Before kicking off a full run of inference or model training, we recommend testing your code with a "dry run". If you are generating predictions, this will run one batch of inference to quickly detect any bugs. If you are trainig a model, this will run one training and validation batch for one epoch. If the dry run completes successfully, predict and train away!

=== "CLI"
    ```console
    $ zamba predict --data-dir example_vids/ --dry-run

    $ zamba train --data-dir example_vids/ --labels example_labels.csv --dry-run
    ```
=== "Python"
    In Python, add `dry_run=True` to [`PredictConfig`](configurations.md#prediction-arguments) or [`TrainConfig`](configurations.md#training-arguments):
    ```python
    predict_config = PredictConfig(
        data_directory="example_vids/", dry_run=True
    )
    ```

## GPU memory errors

The dry run will also catch any GPU memory errors. If you hit a GPU memory error, there are a couple fixes.

#### Reducing the batch size
  
=== "CLI"
    ```console
    zamba train --data-dir example_vids/ --labels example_labels.csv --batch-size 1
    ```
=== "Python"
    In Python, add `batch_size` to [`PredictConfig`](configurations.md#prediction-arguments) or [`TrainConfig`](configurations.md#training-arguments):
    ```python
    predict_config = PredictConfig(
        data_directory="example_vids/", batch_size=1
    )
    ```

#### Decreasing video size

Resize video frames to be smaller before they are passed to the model. The default for all three models is 224x224 pixels. `video_height` and `video_width` cannot be passed directly to the command line, so if you are using the CLI these must be specified in a [YAML file](yaml-config.md).

=== "YAML file"
    ```yaml
    video_loader_config:
        video_height: 100
        video_width: 100
        total_frames: 16 # total_frames is always required
    ```
=== "Python"
    ```python
    video_loader_config = VideoLoaderConfig(
        video_height=100, video_width=100, total_frames=16
    ) # total_frames is always required
    ```

#### Reducing `num_workers`

Reduce the number of workers (subprocesses) used for data loading. By default, `num_workers` will be set to either one less than the number of CPUs in the system, or one if there is only one CPU in the system. `num_workers` cannot be passed directly to the command line, so if you are using the CLI it must be specified in a [YAML file](yaml-config.md).

=== "YAML file"
    In a YAML file, add `num_workers` to `predict_config` or `train_config`:
    ```yaml
    train_config:
        data_directory: "example_vids/" # required
        labels: "example_labels.csv" # required
        num_workers: 1
    ```
=== "Python"
    In Python, add `num_workers` to [`PredictConfig`](configurations.md#prediction-arguments) or [`TrainConfig`](configurations.md#training-arguments):
    ```python
    predict_config = PredictConfig(
        data_directory="example_vids/", num_workers=1
    )
    ```