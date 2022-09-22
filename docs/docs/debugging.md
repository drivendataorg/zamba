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
        data_dir="example_vids/", dry_run=True
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
        data_dir="example_vids/", batch_size=1
    )
    ```

#### Decreasing video size

Resize video frames to be smaller before they are passed to the model. The default for all models is 240x426 pixels. `model_input_height` and `model_input_width` cannot be passed directly to the command line, so if you are using the CLI these must be specified in a [YAML file](yaml-config.md).

If you are using MegadetectorLite to select frames (which is the default for the official models we ship with), you can also decrease the size of the frame used at this stage by setting [`frame_selection_height` and `frame_selection_width`](configurations/#frame_selection_height-int-optional-frame_selection_width-int-optional).

=== "YAML file"
    ```yaml
    video_loader_config:
        frame_selection_height: 400  # if using megadetectorlite
        frame_selection_width: 600  # if using megadetectorlite
        model_input_height: 100
        model_input_width: 100
        total_frames: 16 # total_frames is always required
    ```
=== "Python"
    ```python
    video_loader_config = VideoLoaderConfig(
        frame_selection_height=400, frame_selection_width=600,  # if using megadetectorlite
        model_input_height=100, model_input_width=100,
        total_frames=16,
    ) # total_frames is always required
    ```

#### Reducing `num_workers`

Reduce the number of workers (subprocesses) used for data loading. By default `num_workers` will be set to 3. The minimum value is 0, which means that the data will be loaded in the main process, and the maximum is one less than the number of CPUs in the system. We recommend trying 1 if 3 is too many.

=== "CLI"
    ```console
    $ zamba predict --data-dir example_vids/ --num-workers 1

    $ zamba train --data-dir example_vids/ --labels example_labels.csv --num-workers 1
    ```
=== "Python"
    In Python, add `num_workers` to [`PredictConfig`](configurations.md#prediction-arguments) or [`TrainConfig`](configurations.md#training-arguments):
    ```python
    predict_config = PredictConfig(
        data_dir="example_vids/", num_workers=1
    )
    ```

## Logging

To check that videos are getting loaded and cached as expected, set your environment variabe `LOG_LEVEL` to `DEBUG`. The default log level is `INFO`. For example:

```console
$ LOG_LEVEL=DEBUG zamba predict --data-dir example_vids/
```
