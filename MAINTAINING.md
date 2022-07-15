## Adding a new model (for maintainers)

_Note: These instructions are for adding or updating an [official model](https://github.com/drivendataorg/zamba/tree/master/zamba/models/official_models). User-trained models should instead be shared via the [Community Model Zoo](https://github.com/drivendataorg/zamba/wiki) wiki page._

TLDR; update the the `WEIGHT_LOOKUP` mapping and then run `make publish_models`.

#### Weight lookup mapping

The weight lookup mapping in `zamba/models/config.py` connects each model to the private s3 directory where model training results live. These directories are only used as the source of what to publish. All user-facing model weights come from the public buckets, and official configs are contained within the zamba package.

Here's an example `WEIGHT_LOOKUP` dictionary:
```
WEIGHT_LOOKUP = {
    "time_distributed": "s3://drivendata-client-zamba/data/results/experiments/td_small_set_full_size_mdlite/version_1/",
    "european": "s3://drivendata-client-zamba/data/results/experiments/european_td_dev_base/version_0/",
    "slowfast": "s3://drivendata-client-zamba/data/results/experiments/slowfast_small_set_full_size_mdlite/version_2/",
}
```

The s3 directory for a given model should contain the following files at the root level (extras will be ignored):
- `{model}.ckpt`
- `config.yaml`
- `train_configuration.yaml`
- `predict_configuration.yaml`
- `hparams.yaml`
- `val_metrics.json`

### Instructions

To update an existing model:
- [ ] Update the s3 path of the directory for that model in the [`WEIGHT_LOOKUP` dictionary](https://github.com/drivendataorg/zamba/blob/master/zamba/models/config.py)

Or, to add a new model:
- [ ] Add a new entry in the [`WEIGHT_LOOKUP` dictionary](https://github.com/drivendataorg/zamba/blob/master/zamba/models/config.py)

Then (for both cases):
- [ ] Run `make publish_models`

This will copy the relevant files over to model folder within the official models directory, and modify them to keep only the relevant portions for the package. Model weights will be copied to the relevant public buckets, and the location of the public checkpoints will be added to the official configs.

Example logs:
```
2022-07-15 11:48:13.805 | INFO     | __main__:<module>:148 -
============
Preparing time_distributed model
============
2022-07-15 11:36:54.344 | INFO     | __main__:publish_model:81 - Copying over yaml and json files from s3://drivendata-client-zamba/data/results/zamba_classification_retraining/td_small_set_new_frame_selection/version_1 to /Users/emily/ds/drivendata/zamba/zamba/models/official_models/time_distributed.
2022-07-15 11:36:55.200 | INFO     | __main__:publish_model:93 - Preparing official config file.
2022-07-15 11:36:55.208 | INFO     | __main__:publish_model:125 - Writing out to /Users/emily/ds/drivendata/zamba/zamba/models/official_models/time_distributed/config.yaml
2022-07-15 11:36:55.784 | INFO     | __main__:upload_to_all_public_buckets:141 - Uploading s3://drivendata-client-zamba/data/results/zamba_classification_retraining/td_small_set_new_frame_selection/version_1/time_distributed.ckpt to s3://drivendata-public-assets/zamba_official_models/time_distributed_f5072dafff.ckpt
2022-07-15 11:37:35.419 | INFO     | __main__:upload_to_all_public_buckets:141 - Uploading s3://drivendata-client-zamba/data/results/zamba_classification_retraining/td_small_set_new_frame_selection/version_1/time_distributed.ckpt to s3://drivendata-public-assets-eu/zamba_official_models/time_distributed_f5072dafff.ckpt
2022-07-15 11:37:59.960 | INFO     | __main__:upload_to_all_public_buckets:139 - Uploading s3://drivendata-client-zamba/data/results/zamba_classification_retraining/td_small_set_new_frame_selection/version_1/time_distributed.ckpt to s3://drivendata-public-assets-asia/zamba_official_models/time_distributed_f5072dafff.ckpt
2022-07-15 11:48:17.871 | INFO     | __main__:<module>:148 -
============
Preparing slowfast model
============
2022-07-15 11:48:17.913 | INFO     | __main__:publish_model:81 - Copying over yaml and json files from s3://drivendata-client-zamba/data/results/zamba_v2_classification/experiments/slowfast_small_set_full_size_mdlite/version_2 to /Users/emily/ds/drivendata/zamba/zamba/models/official_models/slowfast.
2022-07-15 11:48:19.028 | INFO     | __main__:publish_model:93 - Preparing official config file.
2022-07-15 11:48:19.035 | INFO     | __main__:publish_model:125 - Writing out to /Users/emily/ds/drivendata/zamba/zamba/models/official_models/slowfast/config.yaml
2022-07-15 11:48:19.150 | INFO     | __main__:upload_to_all_public_buckets:139 - Skipping since s3://drivendata-public-assets/zamba_official_models/slowfast_3c9d5d0c72.ckpt exists.
2022-07-15 11:48:19.340 | INFO     | __main__:upload_to_all_public_buckets:139 - Skipping since s3://drivendata-public-assets-eu/zamba_official_models/slowfast_3c9d5d0c72.ckpt exists.
2022-07-15 11:48:19.554 | INFO     | __main__:upload_to_all_public_buckets:139 - Skipping since s3://drivendata-public-assets-asia/zamba_official_models/slowfast_3c9d5d0c72.ckpt exists.
2022-07-15 11:48:19.555 | INFO     | __main__:<module>:148 -
============
Preparing european model
============
2022-07-15 11:48:19.602 | INFO     | __main__:publish_model:81 - Copying over yaml and json files from s3://drivendata-client-zamba/data/results/zamba_v2_classification/european_td_dev_base/version_0 to /Users/emily/ds/drivendata/zamba/zamba/models/official_models/european.
2022-07-15 11:48:20.638 | INFO     | __main__:publish_model:93 - Preparing official config file.
2022-07-15 11:48:20.653 | INFO     | __main__:publish_model:125 - Writing out to /Users/emily/ds/drivendata/zamba/zamba/models/official_models/european/config.yaml
2022-07-15 11:48:20.766 | INFO     | __main__:upload_to_all_public_buckets:139 - Skipping since s3://drivendata-public-assets/zamba_official_models/european_0a80dc77bf.ckpt exists.
2022-07-15 11:48:20.948 | INFO     | __main__:upload_to_all_public_buckets:139 - Skipping since s3://drivendata-public-assets-eu/zamba_official_models/european_0a80dc77bf.ckpt exists.
2022-07-15 11:48:21.166 | INFO     | __main__:upload_to_all_public_buckets:139 - Skipping since s3://drivendata-public-assets-asia/zamba_official_models/european_0a80dc77bf.ckpt exists.
2022-07-15 11:48:21.166 | INFO     | __main__:<module>:152 -
============
Preparing DensePose model: animals
============
2022-07-15 11:48:21.274 | INFO     | __main__:<module>:157 - Skipping since model exists on main public S3 bucket.
2022-07-15 11:48:21.274 | INFO     | __main__:<module>:152 -
============
Preparing DensePose model: chimps
============
2022-07-15 11:48:21.387 | INFO     | __main__:<module>:157 - Skipping since model exists on main public S3 bucket.
```

Lastly:
- [ ] Submit a PR with the changed or additional files
- [ ] Celebrate that a new model is available! :tada: