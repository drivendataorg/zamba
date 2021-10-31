# Maintainers Documentation

This page contains documentation for Zamba maintainers.

## Release Instructions

To release a new version of `zamba`:

1. Bump the version in [`setup.cfg`](https://github.com/drivendataorg/zamba/blob/master/setup.cfg). The version number should follow [semantic versioning](https://semver.org/).
2. Create a new release using the [GitHub releases UI](https://github.com/drivendataorg/zamba/releases/new). The tag version must have a prefix `v`, e.g., `v2.0.1`.

On publishing of the release, the [`release`](https://github.com/drivendataorg/zamba/blob/master/.github/workflows/release.yml) GitHub action workflow will be triggered. This workflow builds the package and publishes it to PyPI. You will be able to see the workflow status in the [Actions tab](https://github.com/drivendataorg/zamba/actions?query=workflow%3Arelease).

## Documentation Website

The documentation website is an mkdocs static site hosted on Netlify. The built website assets are first staged on the [`gh-pages` branch](https://github.com/drivendataorg/zamba/tree/gh-pages) and then deployed to Netlify automatically using GitHub Actions workflows.

We use the [`mike` tool](https://github.com/jimporter/mike) to manage the documentation versions with the following conventions.
- We keep the docs of the latest patch of a `<major>.<minor>` version, e.g., `v2.2.1` is keyed as `"v2.2"` and titled as `"v2.2.1"`.
- The current stable version is tagged with the alias `"stable"` and has `"(stable)"` as part of the title.
- The head of the `master` branch is keyed as `"~latest"` and titled as `"latest"`

**To deploy the latest docs from the master branch, all you need to do is manually trigger the [`docs-latest` GitHub Actions workflow](https://github.com/drivendataorg/zamba/actions/workflows/docs-latest.yml).**

**To manually deploy a previously released version, you will need to use `mike`. Follow the instructions in the following section.**

### Manual deploy

Note that `mike` needs to be run in the same direct as `mkdocs.yml`. To avoid changing directories all the time (since we keep it inside the `docs/` subdirectory), you can shadow the `mike` command with the following shell function:

```bash
# Put this in your .bash_profile or wherever you put aliases
mike() {
    if [[ -f mkdocs.yml ]]; then
        command mike "$@"
    else
        (cd docs && command mike "$@")
    fi
}
```

The general steps of deploying docs for a specific version involve:

1. Make sure your local `gh-pages` is up to date with GitHub, i.e., `git fetch origin && git checkout gh-pages && git pull origin gh-pages`
2. Switch to whatever commit you're intending to deploy from.
3. Run `make docs`. (This is necessary because of steps needed before running mkdocs things.)
4. Run whatever `mike` command (see below). If you include the `--push` flag, it will also directly push your changes to GitHub. If you don't, it will only commit to your local `gh-pages` and you'll need to then push that branch to GitHub.
5. Trigger the `docs-latest` GitHub actions workflow, which will deploy the `gh-pages` branch to Netlify.

Staging the stable version will be something like this:

```bash
mike deploy v2.1 stable --title="v2.1.1 (stable)" --no-redirect --update-aliases
```

Staging an older version looks something like this:

```bash
mike deploy v1.1 --title="v1.1.5"
```

## Adding a new model

Only models that were trained with `zamba` can be released.

### Training a model with zamba

To train a new model, specify the desired parameters in a config.yaml and then run
```
zamba train --config my_config.yaml
```

To retrain an existing model, it's easiest to work from one of the templates. You'll want to add any data and machine specific parameters.

If you're retraining and not finetuning, remember to set `from_scratch: true`. This means starting with ImageNet weights for TimeDistributedEfficientNet and Ketics weights for SlowFast.

For example, you might want to add the following to `train_config` in your model template of choice.

```
labels: my_labels.csv
batch_size: 2
num_workers: 3
gpus: 1
save_dir: my_save_dir
skip_load_validation: true
from_scratch: true
```

### Publishing a model

Once your model is trained, copy the resulting directory to a folder in `s3://drivendata-client-zamba/data/results`. The directory should contain the following files:

- model checkpoint file
- `config.yaml` (configuration used for training)
- `train_configuration.yaml` (automatically generated)
- `predict_configuration.yaml` (automatically generated)
- `hparams.yaml` (automatically generated)

*Note: all files should be in the same level.*

To publish an improved version of an existing model,
- update the folder containing model weights in `WEIGHT_LOOKUP` in `zamba/models/config.py`

If you are adding a new model,
- add the model and weights folder as a new entry in the `WEIGHT_LOOKUP` dictionary
- add the model name to `ModelEnum`
- incorporate the new model into the test suite

Then, run
```
make publish_models
```

This will generate a public file name for each model based on the config hash and upload the model weights to the three DrivenData public s3 buckets. This will generate a folder in `zamba/models/official_models/{your_name_name}` that contains the official config as well as reference yaml and json files. You should PR everything in this folder.

Lastly, you need to update the template in `templates`. The template should contain all the same info as the model's `config.yaml`, plus placeholders for `data_dir` and `labels` in `train_config`, and `data_dir`, `filepaths`, and `checkpoint` in `predict_config`.

### New model checklist

- [ ] copy trained model folder to s3 with checkpoint, yaml, and json files
- [ ] update private weights file in `WEIGHT_LOOKUP`
- [ ] run `make publish_models` to generate or update configs in official models folder
- [ ] update model template
- [ ] PR template and subfolder in `official_models`
