# zamba-algorithms

A repo for testing algorithms for inclusion in the Zamba package.

## Checklist for adding a new model

PRs with new models should include:
- [ ] model class that subclasses ZambaVideoClassificationLightningModule
- [ ] script that trains and performs inference
- [ ] model evaluation notebook: copy 0.1 and update the notebook name, notebook title, and re-run
- [ ] model training results published to s3: e.g. `make publish_model model_dir=zamba_algorithms/models/tensorboard_logs/first_frame_resnet/version_0`

For example:
- model class: `SingleFrameResnet50` in `models/resnet_models.py`
- model script: `models/single_frame_resnet.py`
- eval notebook: `reports/evaluation_notebooks/0.1-ejm-first-frame-resnet-evaluation.ipynb`
- s3 training results: `data/results/first_frame_resnet/version_0`

## Spikes

The `spikes` folder contains initial experiments with different libraries that we could use to do video classification. These libaries have heavy and sometimes incompatible dependencies. Each one has a separate `requirements.txt` file in it that could be used to install the necessities for that spike.

Each spike has a script `zamba_algorithms/<spike>/train.py` that you can use to train a model using that spike's framework.


## Using AWS `ec2` machines for development (`us-west-1` region)

We've created a zamba deep learning AMI that helpfully has a few things installed and ready to go. In order to use it, create an EC2 machine and select "Zamba deep learning AMI" when asked for the AMI, or [go to the list of AMIs](https://us-west-1.console.aws.amazon.com/ec2/v2/home?region=us-west-1#Images:sort=name) and select it, then click "Launch."

Machine recommendations in `us-west-1`:
 - Get a machine with a GPU. Currently `g4dn` machines are an [ok balance of availability, price, and GPU power](https://instances.vantage.sh/?region=us-west-1).
 - Add >256GB of EBS storage so you can move weights and a fair number of videos locally if necessary.

The AMI is based on the Ubuntu Deep Learning AMI, so the username is `ubuntu`. Follow the normal procedure to use the AWS SSH key for access.

The machine has both the `zamba` repo, and this repo (`zamba-algorithms`) in the home directory. It has a `zamba-algorithms` conda environment already set up. It also has `ffmpeg` installed.

Here are a few things you may need to do to get started:

 - `git pull` in both the `zamba` and `zamba-algorithms` repos to get the latest (NOTE: if you have not configured a [Personal Acces Token](https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token) on GitHub, you will need to create one to use instead of your password for making these pulls at the command line.)
 - `conda activate` both the `zamba-algorithms` and `zamba` environments and `make requirements` in both to ensure you have the latest
 - run `nbautoexport install` in the `zamba-algorithms` environment so that the [configuration is created for your machine user](https://nbautoexport.drivendata.org/command-reference/install/).
 - add your `~/.aws/credentials` by running `aws configure`
 - mount the data with the `make` commands so you can use it
 - if you need `zamba` package features that are not in PyPI, you'll need to `pip install -e ../zamba` within the `zamba-algorithms` environment to install that development version rather than the PyPI one.

## Environmental variables

Environmental variables can be specified in the usual ways and in a `.env` file in the repository root directory.

- `LOAD_VIDEO_FRAMES_CACHE_DIR`: If provided, a persistent cache directory for the `zamba_algorithms.data.video.load_video_frames` function. If not specified, an ephemeral temporary directory will be created for each Python session.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── spikes            <- Experiments with different libraries that require complex dependencies.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so zamba_algorithms can be imported
    ├── zamba_algorithms                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes zamba_algorithms a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

