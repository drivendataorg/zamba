# `zamba` changelog

## v.2.7.0 (2026-02-17)

 - Split core dependencies into optional extras: `video` (av, ffmpeg-python, pytorchvideo, pixeltable-yolox, etc.), `image` (megadetector, Pillow), `tests` (pytest, black, flake8, coverage, nvidia-ml-py, etc.), and `docs` (mkdocs, mike, mkdocstrings). Base install no longer pulls in video/image stacks; use `pip install zamba[video]`, `zamba[image]`, or `zamba[video,image]`.
 - Remove `requirements-dev.txt` and `requirements-dev/`; use `uv pip install -e ".[image,video]" --group dev` for development or `pip install -e ".[tests,image,video,docs]"` with pip.
 - Replace `mlflow` with `mlflow-skinny` in core dependencies. Add `pyarrow>=23.0.0`. Add Windows-specific torch version constraint for gloo bug. Declare Python 3.11–3.13 support and `requires-python = ">=3.11, <3.14"`.
 - Remove DensePose from optional dependencies so the package is publishable to PyPI. DensePose (detectron2, detectron2-densepose) must be installed from GitHub; see docs. Running `zamba densepose` without them prints install instructions and exits. Add `[tool.uv.extra-build-dependencies]` for detectron2 (torch) so CI can build it from GitHub.
 - Pin `setuptools<82` so the image extra works on Python 3.12 (megadetector’s yolov5 stack requires `pkg_resources`, which setuptools 82 removed). Add `[tool.uv]` override-dependencies for protobuf and setuptools.
 - Add `zamba.models.config_common` with shared types and validators (ModelEnum, MonitorEnum, ZambaBaseModel, RegionEnum, get_filepaths, validate_model_cache_dir, etc.). Move video-specific file/checkpoint logic into `zamba.models.config` and reuse common helpers.
 - Add `zamba.models.instantiation` and move `instantiate_model`, head-replacement, and resume logic out of `model_manager` for clearer separation.
 - Add lazy model registration in `zamba.models.registry` (`ensure_registered()`): video model classes are imported only when needed so the package can be imported without video dependencies. Config validation calls `ensure_registered()` when resolving checkpoint/model name.
 - Register image and utils sub-apps lazily so `zamba --help` and non-image/non-utils commands work without image or densepose dependencies. Defer imports of VideoLoaderConfig and config classes into the train, predict, and depth command callbacks.
 - Handle missing DensePose dependencies in the `densepose` command: catch ImportError when running the model and print install-from-GitHub instructions and exit with code 1.
 - Make NPY cache path hashing stable: use a JSON-serializable, order-invariant representation of the config (including Enum and Path) instead of `str(hashed_part)`. For local absolute paths, use a relative-style path in the cache key.
 - Fix cache cleanup in `npy_cache.__del__` by comparing `Path(cache_path).parents[0]` with `Path(tempfile.gettempdir())`.
 - Import image classifier lazily from `zamba.images` to avoid pulling in torch at package import time. Images config and manager use `zamba.models.config_common` and `zamba.models.instantiation` instead of config/utils from video.
 - Make megadetector import resilient: try `megadetector.detection.run_detector`, fall back to `detection.run_detector`. Make MLflow optional in image training (log warning and continue without it if import or setup fails). Disable `torch.compile` on Windows in addition to macOS.
 - Add pytest markers `video` and `image` and skip test files when the corresponding extra is not installed (conftest `collect_ignore` based on `_HAS_VIDEO` / `_HAS_IMAGE`). Define video-only fixtures and the dummy video model only when video deps are present. Set `CUDA_VISIBLE_DEVICES=0` in conftest to avoid DDP issues under pytest-xdist.
 - Add Makefile targets `test-fast` (fail on first failure) and `test-image-only` / `test-video-only` (isolated venvs with only image or video extra). CI runs these isolation steps and installs image,video extras for the main test matrix. DensePose tests install detectron2 from GitHub with `--no-build-isolation` instead of using a densepose extra.
 - Update tests to use new config/instantiation imports and markers; add image dataset import check in the install smoke test.
 - Installation docs: document optional extras (video, image), PyPI install (`pip install zamba[video]` etc.), and Windows (image extra no extra tools; video needs Visual Studio Build Tools and FFmpeg). State FFmpeg only required for video workflows and recommend FFmpeg 4.x. Contribute page: dev install with `--group dev`, DensePose deps from GitHub, `make requirements` with uv. DensePose doc: install detectron2/detectron2-densepose from GitHub; note that `zamba densepose` prints instructions if deps missing.
 - Re-enable PyPI publish steps in the release workflow (Test PyPI and Production PyPI). Docs workflows and release job install `.[docs]` instead of `requirements-dev/docs.txt`.
 - Add `deterministic` option (default `True`) to video and image prediction configs to seed RNGs and request best-effort deterministic CUDA/cuDNN during inference. Control the seed with the `INFERENCE_SEED` environment variable (default `55`).
 - Fix non-deterministic frame ordering in MegadetectorLite `score_sorted` fill mode when frame scores tie; equal scores now break by lowest frame index.

## v.2.6.1 (2025-03-18)

 - Fix bug that prevented loading checkpoints when model name was provided ([PR #359](https://github.com/drivendataorg/zamba/pull/359))

## v.2.6.0 (2025-03-14)

* Added support for training and classifying on images ([PR #349](https://github.com/drivendataorg/zamba/pull/349))

## v.2.5.0 (2024-09-27)

* Removed support for Python 3.8 and 3.9 plus other requirement updates ([PR #337](https://github.com/drivendataorg/zamba/pull/337), [PR #335](https://github.com/drivendataorg/zamba/pull/335)). New minimum python version is 3.11.

## v2.4.1 (2024-04-20)

* Bug fixes for docs

## v2.4.0 (2024-04-19)

* Adds experimental image support ([PR #314](https://github.com/drivendataorg/zamba/pull/314))
* Clarifies installation instructions for Linux and Windows operating systems ([PR #299](https://github.com/drivendataorg/zamba/pull/299))

## v2.3.2 (2023-07-17)

* Pin Pydantic to less than v2.0 ([PR #277](https://github.com/drivendataorg/zamba/pull/277))

## v2.3.1 (2023-05-12)

* Code updates for PyTorch Lightning v2.0.0 ([PR #266](https://github.com/drivendataorg/zamba/pull/266), [PR #272](https://github.com/drivendataorg/zamba/pull/272))
* Switch to pyproject.toml-based build and other requirement updates ([PR #254](https://github.com/drivendataorg/zamba/pull/254), [PR #255](https://github.com/drivendataorg/zamba/pull/255), [PR #260](https://github.com/drivendataorg/zamba/pull/260), [PR #262](https://github.com/drivendataorg/zamba/pull/262), [PR #268](https://github.com/drivendataorg/zamba/pull/268))

## v2.3.0 (2022-12-01)

### Model release

* Adds a depth estimation module for predicting the distance between animals and the camera ([PR #247](https://github.com/drivendataorg/zamba/pull/247)). This model comes from one of the winning solutions in the [Deep Chimpact: Depth Estimation for Wildlife Conservation](https://www.drivendata.org/competitions/82/competition-wildlife-video-depth-estimation/) machine learning challenge hosted by DrivenData.

## v2.2.4 (2022-11-10)

* Do not cache videos if the `VIDEO_CACHE_DIR` environment variable is an empty string or zero ([PR #245](https://github.com/drivendataorg/zamba/pull/245))

## v2.2.3 (2022-11-01)

* Fixes Lightning deprecation of DDPPlugin ([PR #244](https://github.com/drivendataorg/zamba/pull/244))

## v2.2.2 (2022-10-04)

* Adds a page to the docs summarizing the performance of the African species classification model on a holdout set ([PR #235](https://github.com/drivendataorg/zamba/pull/235))

## v2.2.1 (2022-09-27)

* Turn off showing local variables in Typer's exception and error handling ([PR #237](https://github.com/drivendataorg/zamba/pull/237))
* Fixes bug where the column order was incorrect for training models when the provided labels are a subset of the model's default labels ([PR #236](https://github.com/drivendataorg/zamba/pull/236))

## v2.2.0 (2022-09-26)

### Model releases and new features

* The default `time_distributed` model (African species classification) has been retrained on over 250,000 videos. This 16x increase in training data significantly improves accuracy. This new version replaces the previous one. ([PR #226](https://github.com/drivendataorg/zamba/pull/226), [PR #232](https://github.com/drivendataorg/zamba/pull/232))
* A new default model option is added: `blank_nonblank`. This model only does blank detection. This binary model can be trained and finetuned in the same way as the species classification models. This model was trained on both African and European data, totaling over 263,000 training videos. ([PR #228](https://github.com/drivendataorg/zamba/pull/228))
* Detect if a user is training in a binary model and preprocess the labels accordingly ([PR #215](https://github.com/drivendataorg/zamba/pull/215))

### Bug fixes and improvements

* Add a validator to ensure that using a model’s default labels is only possible when the species in the provided labels file are a subset of those ([PR #229](https://github.com/drivendataorg/zamba/pull/229))
* Refactor the logic in `instantiate_model` for clarity ([PR #229](https://github.com/drivendataorg/zamba/pull/229))
* Use pqdm to check for missing files in parallel ([PR #224](https://github.com/drivendataorg/zamba/pull/224))
* Set `model_name` based on the provided checkpoint so that user-trained models use the appropriate video loader config ([PR #221](https://github.com/drivendataorg/zamba/pull/221))
* Leave `data_dir` as a relative path ([PR #219](https://github.com/drivendataorg/zamba/pull/219))
* Ensure hparams yaml files get included in the source distribution ([PR #210](https://github.com/drivendataorg/zamba/pull/210))
* Hold back setuptools so mkdocstrings works ([PR #207](https://github.com/drivendataorg/zamba/pull/207))
* Factor out `get_cached_array_path` ([PR #202](https://github.com/drivendataorg/zamba/pull/202/files))

## v2.1.0 (2022-07-15)

- Retrains the time distributed species classification model using the updated MegadetectorLite frame selection ([PR #199](https://github.com/drivendataorg/zamba/pull/199))
- Replaces the MegadetectorLite frame selection model with an improved model trained on significantly more data ([PR #195](https://github.com/drivendataorg/zamba/pull/195))

## v2.0.4 (2022-06-17)

 - Pins `thop` to an earlier version ([PR #191](https://github.com/drivendataorg/zamba/pull/191))
 - Fixes caching so a previously downloaded checkpoint file actually gets used ([PR #190](https://github.com/drivendataorg/zamba/pull/190), [PR #194](https://github.com/drivendataorg/zamba/pull/194))
 - Removes a lightning deprecation warning for DDP ([PR #187](https://github.com/drivendataorg/zamba/pull/187))
 - Ignores extra columns in the user-provided labels or filepaths csv ([PR #186](https://github.com/drivendataorg/zamba/pull/186))

## v2.0.3 (2022-05-06)

Releasing to pick up #179.

 - PR [#179](https://github.com/drivendataorg/zamba/pull/179) removes the DensePose extra from the default dev requirements and tests. Docs are updated to clarify how to install and run tests for DensePose.

## v2.0.2 (2021-12-21)

Releasing to pick up #172.

 - PR [#172](https://github.com/drivendataorg/zamba/pull/172) fixes bug where video loading that uses the YoloX model (all of the built in models) resulted in videos not being able to load.


## v2.0.1 (2021-12-15)

Releasing to pick up #167 and #169.

 - PR [#169](https://github.com/drivendataorg/zamba/pull/169) fixes error in splitting data into train/test/val when only a few videos.
 - PR [#167](https://github.com/drivendataorg/zamba/pull/167) refactors yolox into an `object_detection` module

Other documentation fixes also included.

## v2.0.0 (2021-10-22)

### Previous model: Machine learning competition

The algorithms used by `zamba` v1 were based on the winning solution from the
[Pri-matrix Factorization](https://www.drivendata.org/competitions/49/deep-learning-camera-trap-animals/) machine learning competition, hosted by [DrivenData](https://www.drivendata.org/). Data for the competition was provided by the [Chimp&See project](https://www.chimpandsee.org/#/) and manually labeled by volunteers. The competition had over 300 participants and over 450 submissions throughout the three month challenge. The v1 algorithm was adapted from the winning competition submission, with some aspects changed during development to improve performance.

The core algorithm in `zamba` v1 was a [stacked ensemble](https://en.wikipedia.org/wiki/Ensemble_learning#Stacking) which consisted of a first layer of models that were then combined into a final prediction in a second layer. The first level of the stack consisted of 5 `keras` deep learning models, whose individual predictions were combined in the second level
of the stack to form the final prediction.

In v2, the stacked ensemble algorithm from v1 is replaced with three more powerful [single-model options](models/species-detection.md): `time_distributed`, `slowfast`, and `european`. The new models utilize state-of-the-art image and video classification architectures, and are able to outperform the much more computationally intensive stacked ensemble model.

### New geographies and species

`zamba` v2 incorporates data from Western Europe (Germany). The new data is packaged in the pretrained `european` model, which can predict 11 common European species not present in `zamba` v1.

`zamba` v2 also incorporates new training data from 15 countries in central and west Africa, and adds 12 additional species to the pretrained African models.

### Retraining flexibility

Model training is made available `zamba` v2, so users can finetune a pretrained model using their own data to improve performance for a specific ecology or set of sites. `zamba` v2 also allows users to retrain a model on completely new species labels.
