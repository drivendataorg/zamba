# Help Make `zamba` Better

`zamba` is an open source project, which means _you_ can help make it better!

## Develop the Github Repository
To get involved, check out the Github [code repository](https://github.com/drivendataorg/zamba).
There you can find [open issues](https://github.com/drivendataorg/zamba/issues)
, [project goals](https://github.com/drivendataorg/zamba/projects), and plenty
of comments and links to help you along.

`zamba` uses continuous integration and test-driven-development to ensure
that we always have a working project. So what are you
waiting for? `git` going!

## Installation for development

To install `zamba` for development, you need to clone the git repository and then install the cloned version of the library for local development.

To install for development:
```console
$ git clone https://github.com/drivendataorg/zamba.git
$ cd zamba
$ pip install --editable .[cpu]
```
** Note: You can change `cpu` to `gpu` to develop against a gpu version on tensorflow. **

## Running the `zamba` test suite

The included `Makefile` contains code that uses pytest to run all tests in `zamba/tests`.

The command is (from the project root),

```console
$ make test
```

### Testing End-To-End Prediction With `test_cnnensemble.py`
The test `tests/test_cnnensemble.py` runs an end-to-end prediction with `CnnEnsemble.predict(data_dir)` using a video that automatically gets downloaded along with the `input` directory (this and all required directories are downloaded upon instantiation of `CnnEnsemble` if they are not already present in the project).

This test takes a longer time to execute than is possible on continuous integration, so by default this test is skipped due to the `pytest` decorator:

```python
@pytest.mark.skip(reason="This test takes hours to run, makes network calls, and is really for local dev only.")
def test_predict():
    data_dir = Path(__file__).parent.parent / "models" / "cnnensemble" / "input" / "raw_test"

    manager = ModelManager('', model_class='cnnensemble', proba_threshold=0.5)
    manager.predict(data_dir, save=True)
```

This test is important during local development, so it is recommended that the **decorator be commented out to test end-to-end prediction locally**. However, this change should never be pushed, as the lightweight machines on codeship will not be happy, or able, to complete the end-to-end prediction.

To test end-to-end prediction using `make test` on a different set of videos, simply edit `data_dir`.

The included `Makefile` contains code that uses pytest to run all tests in `zamba/tests`.


## Packaging and pushing to PyPI

If you want to update the packaging, you **must update the version number in `setup.py` and `docs/conf.py`**. To test packaging you can run:

```console
$ make build
```

If you have credentials for PyPI in a `~/.pypirc` file then you can push to PyPI. To upload a new version, you can use the make commands:

```console
$ make distribute_testpypi
```

and, once tested:

```console
$ make distribute_pypi
```

## Submit additional training videos

If you have additional labeled videos that may be useful for improving the basic models that ship with `zamba`, we'd love to hear from you! You can get in touch at [info@drivendata.org](mailto:info@drivendata.org)

