# zamba - a command line interface for species classification

Zamba means "forest" in Lingalese.

The `zamba` command will be the entry point for users (see 
example usage below).

[ ![Codeship Status for drivendataorg/chimps-tool](https://app.codeship.com/projects/03e3a040-0b6d-0136-afe4-3aeedc3a22e1/status?branch=master)](https://app.codeship.com/projects/281856)

## Install

### GPU or CPU
To correctly install, the user must specifiy whether the cpu or gpu version of tensorflow should be installed. If the user fails to make this specification, no version of tensorflow will be installed, thus everything will fail.

To install for development with **tensorflow gpu** 
```
> git clone https://github.com/drivendataorg/zamba.git
> cd zamba
> pip install --editable .[tf_gpu]
```

To install for development with **tensorflow cpu**
```
> git clone https://github.com/drivendataorg/zamba.git
> cd zamba
> pip install --editable .[tf]
```

### AV
As shown in the `Dockerfile`, the cross-platform (minus Windows) approach to installing `pyav` uses the `conda` image. So in order to comply with `requirements.txt` and successfully install `av`, use

```
conda install av -c conda-forge
```

This requires Anaconda.

## Test
The included `Makefile` contains code that uses pytest to run all tests in `zamba/tests`.

The command is (from the project root),

```
> make test
```

### Testing End-To-End Prediction With `test_cnnensemble.py`
The test `tests/test_cnnensemble.py` runs an end-to-end prediction with `CnnEnsemble.predict(data_dir)` using a video that automatically gets downloaded along with the `input` directory (this and all required directories are downloaded upon instantiation of `CnnEnsemble` if they are not already present in the project).

By default this test is skipped due to the `pytest` decorator

```
@pytest.mark.skip(reason="This test takes hours to run, makes network calls, and is really for local dev only.")
def test_predict():


    data_dir = Path(__file__).parent.parent / "models" / "cnnensemble" / "input" / "raw_test"

    manager = ModelManager('', model_class='cnnensemble', proba_threshold=0.5)
    manager.predict(data_dir, save=True)
```

It is reccomended that the **decorator be commented out in order to test end-to-end prediction locally**. However, this change should never be pushed, as the lightweight machines on codeship will not be happy, or able, to complete the end-to-end prediction.

To test end-to-end prediction using `make test` on a different set of videos, simply edit `data_dir`.

## Example usage

`zamba predict path/to/vids`
