# Help make `zamba` better

`zamba` is an open source project, which means _you_ can help make it better!

## Develop the GitHub repository

To get involved, check out the GitHub [code repository](https://github.com/drivendataorg/zamba).
There you can find [open issues](https://github.com/drivendataorg/zamba/issues) with comments and links to help you along.

`zamba` uses continuous integration and test-driven development to ensure that we always have a working project. So what are you waiting for? `git` going!

## Installation for development

To install `zamba` for development, clone the git repository and install the package with the dev extras. We recommend [uv](https://docs.astral.sh/uv/) for managing the environment.

With uv (recommended):
```console
$ git clone https://github.com/drivendataorg/zamba.git
$ cd zamba
$ uv pip install -e ".[tests,image,video,docs]"
```

Or use the Makefile target (which uses uv):
```console
$ make requirements
```

With pip:
```console
$ pip install -e ".[tests,image,video,docs]"
```

If your contribution is to the [DensePose](../models/densepose.md) model, install the DensePose dependencies from GitHub as described in the [DensePose installation](../models/densepose.md#installation) section.

To build the documentation locally, install the `docs` extra (e.g. `uv pip install -e ".[docs]"` or `pip install -e ".[docs]"`).

## Running the `zamba` test suite

The included [`Makefile`](https://github.com/drivendataorg/zamba/blob/master/Makefile) contains code that uses pytest to run all tests in `zamba/tests`.

The command is (from the project root):

```console
$ make tests
```

For [DensePose](../models/densepose.md) related tests, install the DensePose dependencies from GitHub (see [DensePose installation](../models/densepose.md#installation)), then run:
```console
$ make densepose-tests
```

## Submit additional training videos

If you have additional labeled videos that may be useful for improving the basic models that ship with `zamba`, we'd love to hear from you! You can get in touch at [info@drivendata.org](mailto:info@drivendata.org)
