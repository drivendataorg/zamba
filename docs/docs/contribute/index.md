# Help make `zamba` better

`zamba` is an open source project, which means _you_ can help make it better!

## Develop the GitHub repository

To get involved, check out the GitHub [code repository](https://github.com/drivendataorg/zamba).
There you can find [open issues](https://github.com/drivendataorg/zamba/issues) with comments and links to help you along.

`zamba` uses continuous integration and test-driven development to ensure that we always have a working project. So what are you waiting for? `git` going!

## Installation for development

To install `zamba` for development, you need to clone the git repository and then install the cloned version of the library for local development.

To install for development:
```console
$ git clone https://github.com/drivendataorg/zamba.git
$ cd zamba
$ pip install -r requirements-dev.txt
```

If your contribution is to the [DensePose](../models/densepose.md) model, you will need to install the additional dependencies with:
```console
$ pip install -e .[densepose]
```

## Running the `zamba` test suite

The included [`Makefile`](https://github.com/drivendataorg/zamba/blob/master/Makefile) contains code that uses pytest to run all tests in `zamba/tests`.

The command is (from the project root):

```console
$ make tests
```

For [DensePose](../models/densepose.md) related tests, the command is:
```console
$ make densepose-tests
```

## Submit additional training videos

If you have additional labeled videos that may be useful for improving the basic models that ship with `zamba`, we'd love to hear from you! You can get in touch at [info@drivendata.org](mailto:info@drivendata.org)
