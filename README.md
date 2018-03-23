# zamba - a command line interface for species classification

The `zamba` command will be the entry point for users (see 
example usage below).

[ ![Codeship Status for drivendataorg/chimps-tool](https://app.codeship.com/projects/03e3a040-0b6d-0136-afe4-3aeedc3a22e1/status?branch=master)](https://app.codeship.com/projects/281856)

## Install
To install for development, `pip install --editable .` from project root `zamba`.

## Test

Run tests and coverage report (output to terminal) from project root `zamba` 
with `python -m pytest --cov=. --cov-report=term`.

Tests contained in `zamba/zamba/tests`

## Example usage

`zamba predict path/to/vids`

 
 Although the model created and loaded is a `tensorflow.python.keras` model, all it does is
  add 2 numbers as well as multiply them (tensorflow can do math!), it doesn't predict 
  anything based on training data... yet.
