# cmd - a command line interface for species classification

Lacking a name as of yet, we're using `cmd`, short for "command" to refer to 
the 
application. The `cmd` command will be the entry point for users (see 
example usage below).

## Install
To install for development, `pip install --editable .` from project root `cmd`.

## Test

Run tests and coverage report (output to terminal) from project root `cmd` 
with `python -m pytest --cov=. --cov-report=term`.

Note that the tests contained in `src/tests/test_model_save_and_load.py` are
 currently **required** to run in order for the `test-model` referenced in the 
 example below to exist.

## Example usage
To call a test prediction from e.g. a directory that sits next to `cmd`, use
`cmd predict --modelpath ../cmd/src/src/models/assets/test-model/`

This will print the returned DataFrame (one element) and save the DataFrame 
as a csv into the current directory (here assumed to sit outside of `cmd`). 

The path given for `modelpath` is automatically generated after testing (see
 above), so run tests before trying this example.
 
 Although the model created and loaded is a tensorflow model, all it does is
  add 2 numbers and multiply (tensorflow can do math!), it doesn't predict 
  anything based on training data... yet.
