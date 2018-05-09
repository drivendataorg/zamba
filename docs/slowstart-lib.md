# zamba as a Python Module

If you want to use `zamba` as part of a pipeline, it can be used directly in python scripts instead of from the command line.

## Use zamba in Python

The main API for `zamba` is the `ModelManager` class that can be accessed with:

```
>>> from zamba.models.manager import ModelManager
>>>
```

The `ModelManager` class is used by `zamba`'s
[command line interface](slowstart-cli.html) to handle preprocessing the
filenames, loading the videos, serving them to the model, and saving
predictions. Therefore any functionality available to the command line
interface is accessible via the `ModelManager` class.


### Example Prediction in Python using the `ModelManager`

Using the same example directory, `vids_to_classify/` as we used in the
[Quickstart](quickstart.html), we now show how to make the same predictions
within Python. This means that ultimately, the `zamba` functionality could be
wrapped within more complicated Python analysis pipelines.

First list the videos (in the command line, for simplicity):

```
$ ls vids_to_classify/
blank1.mp4
blank2.mp4
eleph.mp4
small-cat.mp4
ungulate.mp4
```

Now start Python, import the `ModelManager`, and make a prediction:

```
$ python
>>> from zamba.models.manager import ModelManager
>>> manager = ModelManager(model_class='cnnensemble')
>>> manager.predict('vids_to_classify/', save=True)
nasnet_mobile
blank2.mp4  1 prepared in 3249 predicted in 24662
eleph.mp4  2 prepared in 0 predicted in 24659
blank1.mp4  3 prepared in 0 predicted in 23172
ungulate.mp4  4 prepared in 0 predicted in 30598
small-cat.mp4  5 prepared in 0 predicted in 27267
inception_v3
blank2.mp4  1 prepared in 3187 predicted in 39506
eleph.mp4  2 prepared in 0 predicted in 38279
blank1.mp4  3 prepared in 0 predicted in 37012
ungulate.mp4  4 prepared in 0 predicted in 41095
small-cat.mp4  5 prepared in 0 predicted in 46365
xception_avg
blank2.mp4  1 prepared in 2941 predicted in 64698
eleph.mp4  2 prepared in 0 predicted in 63569
blank1.mp4  3 prepared in 0 predicted in 62037
ungulate.mp4  4 prepared in 0 predicted in 52290
small-cat.mp4  5 prepared in 0 predicted in 53961
inception_v2_resnet
blank2.mp4  1 prepared in 3995 predicted in 74176
eleph.mp4  2 prepared in 0 predicted in 64319
blank1.mp4  3 prepared in 0 predicted in 71720
ungulate.mp4  4 prepared in 0 predicted in 76737
small-cat.mp4  5 prepared in 0 predicted in 89486
```
A few of things to note.

1. The data directory `vids_to_classify/` should be passed to the `ModelManager` as a string.
2. The data path and prediction path are not listed at the start of prediction,
 as they were in the command line interface. In other words, the verbosity is
 less.
3. The `save=True` argument passed to the predict method will lead to the same
`output.csv` as before. By default this argument is `False`, so calling
`manager.predict('vids_to_classify/')` _without_ `save=True` would **not** save
 the output, although the output would still be sent to standard out.

Otherwise, the output is just the same! Unless otherwise specified using the
`pred_path` argument in `ModelManager.predict`, the predictions will be output
to the working directory under the name `output.csv`, just as specified in the
[Quickstart](quickstart.html).

#### You can find all of the parameters for `predict` in the module documentation.
