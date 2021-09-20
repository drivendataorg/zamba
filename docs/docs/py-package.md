# `zamba` as a Python Module

<!-- TODO: this page><!-->

If you want to use `zamba` as part of a pipeline, it can be used directly in python scripts instead of from the command line. Any functionality available in the command line interface is also accessible in the Python package.

## Generating predictions

To generate predictions for the same folder, `vids_to_classify`, run:

- example python code for generating predictions

- explanation of predictconfig and videoloaderconfig. point to videoloaderconfig section

- output of help(PredictConfig) (if we add docstrings)

- explain required arguments, point to all configurations for the rest of the arguments

- ### examples


## Training a model


## VideoLoaderConfig

- both train and predict use the vidloadconfig class to specify any optional parameters for video loading
- help(VideoLoaderConfig) beginning output. 
- none of these are required
- point to all configurations page




reference: https://github.com/drivendataorg/zamba-algorithms/pull/405



### Advanced example

```python
from cloudpathlib import S3Path
import pandas as pd

from zamba_algorithms.data.video import VideoLoaderConfig
from zamba_algorithms.models.config import TrainConfig
from zamba_algorithms.models.model_manager import train_model
from zamba_algorithms.settings import ROOT_DIRECTORY

# set up csv of labels
df = pd.read_csv(ROOT_DIRECTORY / "data" / "processed" / "unified_metadata.csv", low_memory=False)
df["filepath"] = df.filepath.apply(lambda x: S3Path(x).key)
df.rename(columns={"zamba_label_new": "label"}, inplace=True)
df = df[df.label.notnull()].copy().head(20)

video_loader_config = VideoLoaderConfig(
    video_height=224,
    video_width=224,
    crop_bottom_pixels=50,
    ensure_total_frames=True,
    megadetector_lite_config={"confidence": 0.25, "fill_mode": "score_sorted", "n_frames": 16},
    total_frames=16,
)

train_config = TrainConfig(
    labels=df[["filepath", "label"]],
    model_name="time_distributed",
    batch_size=8,
    backbone_finetune=True,
    backbone_finetune_params={
        "unfreeze_backbone_at_epoch": 3,
        "verbose": True,
        "pre_train_bn": True,
        "multiplier": 1,
    },
    num_workers=3,
    auto_lr_find=True,
    early_stopping_params={
        "patience": 5,
    },
    model_params={
        "scheduler": "MultiStepLR",
        "scheduler_params": {"milestones": [3], "gamma": 0.5, "verbose": True},
    },
)

train_model(train_config, video_loader_config)
```




***
OLD

## Use zamba in Python

The main API for `zamba` is the `ModelManager` class that can be accessed with:

```python
from zamba.models.manager import ModelManager
```

The `ModelManager` class is used by `zamba`'s
[command line interface](cli.md) to handle preprocessing the
filenames, loading the videos, serving them to the model, and saving
predictions. Therefore any functionality available to the command line
interface is accessible via the `ModelManager` class.


### Example Prediction in Python using the `ModelManager`

Using the same example directory, `vids_to_classify/` as we used in the
[Quickstart](index.md), we now show how to make the same predictions
within Python. This means that ultimately, the `zamba` functionality could be
wrapped within more complicated Python analysis pipelines.

First list the videos (in the command line, for simplicity):

```
$ ls vids_to_classify/
blank.mp4
chimp.mp4
eleph.mp4
leopard.mp4
```

Now start Python, import the `ModelManager`, and make a prediction:

<!-- TODO: update the outputs below once zamba is running><!-->

<!-- TODO: update import path and syntax if it changes><!-->

<!-- TODO: right now .predict doesn't return the df, it just saves it. I think we should add that. Either add it or change the code below><!-->
```python
from zamba.models.model_manager import ModelManager
manager = ModelManager('time_distributed')
preds = manager.predict('vids_to_classify/', save=True)

 # Predicting on 4 L1 models:  25%|███████████▌                                  | 1/4 [02:04<06:14, 124.92s/it]
 # Processing 5 videos:  80%|███████████████████████████████████████████████████████▏             | 4/5 [01:34<00:23, 23.74s/it
...

preds.head()

#                        bird     blank        cattle    chimpanzee  elephant  \
# filename
# blank2.mp4     3.095961e-05  0.998250  5.992575e-08  1.880314e-05  0.000001
# eleph.mp4      2.491223e-06  0.000169  1.576770e-08  8.856079e-06  0.999592
# blank1.mp4     1.860761e-03  0.974185  3.433625e-06  9.265547e-04  0.000017
# ungulate.mp4   4.071459e-07  0.022434  6.694263e-04  7.227396e-08  0.000007
# small-cat.mp4  9.654887e-05  0.022991  3.805580e-04  4.553670e-05  0.000002

#                forest buffalo       gorilla  hippopotamus     human  \
# filename
# blank2.mp4       3.058858e-07  3.417223e-08  3.069252e-07  0.000079
# eleph.mp4        1.278719e-07  1.288049e-08  1.723544e-08  0.000018
# blank1.mp4       5.102556e-06  3.214902e-06  6.939250e-06  0.012967
# ungulate.mp4     2.905669e-04  2.747548e-04  1.117668e-05  0.000029
# small-cat.mp4    1.159542e-03  9.205705e-05  2.968036e-04  0.001000

#                       hyena      ...       other (primate)      pangolin  \
# filename                         ...
# blank2.mp4     3.319698e-07      ...              0.000073  2.684126e-07
# eleph.mp4      2.941773e-08      ...              0.000006  3.506190e-08
# blank1.mp4     6.278733e-06      ...              0.005580  5.494129e-06
# ungulate.mp4   9.451887e-05      ...              0.000015  6.508603e-06
# small-cat.mp4  3.999528e-04      ...              0.000077  2.511233e-04

#                   porcupine       reptile    rodent  small antelope  \
# filename
# blank2.mp4     2.626962e-06  4.384665e-07  0.000429    3.357736e-07
# eleph.mp4      1.264149e-08  1.647070e-07  0.000003    6.065281e-07
# blank1.mp4     2.311535e-05  8.976383e-06  0.000591    7.157736e-06
# ungulate.mp4   5.827179e-08  4.493329e-05  0.000018    6.327724e-05
# small-cat.mp4  4.427238e-05  1.429361e-04  0.000206    6.781974e-04

#                   small cat      wild dog    duiker           hog
# filename
# blank2.mp4     2.783504e-08  3.663065e-07  0.000287  5.982519e-05
# eleph.mp4      4.909233e-09  2.605069e-08  0.000016  8.522910e-07
# blank1.mp4     1.417417e-06  6.247396e-06  0.004685  2.489997e-04
# ungulate.mp4   2.148184e-05  1.775765e-04  0.000051  2.075789e-05
# small-cat.mp4  9.729478e-01  3.508627e-04  0.008978  5.116147e-04
```

A few of things to note.<!-- TODO: update these>

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
[Quickstart](index.md).

#### You can find all of the parameters for `predict` in the module documentation.
