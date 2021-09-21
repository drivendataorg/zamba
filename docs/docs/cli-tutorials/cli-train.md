# User Tutorials: Training a model in the Command Line Interface


## Minimum example

To train a model based on the videos in `vids_to_classify` and the labels in `example_labels.csv`:

```console
$ zamba train --data-dir vids/ --labels example_labels.csv
```

<!-- TODO: where will the model be saved?><!-->

## Step-by-step tutorial

## Optional arguments

Each model that ships with the `zamba` package comes with a default configuration saved as a YAML file. The default value of any non-specified parameter will be set based on the model being used - `time_distributed`, `slowfast`, or `european`. Default algorithm configurations can be found in `models/config`.
<!-- TODO: update path to default configs and add link to github folder><!-->

There are two ways to override the default model parameter value, listed in order of precendence:

1. By passing an optional flag directly to the command line
2. By specifying the parameter in a custom YAML configuration file, and passing the YAML filepath to the command line with the `--config` flag. For more details on YAML configuration file options, see the All Configuration Options section. 

If a parameter is both passed directly to the CLI *and* specified in a YAMl file that is passed to the CLI, the value that is passed directly to the CLI will be used.