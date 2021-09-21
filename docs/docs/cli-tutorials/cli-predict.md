# User Tutorials: Classifying videos in the Command Line Interface

This section walks through how to classify videos using the `zamba` command line interface (CLI) tool. If you are new to `zamba` and just want to classify some videos as soon as possible, see the Quickstart guide.

This tutorial goes over the steps for using `zamba` if:

* You already have `zamba` installed (for details see the Installation page)
* You have unlabeled videos that you want to generate labels for, and
* The possible class species labels you want to use are included in the list of possible zamba labels

All of the commands here should be run at the command line. On macOS, this can be done in the terminal (⌘+space, "Terminal"). On Windows, this can be done in a command prompt, or if you installed Anaconda an anaconda prompt (Start > Anaconda3 > Anaconda Prompt).

## Minimum example

If `vids_to_classify` is a folder containing camera trap videos, the minimum for generating predictions is to run:

```console
$ zamba predict --data-dir vids_to_classify/
```

`zamba` will output a `.csv` file with rows labeled by each video filename and columns for each class (ie. species). The default prediction will store all class probabilities, so that cell (i,j) can be interpreted as *the probability that animal j is present in video i.* 

Predictions will be saved to `{model name}_{current timestamp}_preds.csv`. For example, running `zamba predict` on 9/15/2021 with the `time_distributed` model (the default) will save predictions to `time_distributed_2021-09-15_preds.csv`. 

## Step-by-step tutorial

The steps below walk through how to use `zamba predict` in more detail.

### 1. Make sure your videos are in the right format

- Your videos should all be a type that ffmpeg can recognize (link to ffmpeg filetype options)
- they should all be in one folder, and that folder shouldn't have any other files in it. borrow language from quickstart
- Make sure you know the path to your videos from where you are
- You'll add this to the CLI with `--data-dir <path-to-your-data>`.

If your videos are in a subfolder of your current working directory called `vids_to_classify`, so far the command is:
```console
$ zamba predict --data-dir vids_to_classify/
```

### 2. Choose a model for prediction

- central or west africa - slowfast if priority is blank v non blank, timedistributed if priority is species classification
- europe - european
- you'll add this to the CLI with `--model <your model>`
- The default is the time distributed model
- for more details see available models page

If we want to use the `slowfast` model to generate predictions for the videos in `vids_to_classify`:
```console
$ zamba predict --data-dir vids_to_classify/ --model slowfast
```

### 3. Choose the output type you want

There are three options for output type:
1. cell for the probability of each species in each video (default)
2. only the name of the most likely class for each video. specified with `--output-class-names`
3. For each video, a prediction (0 or 1) of whether each species is present in that video. This is specified by passing in a probability threshold that will be used to convert probability of detection to a True or False prediction, ie `--proba-threshold 0.5`. 1 means probability is greater than the threshold, 0 means less than or equal to.

Say we want to use the `slowfast` model to generate predictions for the videos in `vids_to_classify`, and we want to use a probability threshold of 0.5:
```console
$ zamba predict --data-dir vids_to_classify/ --model slowfast --proba-threshold 0.5
```

### 4. Specify any additional parameters

And there's so much more! You can also do things like specify your region for faster model download, use a saved model checkpoint, or run only one batch for faster debugging. See the Optional arguments section for all of the possibilities for customizing your predictions.

## Optional arguments

Each model that ships with the `zamba` package comes with a default configuration saved as a YAML file. The default value of any non-specified parameter will be set based on the model being used - `time_distributed`, `slowfast`, or `european`. Default algorithm configurations can be found in `models/config`.
<!-- TODO: update path to default configs and add link to github folder><!-->

There are two ways to override the default model parameter value, listed in order of precendence:

1. By passing an optional flag directly to the command line
2. By specifying the parameter in a custom YAML configuration file, and passing the YAML filepath to the command line with the `--config` flag. For more details on YAML configuration file options, see the All Configuration Options section. 

If a parameter is both passed directly to the CLI *and* specified in a YAMl file that is passed to the CLI, the value that is passed directly to the CLI will be used.
