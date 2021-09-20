# I have unlabeled videos to classify. What model should I use?
<!-- TODO><!-->

This tutorial goes over the steps for using `zamba` if you:
- Have unlabeled videos that you want to generate labels for
- The possible class species labels you want to use are included in the list of possible [zamba labels](../models.md#species-classes)

- we'll use zamba predict (link to CLI page) in the command line tool
- assumes you have already installed zamba (link to install page)

## 1. Make sure your videos are in the right format

- Your videos should all be a type that ffmpeg can recognize (link to ffmpeg filetype options)
- they should all be in one folder, and that folder shouldn't have any other files in it. borrow language from quickstart
- Make sure you know the path to your videos from where you are
- You'll add this to the CLI with `--data-dir <path-to-your-data>`.

If your videos are in a subfolder of your current working directory called `vids_to_classify`, so far the command is:
```console
$ zamba predict --data-dir vids_to_classify/
```

## 2. Choose a model for prediction

- central or west africa - slowfast if priority is blank v non blank, timedistributed if priority is species classification
- europe - european
- you'll add this to the CLI with `--model <your model>`
- The default is the time distributed model
- for more details see available models page

If we want to use the `slowfast` model to generate predictions for the videos in `vids_to_classify`:
```console
$ zamba predict --data-dir vids_to_classify/ --model slowfast
```

## 3. Choose the output type you want

There are three options for output type:
1. cell for the probability of each species in each video (default)
2. only the name of the most likely class for each video. specified with `--output-class-names`
3. For each video, a prediction (0 or 1) of whether each species is present in that video. This is specified by passing in a probability threshold that will be used to convert probability of detection to a True or False prediction, ie `--proba-threshold 0.5`. 1 means probability is greater than the threshold, 0 means less than or equal to.

Say we want to use the `slowfast` model to generate predictions for the videos in `vids_to_classify`, and we want to use a probability threshold of 0.5:
```console
$ zamba predict --data-dir vids_to_classify/ --model slowfast --proba-threshold 0.5
```

## 4. Specify any additional parameters

And there's so much more! You can also do things like specify your region for faster model download, use a saved model checkpoint, or run only one batch for faster debugging. Check out the [Command Line Interface](../cli.md) page to see all of the possibilities for customizing your predictions.

## 5. Access your predictions

Predictions will be saved to `<model name>_<current timestamp>_preds.csv`. For example, running `zamba predict` on 9/15/2021 with the slowfast model will save out predictions to `slowfast_2021-09-15_preds.csv`.

Below, we generate predictions and then use python from the command line to view our predictions as a dataframe. We could also see them in the command line with `cat slowfast_2021-09-15_preds.csv`

```console
$ zamba predict --data-dir vids_to_classify/ --model slowfast --proba-threshold 0.5
$ python
>>> import pandas as pd
>>> predictions = pd.read_csv(slowfast_2021-09-15_preds.csv).set_index('filepath')
>>> predictions
```

| filepath                     | aardvark | antelope_duiker | badger | bat | bird | blank | cattle | cheetah | chimpanzee_bonobo | civet_genet | elephant | equid | forest_buffalo | fox | giraffe | gorilla | hare_rabbit | hippopotamus | hog | human | hyena | large_flightless_bird | leopard | lion | mongoose | monkey_prosimian | pangolin | porcupine | reptile | rodent | small_cat | wild_dog_jackal |
| ---------------------------- | -------- | --------------- | ------ | --- | ---- | ----- | ------ | ------- | ----------------- | ----------- | -------- | ----- | -------------- | --- | ------- | ------- | ----------- | ------------ | --- | ----- | ----- | --------------------- | ------- | ---- | -------- | ---------------- | -------- | --------- | ------- | ------ | --------- | --------------- |
| vids_to_classify/blank.MP4   | 0        | 0               | 0      | 0   | 0    | 1     | 0      | 0       | 0                 | 0           | 0        | 0     | 0              | 0   | 0       | 0       | 0           | 0            | 0   | 0     | 0     | 0                     | 0       | 0    | 0        | 0                | 0        | 0         | 0       | 0      | 0         | 0               |
| vids_to_classify/chimp.MP4   | 0        | 0               | 0      | 0   | 0    | 0     | 0      | 0       | 1                 | 0           | 0        | 0     | 0              | 0   | 0       | 0       | 0           | 0            | 0   | 0     | 0     | 0                     | 0       | 0    | 0        | 0                | 0        | 0         | 0       | 0      | 0         | 0               |
| vids_to_classify/eleph.MP4   | 0        | 0               | 0      | 0   | 0    | 0     | 0      | 0       | 0                 | 0           | 1        | 0     | 0              | 0   | 0       | 0       | 0           | 0            | 0   | 0     | 0     | 0                     | 0       | 0    | 0        | 0                | 0        | 0         | 0       | 0      | 0         | 0               |
| vids_to_classify/leopard.MP4 | 0        | 0               | 0      | 0   | 0    | 0     | 0      | 0       | 0                 | 0           | 0        | 0     | 0              | 0   | 0       | 0       | 0           | 0            | 0   | 0     | 0     | 0                     | 1       | 0    | 0        | 0                | 0        | 0         | 0       | 0      | 0         | 0               |

```console
>>> predictions.idxmax(axis=1)
```

| filepath                     | Column with max value |
| ---------------------------- | --------------------- |
| vids_to_classify/blank.MP4   | blank                 |
| vids_to_classify/chimp.MP4   | chimpanzee_bonobo     |
| vids_to_classify/eleph.MP4   | elephant              |
| vids_to_classify/leopard.MP4 | leopard               |