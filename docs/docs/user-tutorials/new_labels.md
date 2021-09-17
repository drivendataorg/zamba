# I have labeled videos, and the labels are completely new to zamba. How do I train a model?

<!-- TODO><!-->

This tutorial goes over the steps for using `zamba` if you:
- Have labeled video data,
- Your species labels are NOT included in the list of possible [zamba labels](../models.md#species-classes), and
- Want to train or fine tune a model

- we'll use zamba train (link to CLI page) in the command line tool
- assumes you have already installed zamba (link to install page)

## 1. Make sure your videos are in the right format

- video format - see no labels
- add with data-dir

## 2. Make sure your labels are in the right format

- labels - columns for filepath and labels
- add path with labels

## 3. Choose your model

use either time_distributed or european, depending on your geography. we don't recommend slowfast bc it's so slow.

### 4. Input a list of species

TODO. can this only happen in a yaml file, not in CLI, with TrainConfig.species?

## 5. Specify any additional parameters

And there's so much more! You can also do things like specify your region for faster model download, run only one batch for faster debugging, or change where to cache your model weights. Check out the [Command Line Interface](../cli.md) page to see all of the possibilities for customizing your predictions.

## 6. Access your model

TODO


