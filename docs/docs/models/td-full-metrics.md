# Time-distributed model performance


## African forest model

The African species time-distributed model was trained using almost **250,000 videos from 14 countries** in West, Central, and East Africa.
These videos include examples of 30 animal species, plus some blank videos and some showing humans.
To evaluate the performance of the model, we held out 30,324 videos from 101 randomly-chosen sites.


### Removing blank videos

One use of this model is to identify blank videos so they can be discarded or ignored.
In this dataset, 42% of the videos are blank, so removing them can save substantial amounts of
viewing time and storage space.

The model assigns a probability that each video is blank, so one strategy is to remove videos 
if their probability exceeds a given threshold.
Of course, the model is not perfect, so there is a chance we will wrongly remove a video that actually
contains an animal.

To assess this tradeoff, we can use the holdout set to simulate this strategy with a range of thresholds.
For each threshold, we compute the fraction of blank videos correctly discarded and the fraction of non-blank
videos incorrectly discarded.
The following figure shows the results.

<img src="https://s3.amazonaws.com/drivendata-public-assets/zamba/td_full_set_recall_recall_curve.png" alt="" style="width:600px;"/>

The markers indicate three levels of tolerance for losing non-blank videos. For example, if it's acceptable to
lose 5% of non-blank videos, we can choose a threshold that removes 63% of the blank videos.
If we can tolerate a loss of 10%, we can remove 80% of the blanks.
And if we can tolerate a loss of 15%, we can remove 90% of the blanks.
Above that, the percentage of lost videos increases steeply.


### Accuracy

In addition to identifying blank videos, the model also computes a probability that each of 30 animal species appears in each video (plus human and blank).
We can use these probabilities to quantify the accuracy of the model for species classification.
Specifically, we computed:

* Top-1 accuracy, which is the fraction of videos where the species with the highest predicted probability is, in fact, present.

* Top-3 accuracy, which is the fraction of videos where one of the three species the model considered most likely is present.

Over all videos in the holdout set, the **top-1 accuracy is 82%; the top-3 accuracy is 94%**.
As an example, if you choose a video at random and the species with the highest predicted probability is elephant, the probability is 82% that the video contains an elephant, according to the human-generated labels.
If the three most likely species were elephant, hippopotamus, and cattle, the probability is 94% that the video
contains at least one of those species.

These results depend in part on the species represented in a particular dataset. For example, in the small
number of videos from Equatorial Guinea, only three species appear. For these videos, the top-1 accuracy is 97%, much
higher than the overall accuracy.
In the videos from Ivory Coast, 21 species are represented, so the problem is harder. For these
videos, top-1 accuracy is 80%, a little lower than the overall accuracy.


### Recall and precision by species

One of the goals of classification is to help with retrieval, that is, efficiently finding videos containing
a particular species. To evaluate the performance of the model for retrieval, we can use

* Recall, which is the fraction of videos containing a particular species that are correctly classified, and

* Precision, which is the fraction of videos the model labels with a particular species that actually contain that species.

The following figure shows recall and precision for the species in the holdout set, 
excluding 11 species where there are too few examples in the holdout set to compute meaningful estimates of these metrics.

<img src="https://s3.amazonaws.com/drivendata-public-assets/zamba/td_full_set_precision_recall_by_species.png" alt="" style="width:800px;"/>

It's clear that we are able to retrieve some species more efficiently than others. For example, elephants are relatively
easy to find. Of the videos that contain elephants, 84% are correctly classified; and of the videos that the model
labels "elephant", 94% contain elephants.
So a researcher using the model to find elephant videos could find a large majority of them while viewing only a
small number of non-elephant videos.

Not surprisingly, smaller animals are harder to find. For example, the recall for rodent videos is only 22%.
However, it is still possible to search for rodents by selecting videos that assign a relatively high probability
to "rodent", even if it assigns a higher probability to another species or "blank".


### Description of the holdout set

The videos in the holdout set are a random sample from the complete set of labeled videos, but they are
selected on a transect-by-transect basis; that is, videos from each transect are assigned entirely to the
training set or entirely to the holdout set.
So the performance of the model on the holdout set should reflect its performance on videos from a transect
the model has never seen.

All 14 countries are represented in the holdout set; the following table shows the number of videos from
each country.
These proportions are roughly consistent with the proportions in the complete set.

<table>
  <thead>
    <tr>
      <th>Country</th>
      <th>Number of videos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ivory Coast</th>
      <td>10,987</td>
    </tr>
    <tr>
      <th>Guinea</th>
      <td>4,300</td>
    </tr>
    <tr>
      <th>DR Congo</th>
      <td>2,750</td>
    </tr>
    <tr>
      <th>Uganda</th>
      <td>2,497</td>
    </tr>
    <tr>
      <th>Tanzania</th>
      <td>1,794</td>
    </tr>
    <tr>
      <th>Mozambique</th>
      <td>1,168</td>
    </tr>
    <tr>
      <th>Senegal</th>
      <td>1,131</td>
    </tr>
    <tr>
      <th>Gabon</th>
      <td>1,116</td>
    </tr>
    <tr>
      <th>Cameroon</th>
      <td>1,114</td>
    </tr>
    <tr>
      <th>Liberia</th>
      <td>1,065</td>
    </tr>
    <tr>
      <th>Central African Republic</th>
      <td>997</td>
    </tr>
    <tr>
      <th>Nigeria</th>
      <td>889</td>
    </tr>
    <tr>
      <th>Congo Republic</th>
      <td>678</td>
    </tr>
    <tr>
      <th>Equatorial Guinea</th>
      <td>38</td>
    </tr>
  </tbody>
</table>

The following figure shows the number of videos containing each of 30 animal species, plus some videos showing humans and
a substantial number of blank videos.

<img src="https://s3.amazonaws.com/drivendata-public-assets/zamba/td_full_set_number_videos_by_species.png" alt="" style="width:800px;"/>

One of the challenges of this kind of classification is that some species are 
much more common than others. For species that appear in a small number of videos, we expect
the model to be less accurate because it has fewer examples to learn from.
Also, for these species it is hard to assess performance precisely because there are few examples in the holdout set. If you would like to add more examples of the species you work with, see [how to build on the model](../../train-tutorial/).

