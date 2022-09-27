# Time dependent model performance

## Species model

The model was trained using 249,269 videos 14 countries in West, Central, and East Africa.
To evaluate the performance of the model, we held out 30,324 videos from 689 randomly-chosen transects.
Importantly, all videos from each transect were either used for training or held out for evaluation.
So the performance of the model on the holdout set should reflect its performance on videos from a transect
the model has never seen.

### Description of the holdout set

All 14 countries are represented in the holdout set; the following table shows the number of videos from
each country.
These proportions are roughly consistent with the proportions in the training set.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: center;">
      <th>Country</th>
      <th>Number of videos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ivory Coast</th>
      <td>10987</td>
    </tr>
    <tr>
      <th>Guinea</th>
      <td>4300</td>
    </tr>
    <tr>
      <th>DR Congo</th>
      <td>2750</td>
    </tr>
    <tr>
      <th>Uganda</th>
      <td>2497</td>
    </tr>
    <tr>
      <th>Tanzania</th>
      <td>1794</td>
    </tr>
    <tr>
      <th>Mozambique</th>
      <td>1168</td>
    </tr>
    <tr>
      <th>Senegal</th>
      <td>1131</td>
    </tr>
    <tr>
      <th>Gabon</th>
      <td>1116</td>
    </tr>
    <tr>
      <th>Cameroon</th>
      <td>1114</td>
    </tr>
    <tr>
      <th>Liberia</th>
      <td>1065</td>
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


The training and holdout sets contain examples of 30 animal species, plus some videos showing humans and
a substantial number of blank videos.
The following figure shows the number of videos containing each species.

<img src="../../../docs/media/td_full_set_number_videos_by_species.png" alt="" style="width:800px;"/>

One of the challenges of this kind of classification is "class imbalance"; that is, some species are 
much more common than others. For some species, there are only a few examples in the holdout set.
For these species, we will not be able to assess the performance of the model precisely.

### Accuracy

One of the ways we'll evaluate the model is classification accuracy, which is the percentage of videos
that are classified correctly. Specifically, we computed:

* Top-1 accuracy, which is the fraction of videos where the species the model gave the highest probability was correct.

* Top-3 accuracy, which is the fraction of videos where one of the three species the model considered most likely was correct.

Over all videos in the holdout set, the Top-1 accuracy is 80%; the Top-3 accuracy is 94%.
As an example, if you chose a video at random from the holdout set, and the species with the highest probability is elephant,
the probability is 80% that the video contains an elephant, according to the human-generated labels.
If the three most likely species were elephant, hippopotamus, and cattle, the probability is 94% that the video
contains at least one of those species.

These results depend in part on the number of species represented in a particular dataset. For example, in the small
number of videos from Equatorial Guinea, only three species appear. For these videos, the Top-1 accuracy is 97%!
In the videos from Ivory Coast, 21 species are represented, so the problem is substantially harder. For these
videos, Top-1 accuracy is only 77%, but Top-3 accuracy is 93%.

### Recall and precision by species

One of the goals of classification is to help with retrieval, that is, efficiently finding videos containing
a particular species. To evaluate the performance of the model for retrieval, we can use

* Recall, which is the fraction of videos containing a particular species that are correctly classified, and

* Precision, which is the fraction of videos the model labels with a particular species that actually contain that species.

The following figure shows recall and precision for the species in the holdout set.

<img src="../../../docs/media/td_full_set_precision_recall_by_species.png" alt="" style="width:800px;"/>

It's clear that we are able to retrieve some species more efficiently than others. For example, elephants are relatively
easy to find. Of the videos that contain elephants, 78% are correctly classified; and of the videos that the model
labels "elephant", 96% contain elephants.
So a researcher using the model to find elephant videos could find a large majority of them while viewing only a
small number of non-elephant videos.

Not surprisingly, smaller animals tend to be harder to find. For example, the recall for rodent videos is only 6%.
However, it is still possible to search for rodents by selecting videos that assign a relatively high probability
to "rodent", even if it assigns a higher probability to another species or blank.

Recall is highest for blank videos, which indicates that we can rule out blank videos with high accuracy.
If the goal is to reduce time spent watching blank videos, we could use the classifications from the model to 
eliminate 94% of them, and 78% of the videos we discard would be blank. However, that means we would discard
22% of the videos that are not actually blank.
If that's not acceptable, we can use the probabilities generated by the model to trade off some recall for 
higher precision (fewer discarded non-blank videos).


### False blank by species

If we discard videos classified as blank, that has a bigger impact on some species than others.
We can quantify this with the false blank rate, which is the fraction of videos containing a particular
species that are falsely labeled blank.
The following figure shows these false blank rates.

<img src="../../../docs/media/td_full_set_false_blank_by_species.png" alt="" style="width:800px;"/>

Again, we see that small animals are harder to find. For bats, reptiles, rodents, and porcupines, the false
blank rate is greater than 50%, meaning that more than half of the videos containing these species would
be lost if we discard all videos the model labels blank.

The false blank rate is also higher for some rare species (in the sense that they appear in few videos),
probably because the model had too few examples to learn from. For example, there are only three giraffe
videos in the holdout set; the model misses two of them.

The results are better for species that are not too small and not too rare.


## `blank_nonblank` model




