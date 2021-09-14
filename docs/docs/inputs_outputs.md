# Input Videos and Output Species

This section covers what data you can pass to zamba and what the output looks like.

## What species can `zamba` detect?

Using data collected and annotated by partners at [The Max Planck Institute for
Evolutionary Anthropology](https://www.eva.mpg.de/index.html) and [Chimp &
See](https://www.chimpandsee.org/), the prepackaged model in `zamba` was
trained to detect 31 species of animal––as well as `blank` videos (no animal
present).

The possible class labels in `zamba`'s default model are:

* `aardvark`
* `antelope_duiker`
* `badger`
* `bat`
* `bird`
* `blank`
* `cattle`
* `cheetah`
* `chimpanzee_bonobo`
* `civet_genet`
* `elephant`
* `equid`
* `forest_buffalo`
* `fox`
* `giraffe`
* `gorilla`
* `hare_rabbit`
* `hippopotamus`
* `hog`
* `human`
* `hyena`
* `large_flightless_bird`
* `leopard`
* `lion`
* `mongoose`
* `monkey_prosimian`
* `pangolin`
* `porcupine`
* `reptile`
* `rodent`
* `small_cat`
* `wild_dog_jackal`

*New in Zamba v2:* Thanks to new data shared by [<SOURCE>], `zamba` now also includes a prepackaged model to identify 11 common European species. The class labels in `zamba`'s European model are:

* `bird`
* `blank`
* `domestic_cat`
* `european_badger`
* `european_beaver`
* `european_hare`
* `european_roe_deer`
* `north_american_raccoon`
* `red_fox`
* `unidentified`
* `weasel`
* `wild_boar`

## What Videos Can I use?

The `zamba` models were trained using 15 second `.mp4` videos. Although videos
longer than 15 seconds will process without error, we recommend segmenting your videos into 15 second segments for the most accurate results since videos are downsampled to a smaller number of frames during processing. This can be done with the `ffmpeg` commandline tool. Here's an example:

```console
$ ffmpeg -i input_video.mp4 -c copy -map 0 -segment_time 15 -f segment -reset_timestamps 1 output_video%%03d.mp4
```

`zamba` supports the same formats as FFMPEG, [which are listed here](https://www.ffmpeg.org/general.html#Supported-File-Formats_002c-Codecs-or-Features).

Videos do not need to be resized to a particular resolution since `zamba`
resizes every frame it uses to `404 x 720` pixels.

### Know Where Your Videos Are

Suppose you have `zamba` installed, your command line is open, and you have a
directory of videos, `vids_to_classify/`, that you want to classify using
`zamba`.

**The folder must contain only valid video files since zamba will try to load all of the files in the directory.**

List the videos:

```console
$ ls vids_to_classify/
blank.mp4
chimp.mp4
eleph.mp4
leopard.mp4
```


### Predict Using Default Settings

Using `zamba` to produce predictions for these videos is as easy as:

```console
$ zamba predict vids_to_classify/
```

**NOTE: `zamba` needs to download the "weights" files for the neural networks
that it uses to make predictions. On first run it will download ~1GB of files
with these weights.** Once these are downloaded, the tool will use the local
versions and will not need to perform this download again.

`zamba` will begin prediction on the videos contained in the _top level_ of the
`vids_to_classify` directory (`zamba` does not currently extract videos from
nested directories):

```console
$ zamba predict vids_to_classify/
Using data_path:    vids_to_classify/
Using pred_path:    output.csv
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

You will see command line output during the prediction process, including the
final output. However, we assume that most users will want to save their
predictions for later manipulation rather than have them directly returned 
in the command line.

The top lines in the above code block show where
`zamba` will look for videos as well as where the output, which by default is
called `output.csv` will be saved (the current working directory unless
otherwise specified). The lines that _don't_ begin with a
filename – `nasnet_mobile`, `inception_v3`, `xception_avg`, `inception_v2_resnet` – tell the
user which deep learning model `zamba` is currently using to predict. The final
prediction combines predictions from these models. 

`zamba` will output a `.csv` file with rows
 labeled by each video filename and columns for each class. The default
 prediction will store all
class probabilities, so that cell (i,j) of `output.csv` can be interpreted as
_the probability that animal j is present in video i_.

In this example `output.csv` looks like:

```console
$ head output.csv
filename,aardvark,antelope_duiker,badger,bat,bird,blank,bonobo,cattle,cheetah,chimpanzee,civet_genet,elephant,equid,forest_buffalo,fox,giraffe,gorilla,hare_rabbit,hippopotamus,hog,human,hyena,large_flightless_bird,leopard,lion,mongoose,monkey_or_prosimian,pangolin,porcupine,reptile,rodent,small_cat,wild_dog_jackal
blank.mp4,2.567129e-28,4.3890892e-16,0.0,5.788882000000001e-23,4.2099122e-17,1.0,5.8690688e-36,3.928397e-25,0.0,6.3825355e-22,7.091818e-24,6.206757e-21,2.6836583000000003e-26,9.0656394e-21,0.0,0.0,4.582915300000001e-22,2.685744e-36,1.6775296e-23,1.0984092e-18,1.2292842e-23,1.6366164e-28,1.9885625000000002e-38,1.0181542e-35,0.0,6.3937274e-20,3.4324745e-20,5.4223364e-19,1.01906944e-16,1.7929117e-27,1.8259759e-17,1.7104254000000002e-32,4.81265e-36
chimp.mp4,1.099172e-18,2.176893e-08,7.683882e-07,5.0155355e-17,2.5321875e-10,6.9823985e-10,1.43334416e-11,1.0024781e-09,1.8925349e-14,0.9579364,1.5059592e-14,5.1087654e-09,3.0653349e-12,6.0415e-08,1.9665776e-14,9.061421e-10,3.2564805e-11,1.8022606e-13,1.8314252e-12,1.6785786e-07,0.01889301,1.6086219e-08,6.718011500000001e-16,7.1360104e-09,5.292697e-16,7.3422523e-13,1.3567616e-09,3.985831e-11,2.298666e-11,6.767371300000001e-16,1.7461181e-07,1.4106338e-13,3.8742626e-14
eleph.mp4,4.1982657e-26,1.9667803e-14,1.0499261e-18,1.5674166000000002e-22,1.8034734e-14,1.2275667e-12,1.3773675000000001e-17,1.4526822e-14,7.918141e-23,1.7045772e-16,3.487233e-20,1.0,2.3539393e-23,2.471267e-09,3.910887e-33,1.7822059000000002e-27,1.808321e-24,0.0,5.408536e-08,5.1874727e-10,1.5758247e-11,3.237222e-25,5.5110828000000005e-21,4.4498553e-20,2.0261061e-30,2.3406181e-36,2.6329097e-12,7.182644e-24,2.1643852e-12,3.2944662999999996e-25,4.281294e-09,3.365868e-19,2.5523188e-26
leopard.mp4,3.271403e-37,1.0383028e-20,3.9352542000000004e-16,8.520898e-36,6.3707895e-10,3.1425548e-20,2.3208635e-20,2.2527576999999997e-18,2.6524485000000003e-22,1.3015485e-09,5.5073839999999994e-36,4.2164662e-26,1.05943414e-16,1.8422287e-32,2.4193592e-23,0.99108803,6.6583634e-23,1.9471226e-20,4.5047463e-36,1.4285896e-11,3.5426372e-19,7.108155e-15,1.1524454e-33,1.0,1.6516367e-29,1.7245645e-24,2.618596e-16,1.9737813e-29,6.043798e-18,7.9284636e-19,3.7582455e-14,5.689167e-22,9.124682e-23
```

Or, as a [`pd.DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html):

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>filename</th>
      <th>aardvark</th>
      <th>antelope_duiker</th>
      <th>badger</th>
      <th>bat</th>
      <th>bird</th>
      <th>blank</th>
      <th>bonobo</th>
      <th>cattle</th>
      <th>cheetah</th>
      <th>chimpanzee</th>
      <th>civet_genet</th>
      <th>elephant</th>
      <th>equid</th>
      <th>forest_buffalo</th>
      <th>fox</th>
      <th>giraffe</th>
      <th>gorilla</th>
      <th>hare_rabbit</th>
      <th>hippopotamus</th>
      <th>hog</th>
      <th>human</th>
      <th>hyena</th>
      <th>large_flightless_bird</th>
      <th>leopard</th>
      <th>lion</th>
      <th>mongoose</th>
      <th>monkey_or_prosimian</th>
      <th>pangolin</th>
      <th>porcupine</th>
      <th>reptile</th>
      <th>rodent</th>
      <th>small_cat</th>
      <th>wild_dog_jackal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>blank.mp4</th>
      <td>2.567129e-28</td>
      <td>4.389089e-16</td>
      <td>0.000000e+00</td>
      <td>5.788882e-23</td>
      <td>4.209912e-17</td>
      <td>1.000000e+00</td>
      <td>5.869069e-36</td>
      <td>3.928397e-25</td>
      <td>0.000000e+00</td>
      <td>6.382536e-22</td>
      <td>7.091818e-24</td>
      <td>6.206757e-21</td>
      <td>2.683658e-26</td>
      <td>9.065639e-21</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>4.582915e-22</td>
      <td>2.685744e-36</td>
      <td>1.677530e-23</td>
      <td>1.098409e-18</td>
      <td>1.229284e-23</td>
      <td>1.636616e-28</td>
      <td>1.988563e-38</td>
      <td>1.018154e-35</td>
      <td>0.000000e+00</td>
      <td>6.393727e-20</td>
      <td>3.432474e-20</td>
      <td>5.422336e-19</td>
      <td>1.019069e-16</td>
      <td>1.792912e-27</td>
      <td>1.825976e-17</td>
      <td>1.710425e-32</td>
      <td>4.812650e-36</td>
    </tr>
    <tr>
      <th>chimp.mp4</th>
      <td>1.099172e-18</td>
      <td>2.176893e-08</td>
      <td>7.683882e-07</td>
      <td>5.015536e-17</td>
      <td>2.532187e-10</td>
      <td>6.982399e-10</td>
      <td>1.433344e-11</td>
      <td>1.002478e-09</td>
      <td>1.892535e-14</td>
      <td>9.579364e-01</td>
      <td>1.505959e-14</td>
      <td>5.108765e-09</td>
      <td>3.065335e-12</td>
      <td>6.041500e-08</td>
      <td>1.966578e-14</td>
      <td>9.061421e-10</td>
      <td>3.256481e-11</td>
      <td>1.802261e-13</td>
      <td>1.831425e-12</td>
      <td>1.678579e-07</td>
      <td>1.889301e-02</td>
      <td>1.608622e-08</td>
      <td>6.718012e-16</td>
      <td>7.136010e-09</td>
      <td>5.292697e-16</td>
      <td>7.342252e-13</td>
      <td>1.356762e-09</td>
      <td>3.985831e-11</td>
      <td>2.298666e-11</td>
      <td>6.767371e-16</td>
      <td>1.746118e-07</td>
      <td>1.410634e-13</td>
      <td>3.874263e-14</td>
    </tr>
    <tr>
      <th>eleph.mp4</th>
      <td>4.198266e-26</td>
      <td>1.966780e-14</td>
      <td>1.049926e-18</td>
      <td>1.567417e-22</td>
      <td>1.803473e-14</td>
      <td>1.227567e-12</td>
      <td>1.377368e-17</td>
      <td>1.452682e-14</td>
      <td>7.918141e-23</td>
      <td>1.704577e-16</td>
      <td>3.487233e-20</td>
      <td>1.000000e+00</td>
      <td>2.353939e-23</td>
      <td>2.471267e-09</td>
      <td>3.910887e-33</td>
      <td>1.782206e-27</td>
      <td>1.808321e-24</td>
      <td>0.000000e+00</td>
      <td>5.408536e-08</td>
      <td>5.187473e-10</td>
      <td>1.575825e-11</td>
      <td>3.237222e-25</td>
      <td>5.511083e-21</td>
      <td>4.449855e-20</td>
      <td>2.026106e-30</td>
      <td>2.340618e-36</td>
      <td>2.632910e-12</td>
      <td>7.182644e-24</td>
      <td>2.164385e-12</td>
      <td>3.294466e-25</td>
      <td>4.281294e-09</td>
      <td>3.365868e-19</td>
      <td>2.552319e-26</td>
    </tr>
    <tr>
      <th>leopard.mp4</th>
      <td>3.271403e-37</td>
      <td>1.038303e-20</td>
      <td>3.935254e-16</td>
      <td>8.520898e-36</td>
      <td>6.370790e-10</td>
      <td>3.142555e-20</td>
      <td>2.320864e-20</td>
      <td>2.252758e-18</td>
      <td>2.652449e-22</td>
      <td>1.301549e-09</td>
      <td>5.507384e-36</td>
      <td>4.216466e-26</td>
      <td>1.059434e-16</td>
      <td>1.842229e-32</td>
      <td>2.419359e-23</td>
      <td>9.910880e-01</td>
      <td>6.658363e-23</td>
      <td>1.947123e-20</td>
      <td>4.504746e-36</td>
      <td>1.428590e-11</td>
      <td>3.542637e-19</td>
      <td>7.108155e-15</td>
      <td>1.152445e-33</td>
      <td>1.000000e+00</td>
      <td>1.651637e-29</td>
      <td>1.724564e-24</td>
      <td>2.618596e-16</td>
      <td>1.973781e-29</td>
      <td>6.043798e-18</td>
      <td>7.928464e-19</td>
      <td>3.758246e-14</td>
      <td>5.689167e-22</td>
      <td>9.124682e-23</td>
    </tr>
  </tbody>
</table>

Although it is hard to read this output, `zamba` has done a great job (compare
the file names to the largest probabilities in each row). To see how `zamba`
can be used to generate an immediately interpretable output format, continue
below!

### Predict Using Concise Output Format

The `.csv` format is great for analysis where the user may want to consider
various thresholds of probability as valid. However, `zamba` is capable of
more concise output as well.

If you just want to know the most likely animal in each video, the
`--output_class_names` flag is useful. In this case the output during
prediction is the same, but the final output as well as the resulting `output.csv`
are simplified to show the _most probable_ animal in each video:

```console
$ zamba predict vids_to_classify/ --output_class_names
...
blank.mp4                 blank
chimp.mp4     chimpanzee_bonobo
eleph.mp4              elephant
leopard.mp4             leopard
```

Great work, `zamba`!

