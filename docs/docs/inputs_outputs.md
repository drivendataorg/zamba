# Input Videos and Output Species

This section covers what data you can pass to zamba and what the output looks like.

## What species can `zamba` detect?

Using data collected and annotated by partners at [The Max Planck Institute for
Evolutionary Anthropology](https://www.eva.mpg.de/index.html) and [Chimp &
See](https://www.chimpandsee.org/), the prepackaged model in `zamba` was
trained to detect 23 species of animal––as well as `blank` videos (no animal
present).

The possible class labels in `zamba`'s default model are:

* `bird`
* `blank`
* `cattle`
* `chimpanzee`
* `duiker`
* `elephant`
* `forest buffalo`
* `gorilla`
* `hippopotamus`
* `hog`
* `human`
* `hyena`
* `large ungulate`
* `leopard`
* `lion`
* `other (non-primate)`
* `other (primate)`
* `pangolin`
* `porcupine`
* `reptile`
* `rodent`
* `small antelope`
* `small cat`
* `wild dog`

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
blank1.mp4
blank2.mp4
eleph.mp4
small-cat.mp4
ungulate.mp4
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
final output, however we assume that most users will want to save their
predictions for later manipulation and save those as well.

The top lines in the above code
block show where
`zamba` will look for videos as well as where the output, which by default is
called `output.csv` will be saved (the current working directory unless
otherwise specified). The lines that _don't_ begin with a
filename–`nasnet_mobile`, `inception_v3`, `xception_avg`, `inception_v2_resnet`–tell the
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
filename,bird,blank,cattle,chimpanzee,elephant,forest buffalo,gorilla,hippopotamus,human,hyena,large ungulate,leopard,lion,other (non-primate),other (primate),pangolin,porcupine,reptile,rodent,small antelope,small cat,wild dog,duiker,hog
blank2.mp4,3.0959607e-05,0.9982504,5.9925746e-08,1.8803144e-05,1.0665249e-06,3.0588583e-07,3.417223e-08,3.069252e-07,7.8558594e-05,3.319698e-07,5.2646147e-08,7.802722e-08,6.280087e-07,0.00053427694,7.304648e-05,2.684126e-07,2.6269622e-06,4.3846651e-07,0.00042926858,3.3577365e-07,2.7835036e-08,3.663065e-07,0.00028744357,5.9825194e-05
eleph.mp4,2.491223e-06,0.00016947306,1.57677e-08,8.856079e-06,0.9995918,1.278719e-07,1.2880488e-08,1.7235436e-08,1.8497045e-05,2.9417725e-08,1.6714116e-08,5.1064277e-09,5.5012507e-08,3.2303024e-07,5.6801964e-06,3.5061902e-08,1.2641495e-08,1.6470703e-07,3.107079e-06,6.0652815e-07,4.9092326e-09,2.6050692e-08,1.5590524e-05,8.5229095e-07
blank1.mp4,0.0018607612,0.97418517,3.4336247e-06,0.0009265547,1.7499387e-05,5.1025563e-06,3.2149017e-06,6.9392504e-06,0.01296743,6.278733e-06,2.521268e-06,3.495157e-06,7.778275e-06,0.00030972905,0.0055800937,5.494129e-06,2.3115346e-05,8.976383e-06,0.0005907763,7.1577356e-06,1.4174166e-06,6.2473964e-06,0.0046846806,0.00024899974
ungulate.mp4,4.071459e-07,0.022433572,0.0006694263,7.227396e-08,6.519187e-06,0.00029056694,0.0002747548,1.1176678e-05,2.8578936e-05,9.451887e-05,0.979065,4.4600056e-05,0.00019922768,1.5328165e-05,1.5081801e-05,6.5086033e-06,5.8271795e-08,4.4933295e-05,1.8390525e-05,6.327724e-05,2.1481837e-05,0.00017757648,5.099549e-05,2.075789e-05
small-cat.mp4,9.654887e-05,0.022990959,0.000380558,4.5536704e-05,2.4624824e-06,0.0011595424,9.205705e-05,0.00029680363,0.0010002105,0.00039995275,0.002531375,2.0125433e-05,0.00038491114,0.003318401,7.687087e-05,0.0002511233,4.4272376e-05,0.00014293614,0.00020618754,0.0006781974,0.9729478,0.00035086274,0.008977796,0.0005116147
```

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
blank2.mp4                blank
eleph.mp4              elephant
blank1.mp4                blank
ungulate.mp4     large ungulate
small-cat.mp4         small cat
```

Great work, `zamba`!

