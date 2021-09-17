Welcome to zamba's documentation!
=================================


<div class="embed-responsive embed-responsive-16by9" width=500>
    <iframe width=600 height=340 class="embed-responsive-item" src="https://s3.amazonaws.com/drivendata-public-assets/monkey-vid.mp4" frameborder="0" allowfullscreen=""></iframe>
</div>

*Zamba means "forest" in the Lingala language.*


Zamba is a tool built in Python to automatically identify the species seen
in camera trap videos from sites in Africa and Europe. Using the combined
input of various deep learning models, the tool makes predictions for 42
common species in these videos (as well as blank, or, "no species present").
Zamba can be accessed as both a command-line tool and a Python package.

Zamba ships with three model options. `time_distributed` and `slowfast` are 
trained on 31 common species from central and west Africa. `european` is trained 
on 11 common species from western Europe.

### Getting Started
- [Quickstart](quickstart.md)
- [Installing Zamba](install.md)

### How to Run Zamba
- [Command Line Interface](cli.md)
- [Python Package](py-package.md)

### [Choosing a Model](models.md)

### [Model Configuration](configurations.md)

### User Tutorials
- [I have no labels]("no_labels.md")
- [I have zamba labels]("subset_labels.md")
- [I have new labels]("new_labels.md")

### [Contributing](contribute.md)

### Changelog
- [Version 2](v2_updates.md)

Indices and tables
==================
<!-- TODO: what is this supposed to do? fix><!-->

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`