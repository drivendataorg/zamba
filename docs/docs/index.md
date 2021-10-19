Welcome to zamba's documentation!
=================================


<div class="embed-responsive embed-responsive-16by9" width=500>
    <iframe width=600 height=340 class="embed-responsive-item" src="https://s3.amazonaws.com/drivendata-public-assets/monkey-vid.mp4" frameborder="0" allowfullscreen=""></iframe>
</div>

*Zamba means "forest" in the Lingala language.*

Zamba is a tool built in Python to automatically detect and classify the species seen in camera trap videos. Using state-of-the-art computer vision and machine learning, the tool is trained to identify 42 common species from sites in Africa and Europe (as well as blank, or "no species present"). Users can also input their own labeled videos to finetune a model and make predictions for new species or new contexts. `zamba` can be accessed as both a command-line tool and a Python package.

Zamba ships with three model options. `time_distributed` and `slowfast` are
trained on 32 common species from central and west Africa. `european` is trained
on 11 common species from western Europe. `time_distributed` and `european` are image-based models while `slowfast` is a video-based model.

### Getting Started
- [Installing Zamba](install.md)
- [Quickstart](quickstart.md)

### User Tutorials
- [Classifying Unlabeled Videos](predict-tutorial.md)
- [Training a Model on Labeled Videos](train-tutorial.md)
- [Debugging](debugging.md)

### [Available Models](models.md)

### Advanced Options
- [All Optional Arguments](configurations.md)
- [Using YAML Configuration Files](yaml-config.md)

### [Contributing](contribute.md)

### Changelog
- [Version 2](v2_updates.md)
