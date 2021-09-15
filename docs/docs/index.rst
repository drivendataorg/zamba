.. zamba documentation master file, created by
   sphinx-quickstart on Wed Sep 15 2021.


Welcome to zamba's documentation!
=================================

.. raw:: html

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

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   install
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: How to Run zamba

   cli
   py-package

.. toctree::
   :maxdepth: 2
   :caption: Advanced Options

   configurations
   models


.. toctree::
   :maxdepth: 2
   :caption: Contributing to zamba

   contribute

.. toctree::
   :maxdepth: 4
   :caption: Zamba source documentation

   zamba

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`