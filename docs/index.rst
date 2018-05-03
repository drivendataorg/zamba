.. zamba documentation master file, created by
   sphinx-quickstart on Fri Mar 23 09:21:41 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to zamba's documentation!
=================================

*Zamba means "forest" in the Lingala language.*

Zamba is a command-line tool built in Python to automatically identify the
species seen in camera trap videos from sites in central Africa. Using the
combined input of various deep learning models, the tool makes predictions
for 23 common species in these videos (as well as blank, or, "no species
present").
For more information, see the documentation.

The ``zamba`` command will be the entry point for users (see example usage
below).


.. toctree::
   :maxdepth: 4
   :caption: Installation:

   install

.. toctree::
   :maxdepth: 4
   :caption: Getting Started:

   algorithms
   quickstart
   slowstart-cli
   slowstart-lib

.. toctree::
   :maxdepth: 4
   :caption: Contributing to zamba:

   contribute

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
