"""
All models in our tool inherit from this class
"""

from pathlib import Path
import tempfile


import numpy as np
import pandas as pd
import tensorflow as tf

from src.tests import conftest as test_config
input_names = ["w1", "w2", "bias"]
op_to_restore_name = "op_to_restore"

class Model(object):
    def __init__(self, modeldir):
        """Instantiate model object"""

        # Store model dir as path object
        self.modeldir = Path(modeldir)

        # Use modeldir to get metagraph path
        self.metagraph_path = self.get_metagraph_path()
        # self.checkpoint_path = self.get_checkpoint_path()



    def predict_proba(self, tmp):
        """
        Predict class probabilities
        """

        with tf.Session() as sess:

            # load metagraph
            loader = tf.train.import_meta_graph(str(self.metagraph_path))

            # load checkpoint
            loader.restore(
                           sess,
                           tf.train.latest_checkpoint(str(self.modeldir))
            )
            # access graph
            graph = tf.get_default_graph()

            # create feed dict by accessing input names
            w1 = graph.get_tensor_by_name(f"{input_names[0]}:0")
            w2 = graph.get_tensor_by_name(f"{input_names[1]}:0")
            feed_dict = {w1: 5, w2: 9}

            # Access the operation to run
            op_to_restore = graph.get_tensor_by_name(f"{op_to_restore_name}:0")

            # run operation
            predictions = sess.run(op_to_restore, feed_dict)

            return pd.DataFrame(dict(output=[predictions]))

    def fit(self):
        """Fit to data"""

        pass

    def get_metagraph_path(self):
        """
        If utils.load_model is used to instantiate object,
        this file is guaranteed to exist.
        """
        mgp = [f for f in self.modeldir.resolve().iterdir() if str(f.suffix)
               == ".meta"]
        return mgp[0]

    # def get_checkpoint_path(self):
    #     """
    #     If utils.load_model is used to instantiate object,
    #     this file is guaranteed to exist.
    #     """
    #     ckptp = [f for f in self.modeldir.resolve().iterdir() if str(f.stem)
    #              == "checkpoint"]
    #     return str(ckptp[0])
