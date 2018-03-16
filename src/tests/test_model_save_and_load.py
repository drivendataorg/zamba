from pathlib import Path

import tensorflow as tf

# from src.tests.test_model_config import *

# # Get relevant paths
# project_src = test_config.project_src
# models_dir = test_config.models_dir
# model_name = test_config.model_name
#
# # Create new subdir for model_name's files
# model_subdir = test_config.model_subdir
# model_subdir.mkdir(exist_ok=True)
#
# # Store global step, which will be baked into metagraph file name
# global_step = test_config.global_step
#
# # Store names of input layer tensors and operation (eg predict) to restore
# input_names = test_config.input_names
# op_to_restore_name = test_config.op_to_restore_name


def test_create_and_save_model(input_names,
                               op_to_restore_name,
                               models_dir,
                               model_name,
                               global_step):

    # Prepare input placeholders
    w1 = tf.placeholder("float", name=input_names[0])
    w2 = tf.placeholder("float", name=input_names[1])
    b1 = tf.Variable(2.0, name=input_names[-1])

    # Define a test operation that we will restore
    w3 = tf.add(w1, w2)
    w4 = tf.multiply(w3, b1, name=op_to_restore_name)

    # Session to save
    with tf.Session() as sess:

        # Initialize the network
        sess.run(tf.global_variables_initializer())

        # Create a saver object which will save all the variables
        saver = tf.train.Saver()

        # Run the operation w4 by feeding input
        feed_dict = {w1: 4, w2: 8}

        # 24 is sum of (w1 + w2) * b1
        assert sess.run(w4, feed_dict) == 24

        # Now, save the graph
        saver.save(sess,
                   Path(models_dir, model_name, model_name),
                   global_step=global_step,
                   )


def test_load_and_predict(model_subdir,
                          model_name,
                          global_step,
                          input_names,
                          op_to_restore_name):

    # Session to load
    with tf.Session() as sess:

        # Load the metagraph to restore graph into current session
        metagraph_name = f"{model_name}-{global_step}.meta"
        metagraph_path = model_subdir / metagraph_name

        # Note that model must exist to be tested
        assert metagraph_path.exists()

        loader = tf.train.import_meta_graph(str(metagraph_path))

        # Load checkpoint to restore weights into graph
        loader.restore(sess,
                       tf.train.latest_checkpoint(str(model_subdir)))

        # Get reference to the session's default graph (our loaded model)
        graph = tf.get_default_graph()

        # Access placeholders to create new feed dict
        # NOTE: accessing graph requires knowing name of tensors as written
        w1 = graph.get_tensor_by_name(f"{input_names[0]}:0")
        w2 = graph.get_tensor_by_name(f"{input_names[1]}:0")
        new_feed_dict = {w1: 5, w2: 9}

        # Access the operation to run
        op_to_restore = graph.get_tensor_by_name(f"{op_to_restore_name}:0")

        # 28 is sum of (w1 + w2) * b1
        assert sess.run(op_to_restore, new_feed_dict) == 28
