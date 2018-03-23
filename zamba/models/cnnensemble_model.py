from pathlib import Path

from .model import Model
from .cnnensemble.src.single_frame_cnn import generate_prediction_test

class CnnEnsemble(Model):
    def __init__(self, model_path, tempdir=None):
        # use the model object's defaults
        super().__init__(model_path, tempdir=tempdir)

    def load_data(self, data_path):
        """ Loads data and returns it in a format that can be used
            by this model.

        Args:
            data_path: A path to the input dat

        Returns:
            The data.
        """
        pass

    def predict(self, X):
        """Predict class probabilities for each input, X

        Args:
            X: input data

        Returns:

        """
        # python3.6 single_frame_cnn.py generate_prediction_test --model resnet50_avg --fold 1 --weights ../output/checkpoints/resnet50_avg_fold_1/checkpoint-007-0.0480.hdf5
        generate_prediction_test('resnet50_avg',
                                 (Path(__file__).parent / 'cnnensemble' /
                                  'output/checkpoints/resnet50_avg_fold_1/checkpoint-007-0.0480.hdf5'),
                                 1)

        # python3.6 single_frame_cnn.py generate_prediction_test --model resnet50 --fold 2 --weights ../output/checkpoints/resnet50_fold_2/checkpoint-016-0.0342.hdf5
        # python3.6 single_frame_cnn.py generate_prediction_test --model resnet50_avg --fold 3 --weights ../output/checkpoints/resnet50_avg_fold_3/checkpoint-004-0.0813.hdf5
        # python3.6 single_frame_cnn.py generate_prediction_test --model resnet50_avg --fold 4 --weights ../output/checkpoints/resnet50_avg_fold_4/checkpoint-015-0.0352.hdf5

        # python3.6 single_frame_cnn.py generate_prediction_test --model xception_avg --fold 1 --weights ../output/checkpoints/xception_avg_fold_1/checkpoint-004-0.1295.hdf5
        # python3.6 single_frame_cnn.py generate_prediction_test --model xception_avg --fold 2 --weights ../output/checkpoints/xception_avg_fold_2/checkpoint-005-0.0308.hdf5
        # python3.6 single_frame_cnn.py generate_prediction_test --model xception_avg --fold 3 --weights ../output/checkpoints/xception_avg_fold_3/checkpoint-004*
        # python3.6 single_frame_cnn.py generate_prediction_test --model xception_avg --fold 4 --weights ../output/checkpoints/xception_avg_fold_4/checkpoint-004-0.1665.hdf5

        # python3.6 single_frame_cnn.py generate_prediction_test --model xception_avg_ch10 --fold 1 --weights ../output/checkpoints/xception_avg_fold_1/checkpoint-009*
        # python3.6 single_frame_cnn.py generate_prediction_test --model xception_avg_ch10 --fold 2 --weights ../output/checkpoints/xception_avg_fold_2/checkpoint-010*
        # python3.6 single_frame_cnn.py generate_prediction_test --model xception_avg_ch10 --fold 3 --weights ../output/checkpoints/xception_avg_fold_3/checkpoint-009*
        # python3.6 single_frame_cnn.py generate_prediction_test --model xception_avg_ch10 --fold 4 --weights ../output/checkpoints/xception_avg_fold_4/checkpoint-009*

        # python3.6 single_frame_cnn.py generate_prediction_test --model inception_v3 --fold 1 --weights ../output/checkpoints/inception_v3_fold_1/checkpoint-009*
        # python3.6 single_frame_cnn.py generate_prediction_test --model inception_v3 --fold 2 --weights ../output/checkpoints/inception_v3_fold_2/checkpoint-009*
        # python3.6 single_frame_cnn.py generate_prediction_test --model inception_v3 --fold 3 --weights ../output/checkpoints/inception_v3_fold_3/checkpoint-009*
        # python3.6 single_frame_cnn.py generate_prediction_test --model inception_v3 --fold 4 --weights ../output/checkpoints/inception_v3_fold_4/checkpoint-009*

        # python3.6 single_frame_cnn.py generate_prediction_test --model inception_v2_resnet --fold 1 --weights ../output/checkpoints/inception_v2_resnet_fold_1/checkpoint-005-0.0347.hdf5
        # python3.6 single_frame_cnn.py generate_prediction_test --model inception_v2_resnet --fold 2 --weights ../output/checkpoints/inception_v2_resnet_fold_2/checkpoint-005-0.1555.hdf5
        # python3.6 single_frame_cnn.py generate_prediction_test --model inception_v2_resnet --fold 3 --weights ../output/checkpoints/inception_v2_resnet_fold_3/checkpoint-005-0.0342.hdf5
        # python3.6 single_frame_cnn.py generate_prediction_test --model inception_v2_resnet --fold 4 --weights ../output/checkpoints/inception_v2_resnet_fold_4/checkpoint-005-0.1457.hdf5

        # python3.6 single_frame_cnn.py generate_prediction_test --model inception_v2_resnet_ch10 --fold 1 --weights ../output/checkpoints/inception_v2_resnet_fold_1/checkpoint-011*
        # python3.6 single_frame_cnn.py generate_prediction_test --model inception_v2_resnet_ch10 --fold 2 --weights ../output/checkpoints/inception_v2_resnet_fold_2/checkpoint-011*
        # python3.6 single_frame_cnn.py generate_prediction_test --model inception_v2_resnet_ch10 --fold 3 --weights ../output/checkpoints/inception_v2_resnet_fold_3/checkpoint-009*
        # python3.6 single_frame_cnn.py generate_prediction_test --model inception_v2_resnet_ch10 --fold 4 --weights ../output/checkpoints/inception_v2_resnet_fold_4/checkpoint-011*

        # python3.6 single_frame_cnn.py generate_prediction_test --model resnet152 --fold 1 --weights ../output/checkpoints/resnet152_fold_1/checkpoint-010*
        # python3.6 single_frame_cnn.py generate_prediction_test --model resnet152 --fold 2 --weights ../output/checkpoints/resnet152_fold_2/checkpoint-010*
        # python3.6 single_frame_cnn.py generate_prediction_test --model resnet152 --fold 3 --weights ../output/checkpoints/resnet152_fold_3/checkpoint-010*
        # python3.6 single_frame_cnn.py generate_prediction_test --model resnet152 --fold 4 --weights ../output/checkpoints/resnet152_fold_4/checkpoint-010*

        # python3.6 single_frame_cnn.py generate_prediction_test --model inception_v2_resnet_extra --fold 1 --weights ../output/checkpoints/inception_v2_resnet_fold_1_extra/checkpoint-014*
        # python3.6 single_frame_cnn.py generate_prediction_test --model inception_v2_resnet_extra --fold 2 --weights ../output/checkpoints/inception_v2_resnet_fold_2_extra/checkpoint-014*
        # python3.6 single_frame_cnn.py generate_prediction_test --model inception_v2_resnet_extra --fold 3 --weights ../output/checkpoints/inception_v2_resnet_fold_3_extra/checkpoint-014*
        # python3.6 single_frame_cnn.py generate_prediction_test --model inception_v2_resnet_extra --fold 4 --weights ../output/checkpoints/inception_v2_resnet_fold_4_extra/checkpoint-014*

        # python3.6 single_frame_cnn.py save_all_combined_test_results

        # python3.6 second_stage.py
        # python3.6 second_stage_nn.py

    def fit(self, X, y):
        """Use the same architecture, but train the weights from scratch using
        the provided X and y.

        Args:
            X: training data
            y: training labels

        Returns:

        """
        pass

    def finetune(self, X, y):
        """Finetune the network for a different task by keeping the
        trained weights, replacing the top layer with one that outputs
        the new classes, and re-training for a few epochs to have the
        model output the new classes instead.

        Args:
            X:
            y:

        Returns:

        """
        pass

    def save_model(self):
        """Save the model weights, checkpoints, to model_path.
        """
