from pathlib import Path

from .model import Model
from .cnnensemble.src.single_frame_cnn import generate_prediction_test, save_all_combined_test_results
from .cnnensemble.src import second_stage, second_stage_nn

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
        return Path('.')

    def predict(self, X):
        """Predict class probabilities for each input, X

        Args:
            X: input data

        Returns:

        """

        # # resnet50_avg (except fold 2)
        # generate_prediction_test(model_name='resnet50_avg',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/resnet50_avg_fold_1/checkpoint-007-0.0480.hdf5'),
        #                          fold=1)
        # generate_prediction_test(model_name='resnet50',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/resnet50_fold_2/checkpoint-016-0.0342.hdf5'),
        #                          fold=2)
        # generate_prediction_test(model_name='resnet50_avg',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/resnet50_avg_fold_3/checkpoint-004-0.0813.hdf5'),
        #                          fold=3)
        # generate_prediction_test(model_name='resnet50_avg',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/resnet50_avg_fold_4/checkpoint-015-0.0352.hdf5'),
        #                          fold=4)
        #
        # # xception_avg
        # generate_prediction_test(model_name='xception_avg',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/xception_avg_fold_1/checkpoint-004-0.1295.hdf5'),
        #                          fold=1)
        # generate_prediction_test(model_name='xception_avg',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/xception_avg_fold_2/checkpoint-005-0.0308.hdf5'),
        #                          fold=2)
        # generate_prediction_test(model_name='xception_avg',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/xception_avg_fold_3/checkpoint-004-0.0316.hdf5'),
        #                          fold=3)
        # generate_prediction_test(model_name='xception_avg',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/xception_avg_fold_4/checkpoint-004-0.1665.hdf5'),
        #                          fold=4)
        #
        # # xception_avg_ch10
        # generate_prediction_test(model_name='xception_avg_ch10',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/xception_avg_fold_1/checkpoint-009-0.1741.hdf5'),
        #                          fold=1)
        # generate_prediction_test(model_name='xception_avg_ch10',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/xception_avg_fold_2/checkpoint-010-0.0272.hdf5'),
        #                          fold=2)
        # generate_prediction_test(model_name='xception_avg_ch10',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/xception_avg_fold_3/checkpoint-009-0.0397-0.1153.hdf5'),
        #                          fold=3)
        # generate_prediction_test(model_name='xception_avg_ch10',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/xception_avg_fold_4/checkpoint-009-0.1875.hdf5'),
        #                          fold=4)
        #
        # # inception_v3
        # generate_prediction_test(model_name='inception_v3',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/inception_v3_fold_1/checkpoint-009-0.0499-0.1092.hdf5'),
        #                          fold=1)
        # generate_prediction_test(model_name='inception_v3',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/inception_v3_fold_2/checkpoint-009-0.0391-0.1190.hdf5'),
        #                          fold=2)
        # generate_prediction_test(model_name='inception_v3',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/inception_v3_fold_3/checkpoint-009-0.0372-0.0298.hdf5'),
        #                          fold=3)
        # generate_prediction_test(model_name='inception_v3',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/inception_v3_fold_4/checkpoint-009-0.0375-0.0307.hdf5'),
        #                          fold=4)
        #
        # # inception_v2_resnet
        # generate_prediction_test(model_name='inception_v2_resnet',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/inception_v2_resnet_fold_1/checkpoint-005-0.0347.hdf5'),
        #                          fold=1)
        # generate_prediction_test(model_name='inception_v2_resnet',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/inception_v2_resnet_fold_2/checkpoint-005-0.1555.hdf5'),
        #                          fold=2)
        # generate_prediction_test(model_name='inception_v2_resnet',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/inception_v2_resnet_fold_3/checkpoint-005-0.0342.hdf5'),
        #                          fold=3)
        # generate_prediction_test(model_name='inception_v2_resnet',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/inception_v2_resnet_fold_4/checkpoint-005-0.1457.hdf5'),
        #                          fold=4)
        #
        #
        # # inception_v2_resnet_ch10
        # generate_prediction_test(model_name='inception_v2_resnet_ch10',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/inception_v2_resnet_fold_1/checkpoint-011-0.0380-0.0313.hdf5'),
        #                          fold=1)
        # generate_prediction_test(model_name='inception_v2_resnet_ch10',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/inception_v2_resnet_fold_2/checkpoint-011-0.0371-0.0306.hdf5'),
        #                          fold=2)
        # generate_prediction_test(model_name='inception_v2_resnet_ch10',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/inception_v2_resnet_fold_3/checkpoint-009-0.0344.hdf5'),
        #                          fold=3)
        # generate_prediction_test(model_name='inception_v2_resnet_ch10',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/inception_v2_resnet_fold_4/checkpoint-011-0.0401-0.0314.hdf5'),
        #                          fold=4)
        #
        #
        # # resnet152
        # generate_prediction_test(model_name='resnet152',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/resnet152_fold_1/checkpoint-010-0.0453-0.0715.hdf5'),
        #                          fold=1)
        # generate_prediction_test(model_name='resnet152',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/resnet152_fold_2/checkpoint-010-0.0495-0.0379.hdf5'),
        #                          fold=2)
        # generate_prediction_test(model_name='resnet152',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/resnet152_fold_3/checkpoint-010-0.0347-0.0995.hdf5'),
        #                          fold=3)
        # generate_prediction_test(model_name='resnet152',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/resnet152_fold_4/checkpoint-010-0.0373-0.0355.hdf5'),
        #                          fold=4)
        #
        # # inception_v2_resnet_extra
        # generate_prediction_test(model_name='inception_v2_resnet_extra',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/inception_v2_resnet_fold_1_extra/checkpoint-014-0.0313-0.1366.hdf5'),
        #                          fold=1)
        # generate_prediction_test(model_name='inception_v2_resnet_extra',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/inception_v2_resnet_fold_2_extra/checkpoint-014-0.0319-0.0803.hdf5'),
        #                          fold=2)
        # generate_prediction_test(model_name='inception_v2_resnet_extra',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/inception_v2_resnet_fold_3_extra/checkpoint-014-0.0310-0.0254.hdf5'),
        #                          fold=3)
        # generate_prediction_test(model_name='inception_v2_resnet_extra',
        #                          weights=(Path(__file__).parent / 'cnnensemble' /
        #                                   'output/checkpoints/inception_v2_resnet_fold_4_extra/checkpoint-014-0.0319-0.0252.hdf5'),
        #                          fold=4)


        # save_all_combined_test_results
        save_all_combined_test_results(skip_existing=False)

        # second_stage
        second_stage.main()

        # second_stage_nn
        second_stage_nn.main()

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
