# avoid using a backend when not installed
# as a framework.
import matplotlib
matplotlib.use('Agg')

# import h5py first and ignore warnings which will go away in 2.7.2 (prevents annoying messages on start)
# import tensorflow and ignore compiletime mismatch for tensorflow.python.framework.fast_tensor_util. Does no impact
# performance. see https://github.com/tensorflow/tensorflow/issues/14182
import warnings  # noqa: E402
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py  # noqa: F401

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    import tensorflow  # noqa: F401

# set tensorflow logging to ignore warnings for deprecated alias `normal` for `truncated_normal`. Warning arises from
# within keras that is packaged inside of tensorflow
tensorflow.logging.set_verbosity(tensorflow.logging.ERROR)
