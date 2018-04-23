# avoid using a backend when not installed
# as a framework.
import matplotlib
matplotlib.use('Agg')

# import h5py first and ignore warnings which will go away in 2.7.2 (prevents annoying messages on start)
import warnings  # noqa: E402
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py  # noqa: F401
