from zamba.pytorch_lightning.base_module import ZambaClassificationLightningModule

available_models = {}

_registered = False


def ensure_registered():
    """Lazily import model modules so they register themselves via @register_model.

    This avoids pulling in heavy video dependencies (pytorchvideo, fvcore, etc.)
    at package import time.
    """
    global _registered
    if _registered:
        return
    _registered = True

    try:
        from zamba.models.efficientnet_models import TimeDistributedEfficientNet  # noqa: F401
    except ImportError:
        pass

    try:
        from zamba.models.slowfast_models import SlowFast  # noqa: F401
    except ImportError:
        pass


def register_model(cls):
    """Used to decorate subclasses of ZambaClassificationLightningModule so that they are
    included in available_models."""
    if not issubclass(cls, ZambaClassificationLightningModule):
        raise TypeError(
            "Cannot register object that is not a subclass of "
            "ZambaClassificationLightningModule."
        )
    available_models[cls.__name__] = cls
    return cls
