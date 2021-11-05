from zamba.pytorch_lightning.utils import ZambaVideoClassificationLightningModule

available_models = {}


def register_model(cls):
    """Used to decorate subclasses of ZambaVideoClassificationLightningModule so that they are
    included in available_models."""
    if not issubclass(cls, ZambaVideoClassificationLightningModule):
        raise TypeError(
            "Cannot register object that is not a subclass of "
            "ZambaVideoClassificationLightningModule."
        )
    available_models[cls.__name__] = cls
    return cls
