from zamba.pytorch_lightning.base_module import ZambaClassificationLightningModule

available_models = {}


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
