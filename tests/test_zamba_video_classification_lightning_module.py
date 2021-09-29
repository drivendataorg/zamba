import pytest

from zamba.models.slowfast_models import SlowFast
from zamba.models.efficientnet_models import TimeDistributedEfficientNet

from conftest import DummyZambaVideoClassificationLightningModule


@pytest.mark.parametrize("model_class", (SlowFast, TimeDistributedEfficientNet))
def test_save_and_load(model_class, tmp_path):
    model = model_class(species=["cat", "dog"])
    model.to_disk(tmp_path / model_class.__name__)
    model = model_class.from_disk(tmp_path / model_class.__name__)
    assert model.species == ["cat", "dog"]
    assert model.num_classes == 2


def test_load_dummy(dummy_checkpoint, tmp_path):
    model = DummyZambaVideoClassificationLightningModule.from_disk(dummy_checkpoint)

    assert (model.head.weight == 0).all()
    assert (model.backbone.weight == 1).all()


def test_save_and_load_trainer_checkpoint(dummy_trainer, tmp_path):
    # Check that frozen backbone did not change
    backbone = dummy_trainer.model.model[2]
    assert (backbone.weight == 1).all()

    # Check that model learned something during training
    linear = dummy_trainer.model.model[3]
    assert not (linear.weight == 0).all()

    # Save checkpoint from trainer
    dummy_trainer.save_checkpoint(tmp_path / "dummy.ckpt")

    # Load checkpoint
    model = DummyZambaVideoClassificationLightningModule.from_disk(tmp_path / "dummy.ckpt")

    loaded_backbone = model.model[2]
    assert (loaded_backbone.weight == 1).all()
    for param in backbone.parameters():
        assert not param.requires_grad

    loaded_linear = model.model[3]
    assert (loaded_linear.weight == linear.weight).all()
    for param in loaded_linear.parameters():
        assert param.requires_grad
