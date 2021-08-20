# Fine-tuning


Pytorch and Pytorch Lightning includes helpers for fine-tuning a model.

## Freeze backbone, then unfreeze

Goal: Freeze everything but the head, train for a number of epochs, then unfreeze part of the base and continue training.

Pytorch Lightning has a built-in callback [`BackboneFinetuning`](https://pytorch-lightning.readthedocs.io/en/stable/extensions/generated/pytorch_lightning.callbacks.BackboneFinetuning.html) class that accomplishes that. First, you need to identify the part of the backbone that you want to train and add it as a `self.backbone: torch.nn.Module = ...` in the model `__init__` method. For example, in `zamba_algorithms.models.slowfast_models.SlowFast`:

```python
base = torch.hub.load(
    "facebookresearch/pytorchvideo", model="slowfast_r50", pretrained=True
)

# ...

base.blocks = base.blocks[:-1]  # Remove the pre-trained head

# ...

# self.backbone attribute lets `BackboneFinetune` freeze and unfreeze that module
self.backbone = base.blocks[-2:]
self.base = base
```

After providing the model with an attribute named `backbone` indicating the trainable backbone, training with backbone finetuning (with the default finetuning parameters) can be accomplished by passing `train_model(..., backbone_finetune=True)`. `backbone_finetune_params` can be passed to specify parameters other than the defaults. The following snippet (from `zamba_algorithms/models/slowfast_finetune.py`) will create a callback that unfreezes that trainable backbone at epoch 15 with a learning rate that is 1/100th of the head learning rate.

```python
trainer = train_model(
    data_module=zamba_data_module,
    model_class="slowfast",
    model_name="slowfast_zamba_finetune",
    backbone_finetune=True,
)
```


## Learning rate schedules

Goal: Adjust the learning rate during training.

The `ZambaVideoClassificationLightningModule` class and its subclasses can attach learning rate schedulers with the `ZambaVideoClassificationLightningModule.__init__(..., scheduler: torch.optim.lr_scheduler._LRScheduler, scheduler_params: dict)` parameters. For example, the following call to `train_model` will set up training to use the [`MultiStepLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html) scheduler to halve the learning rate (multiply by `gamma=0.5`) at the start of epoch 15.

```python
trainer = train_model(
    data_module=zamba_data_module,
    model_class="slowfast",
    model_name="slowfast_zamba_finetune",
    model_params={
        "scheduler": torch.optim.lr_scheduler.MultiStepLR,
        "scheduler_params": {"milestones": [15], "gamma": 0.5, "verbose": True},
    },
)
```


## Resuming training

The `train_model(..., resume_from_checkpoint=/path/to/checkpoint/epoch=1-step=1000.ckpt)` resumes training from an existing checkpoint. This can be helpful if you want to experiment with different strategies that start from the same pretrained model, e.g., training a model head then finetuning the backbone with different parameters. A few things to note:
- The epoch number is stored in the checkpoint, so if you want to train the head for 10 epochs, then finetune for another 10, you need to pass `max_epochs=10` the first fime and `max_epochs=20` the second time.
- Resuming will write to the same tensorboard version as the original model (although it will save a new checkpoint), so you might consider copying the source tensorboard version directory before resuming training.

```python

train_model(
    data_module,
    model_class="slowfast",
    model_name="slowfast_zamba_finetune",
    resume_from_checkpoint=False,
    backbone_finetune=False,
    max_epochs=10,
    ...
)
```

Saves tensorboard run to:

```
single_frame_resnet
└── version_0
    ├── checkpoints
    │   └── epoch=9-step=8599.ckpt
    ├── events.out.tfevents.1625685338.meyhorn.443779.0
    └── hparams.yaml
```

Copy the original run `version_0` to a new directory, `version_1`.

```
single_frame_resnet
├── version_0
│   ├── checkpoints
│   │   └── epoch=9-step=8599.ckpt
│   ├── events.out.tfevents.1625685338.meyhorn.443779.0
│   └── hparams.yaml
└── version_1
    ├── checkpoints
    │   └── epoch=9-step=8599.ckpt
    ├── events.out.tfevents.1625685338.meyhorn.443779.0
    └── hparams.yaml
```

Resume training, pointing `train_model` to the `version_1` checkpoint.

```python
train_model(
    data_module,
    model_class="slowfast",
    model_name="slowfast_zamba_finetune",
    resume_from_checkpoint="single_frame_resnet/version_1/checkpoints/epoch=9-step=8599.ckpt",
    backbone_finetune=True,
    backbone_finetune_params={"unfreeze_backbone_at_epoch": 10},
    auto_lr_find=False,
    max_epochs=20,
    ...
)
```

The final model checkpoint is `single_frame_resnet/version_1/checkpoints/epoch=19-step=17198.ckpt`.

```
single_frame_resnet
├── version_0
│   ├── checkpoints
│   │   └── epoch=9-step=8599.ckpt
│   ├── events.out.tfevents.1625685338.meyhorn.443779.0
│   └── hparams.yaml
└── version_1
    ├── checkpoints
    │   ├── epoch=9-step=8599.ckpt
    │   └── epoch=19-step=17198.ckpt
    ├── events.out.tfevents.1625685338.meyhorn.443779.0
    ├── events.out.tfevents.1625686647.meyhorn.443779.1
    └── hparams.yaml
```
