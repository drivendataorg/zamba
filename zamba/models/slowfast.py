from pathlib import Path

import numpy as np
import torch
import typer

from zamba_algorithms.data.video import VideoLoaderConfig
from zamba_algorithms.models.model_manager import train_model, predict_model
from zamba_algorithms.pytorch.transforms import slowfast_transforms
from zamba_algorithms.pytorch_lightning.utils import ZambaDataModule
from zamba_algorithms.settings import DATA_DIRECTORY

app = typer.Typer()

video_loader_config = VideoLoaderConfig(
    frame_indices=np.r_[
        np.arange(2, 2 + 2 * 16, 2), np.arange(2 + 2 * 16, 2 + 2 * 16 + 8 * 16, 8)
    ].tolist(),  # Like early_bias, but slowfast needs 32 frames
    video_height=224,
    video_width=224,
    crop_bottom_pixels=50,
)

zamba_data_module = ZambaDataModule(
    batch_size=32,
    video_loader_config=video_loader_config,
    transform=slowfast_transforms(),
    load_metadata_config={"zamba_label": "original", "subset": "half"},
)


@app.command()
def slowfast_zamba_freeze_backbone():
    trainer = train_model(
        data_module=zamba_data_module,
        model_class="slowfast",
        model_name="slowfast_zamba_freeze_backbone",
        model_params={"backbone_mode": "eval"},
        max_epochs=100,
    )

    predict_model(
        data_module=zamba_data_module,
        trainer=trainer,
    )


@app.command()
def slowfast_zamba_finetune():
    trainer = train_model(
        data_module=zamba_data_module,
        model_class="slowfast",
        model_name="slowfast_zamba_finetune",
        model_params={
            "scheduler": torch.optim.lr_scheduler.MultiStepLR,
            "scheduler_params": {"milestones": [1], "gamma": 0.5, "verbose": True},
        },
        backbone_finetune=True,
        backbone_finetune_params={
            "multiplier": 10,
            "unfreeze_backbone_at_epoch": 1,
        },
        early_stopping_params={"patience": 10},
        max_epochs=100,
    )

    predict_model(
        data_module=zamba_data_module,
        trainer=trainer,
    )


@app.command()
def slowfast_zamba_multilayer_head_finetune():
    zamba_data_module = ZambaDataModule(
        batch_size=32,
        video_loader_config=video_loader_config,
        transform=slowfast_transforms(),
        load_metadata_config={"zamba_label": "original", "subset": "dev"},
    )
    trainer = train_model(
        data_module=zamba_data_module,
        model_class="slowfast",
        model_name="slowfast_zamba_multilayer_head_finetune",
        model_params={
            "head_hidden_layer_sizes": [512],
            "head_dropout_rate": 0.25,
            "scheduler": torch.optim.lr_scheduler.MultiStepLR,
            "scheduler_params": {"milestones": [15], "gamma": 0.5, "verbose": True},
        },
        backbone_finetune=True,
        backbone_finetune_params={
            "multiplier": 2,
            "unfreeze_backbone_at_epoch": 15,
        },
        early_stopping_params={"patience": 10},
        max_epochs=100,
    )

    predict_model(
        data_module=zamba_data_module,
        trainer=trainer,
    )


@app.command()
def slowfast_zamba_finetune_mdlite():
    zamba_data_module = ZambaDataModule(
        batch_size=32,
        video_loader_config=VideoLoaderConfig(
            video_height=224,
            video_width=224,
            crop_bottom_pixels=50,
            ensure_total_frames=True,
            megadetector_lite=0.25,
            megadetector_lite_kwargs={"fill_mode": "score_sorted", "n_frames": 32},
            total_frames=32,
        ),
        num_classes=33,
        num_workers=0,
        prefetch_factor=2,
        transform=slowfast_transforms(),
        load_metadata_config={"zamba_label": "new", "subset": "half"},
    )

    trainer = train_model(
        data_module=zamba_data_module,
        model_class="slowfast",
        model_name="slowfast_zamba_finetune_mdlite",
        model_params={
            "scheduler": torch.optim.lr_scheduler.MultiStepLR,
            "scheduler_params": {"milestones": [1], "gamma": 0.5, "verbose": True},
        },
        backbone_finetune=True,
        backbone_finetune_params={
            "multiplier": 10,
            "unfreeze_backbone_at_epoch": 1,
        },
        early_stopping_params={"patience": 5},
        max_epochs=100,
    )

    predict_model(
        data_module=zamba_data_module,
        trainer=trainer,
    )


@app.command()
def slowfast_zamba_finetune_mdlite_half_with_reduced_blanks():
    zamba_data_module = ZambaDataModule(
        batch_size=32,
        video_loader_config=VideoLoaderConfig(
            video_height=224,
            video_width=224,
            crop_bottom_pixels=50,
            ensure_total_frames=True,
            megadetector_lite=0.25,
            megadetector_lite_kwargs={"fill_mode": "score_sorted", "n_frames": 32},
            total_frames=32,
        ),
        num_classes=33,
        num_workers=0,
        prefetch_factor=2,
        transform=slowfast_transforms(),
        train_metadata=DATA_DIRECTORY / "processed" / "half_metadata_with_reduced_blanks.csv",
    )

    trainer = train_model(
        data_module=zamba_data_module,
        model_class="slowfast",
        model_name="slowfast_zamba_finetune_mdlite_half_with_reduced_blanks",
        model_params={
            "scheduler": torch.optim.lr_scheduler.MultiStepLR,
            "scheduler_params": {"milestones": [5], "gamma": 0.5, "verbose": True},
        },
        backbone_finetune=True,
        backbone_finetune_params={
            "multiplier": 1.5,
            "unfreeze_backbone_at_epoch": 5,
        },
        early_stopping_params={"patience": 10},
    )

    predict_model(data_module=zamba_data_module, trainer=trainer)


@app.command()
def slowfast_zamba_finetune_resume():
    trainer = train_model(
        data_module=zamba_data_module,
        model_class="slowfast",
        model_name="slowfast_zamba_finetune",
        model_params={
            "scheduler": torch.optim.lr_scheduler.MultiStepLR,
            "scheduler_params": {"milestones": [15], "gamma": 0.5, "verbose": True},
        },
        backbone_finetune=True,
        backbone_finetune_params={
            "multiplier": 2,
            "unfreeze_backbone_at_epoch": 15,
        },
        early_stopping_params={"patience": 10},
        max_epochs=100,
        auto_lr_find=False,
        resume_from_checkpoint=Path(
            "./zamba_algorithms/models/tensorboard_logs/slowfast_zamba_finetune/version_0/checkpoints/epoch=12-step=754.ckpt"
        ).resolve(),
    )

    predict_model(
        data_module=zamba_data_module,
        trainer=trainer,
    )


if __name__ == "__main__":
    app()
