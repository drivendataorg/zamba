import torch
import typer

from zamba_algorithms.data.video import VideoLoaderConfig
from zamba_algorithms.models.model_manager import train_model, predict_model
from zamba_algorithms.pytorch.transforms import zamba_image_model_transforms
from zamba_algorithms.pytorch_lightning.utils import ZambaDataModule
from zamba_algorithms.settings import DATA_DIRECTORY


app = typer.Typer()


@app.command()
def time_distributed_resnet():

    video_loader_config = VideoLoaderConfig(
        early_bias=True,
        video_height=224,
        video_width=224,
        ensure_total_frames=True,
        crop_bottom_pixels=50,
    )

    zamba_data_module = ZambaDataModule(
        batch_size=16,
        num_workers=8,
        video_loader_config=video_loader_config,
        transform=zamba_image_model_transforms(),
    )

    trainer = train_model(
        model_class="time_distributed_resnet",
        data_module=zamba_data_module,
        max_epochs=10,
    )

    predict_model(
        data_module=zamba_data_module,
        trainer=trainer,
    )


@app.command()
def time_distributed_efficientnet():

    video_loader_config = VideoLoaderConfig(
        early_bias=True,
        # note: small, square inputs for faster training and experimentation
        # will likely want to increase resolution
        video_height=224,
        video_width=224,
        ensure_total_frames=True,
        crop_bottom_pixels=50,
    )

    zamba_data_module = ZambaDataModule(
        batch_size=16,
        num_workers=16,
        video_loader_config=video_loader_config,
        transform=zamba_image_model_transforms(),
        load_metadata_config=dict(
            zamba_label="original",
            subset="half",
        ),
    )

    trainer = train_model(
        model_class="time_distributed_efficientnet",
        data_module=zamba_data_module,
    )

    predict_model(
        data_module=zamba_data_module,
        trainer=trainer,
    )


@app.command()
def time_distributed_efficientnet_finetuned_european():

    video_loader_config = VideoLoaderConfig(
        video_height=224,
        video_width=224,
        crop_bottom_pixels=50,
        ensure_total_frames=True,
        megadetector_lite=0.25,
        megadetector_lite_kwargs=dict(fill_mode="score_sorted", n_frames=16),
        total_frames=16,
    )

    zamba_data_module = ZambaDataModule(
        batch_size=16,
        num_workers=0,
        prefetch_factor=2,
        num_classes=11,
        video_loader_config=video_loader_config,
        transform=zamba_image_model_transforms(),
        load_metadata_config={"zamba_label": "european", "subset": "european"},
    )

    trainer = train_model(
        model_class="time_distributed_efficientnet_multilayer_head",
        model_name="time_distributed_efficientnet_finetuned_european",
        model_params={
            "finetune_from": DATA_DIRECTORY
            / "results"
            / "time_distributed_efficientnet_multilayer_head_mdlite/"
            / "version_0"
            / "checkpoints"
            / "epoch=5-step=48678.ckpt"
        },
        data_module=zamba_data_module,
    )

    predict_model(
        data_module=zamba_data_module,
        trainer=trainer,
    )


@app.command()
def time_distributed_efficientnet_multilayer_head():

    video_loader_config = VideoLoaderConfig(
        early_bias=True,
        video_height=224,
        video_width=224,
        ensure_total_frames=True,
        crop_bottom_pixels=50,
    )

    zamba_data_module = ZambaDataModule(
        batch_size=16,
        num_workers=8,
        video_loader_config=video_loader_config,
        transform=zamba_image_model_transforms(),
    )

    trainer = train_model(
        model_class="time_distributed_efficientnet_multilayer_head",
        data_module=zamba_data_module,
    )

    predict_model(
        data_module=zamba_data_module,
        trainer=trainer,
    )


@app.command()
def time_distributed_efficientnet_finetune():

    video_loader_config = VideoLoaderConfig(
        early_bias=True,
        # note: small, square inputs for faster training and experimentation
        # will likely want to increase resolution
        video_height=224,
        video_width=224,
        ensure_total_frames=True,
        crop_bottom_pixels=50,
    )

    zamba_data_module = ZambaDataModule(
        batch_size=4,
        num_workers=16,
        video_loader_config=video_loader_config,
        transform=zamba_image_model_transforms(),
        load_metadata_config=dict(
            zamba_label="original",
            subset="half",
        ),
    )

    trainer = train_model(
        model_class="time_distributed_efficientnet",
        model_name="time_distributed_efficientnet_finetune",
        data_module=zamba_data_module,
        backbone_finetune=True,
        backbone_finetune_params={
            "unfreeze_backbone_at_epoch": 3,
            "verbose": True,
            "pre_train_bn": True,
            "multiplier": 1,
        },
        auto_lr_find=True,
        early_stopping_params={
            "patience": 5,
        },
        model_params={
            "scheduler": torch.optim.lr_scheduler.MultiStepLR,
            "scheduler_params": {"milestones": [3], "gamma": 0.5, "verbose": True},
        },
    )

    predict_model(
        data_module=zamba_data_module,
        trainer=trainer,
    )


@app.command()
def time_distributed_efficientnet_multilayer_head_finetune():

    video_loader_config = VideoLoaderConfig(
        early_bias=True,
        video_height=224,
        video_width=224,
        ensure_total_frames=True,
        crop_bottom_pixels=50,
    )

    zamba_data_module = ZambaDataModule(
        batch_size=4,
        num_workers=16,
        video_loader_config=video_loader_config,
        transform=zamba_image_model_transforms(),
        load_metadata_config=dict(
            zamba_label="new",
            subset="half",
        ),
    )

    trainer = train_model(
        model_class="time_distributed_efficientnet_multilayer_head",
        model_name="time_distributed_efficientnet_multilayer_head_finetune",
        data_module=zamba_data_module,
        backbone_finetune=True,
        backbone_finetune_params={
            "unfreeze_backbone_at_epoch": 3,
            "verbose": True,
            "pre_train_bn": True,
            "multiplier": 1,
        },
        auto_lr_find=True,
        early_stopping_params={
            "patience": 5,
        },
        model_params={
            "scheduler": torch.optim.lr_scheduler.MultiStepLR,
            "scheduler_params": {"milestones": [3], "gamma": 0.5, "verbose": True},
        },
    )

    predict_model(
        data_module=zamba_data_module,
        trainer=trainer,
    )


@app.command()
def time_distributed_efficientnet_mdlite():

    video_loader_config = VideoLoaderConfig(
        video_height=224,
        video_width=224,
        crop_bottom_pixels=50,
        ensure_total_frames=True,
        megadetector_lite=0.25,
        megadetector_lite_kwargs=dict(fill_mode="score_sorted", n_frames=16),
        total_frames=16,
    )

    zamba_data_module = ZambaDataModule(
        batch_size=16,
        num_workers=0,
        prefetch_factor=2,
        video_loader_config=video_loader_config,
        transform=zamba_image_model_transforms(),
        load_metadata_config=dict(
            zamba_label="original",
            subset="half",
        ),
    )

    trainer = train_model(
        model_class="time_distributed_efficientnet",
        model_name="time_distributed_efficientnet_mdlite",
        data_module=zamba_data_module,
    )

    predict_model(
        data_module=zamba_data_module,
        trainer=trainer,
    )


@app.command()
def time_distributed_efficientnet_multilayer_head_mdlite():

    video_loader_config = VideoLoaderConfig(
        video_height=224,
        video_width=224,
        crop_bottom_pixels=50,
        ensure_total_frames=True,
        megadetector_lite=0.25,
        megadetector_lite_kwargs=dict(fill_mode="score_sorted", n_frames=16),
        total_frames=16,
    )

    zamba_data_module = ZambaDataModule(
        batch_size=8,
        num_workers=0,
        prefetch_factor=2,
        video_loader_config=video_loader_config,
        transform=zamba_image_model_transforms(),
        load_metadata_config=dict(
            zamba_label="new",
            subset="half",
        ),
        num_classes=33,
    )

    trainer = train_model(
        model_class="time_distributed_efficientnet_multilayer_head",
        model_name="time_distributed_efficientnet_multilayer_head_mdlite",
        data_module=zamba_data_module,
        backbone_finetune=True,
        backbone_finetune_params={
            "unfreeze_backbone_at_epoch": 3,
            "verbose": True,
            "pre_train_bn": True,
            "multiplier": 1,
        },
        auto_lr_find=True,
        early_stopping_params={
            "patience": 5,
        },
        model_params={
            "scheduler": torch.optim.lr_scheduler.MultiStepLR,
            "scheduler_params": {"milestones": [3], "gamma": 0.5, "verbose": True},
        },
    )

    predict_model(
        data_module=zamba_data_module,
        trainer=trainer,
    )


if __name__ == "__main__":
    app()
