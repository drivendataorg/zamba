import numpy as np
import torch
import typer

from zamba.data.video import VideoLoaderConfig
from zamba.models.model_manager import train_model, predict_model
from zamba.pytorch.transforms import slowfast_transforms
from zamba.pytorch_lightning.utils import ZambaDataModule

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



if __name__ == "__main__":
    app()
