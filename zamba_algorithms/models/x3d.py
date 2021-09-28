import numpy as np
import torch
import typer

from zamba_algorithms.data.video import VideoLoaderConfig
from zamba_algorithms.models.model_manager import train_model, predict_model
from zamba_algorithms.pytorch.transforms import x3d_transforms
from zamba_algorithms.pytorch_lightning.utils import ZambaDataModule

app = typer.Typer()


@app.command()
def x3d_zamba_finetune():
    video_loader_config = VideoLoaderConfig(
        frame_indices=np.r_[
            np.arange(2, 2 + 2 * 16, 4), np.arange(2 + 2 * 16, 2 + 2 * 16 + 8 * 16, 16)
        ].tolist(),  # Like early_bias, but x3d needs > 16 frames
        video_height=224,
        video_width=224,
        crop_bottom_pixels=50,
    )

    zamba_data_module = ZambaDataModule(
        batch_size=32,
        video_loader_config=video_loader_config,
        transform=x3d_transforms(),
    )

    trainer = train_model(
        data_module=zamba_data_module,
        model_class="x3d",
        model_name="x3d_zamba_finetune",
        model_params={
            "scheduler": torch.optim.lr_scheduler.MultiStepLR,
            "scheduler_params": {"milestones": [5], "gamma": 0.5, "verbose": True},
        },
        backbone_finetune=True,
        backbone_finetune_params={"unfreeze_backbone_at_epoch": 5},
        max_epochs=50,
    )

    predict_model(
        dataloader=zamba_data_module.test_dataloader(),
        trainer=trainer,
    )


if __name__ == "__main__":
    app()
