import torch
import typer

from zamba_algorithms.data.video import VideoLoaderConfig
from zamba_algorithms.models.model_manager import train_model, predict_model
from zamba_algorithms.pytorch.transforms import timesformer_transforms
from zamba_algorithms.pytorch_lightning.utils import ZambaDataModule

app = typer.Typer()


@app.command()
def timesformer_zamba_finetune():
    video_loader_config = VideoLoaderConfig(
        frame_indices=[2, 12, 24, 48, 72, 96, 120, 144],  # Like early_bias, but needs 8 frames
        video_height=224,
        video_width=224,
        crop_bottom_pixels=50,
    )

    zamba_data_module = ZambaDataModule(
        batch_size=32, video_loader_config=video_loader_config, transform=timesformer_transforms()
    )

    trainer = train_model(
        data_module=zamba_data_module,
        model_class="timesformer",
        model_name="timesformer_zamba_finetune",
        model_params={
            "scheduler": torch.optim.lr_scheduler.MultiStepLR,
            "scheduler_params": {"milestones": [5], "gamma": 0.5, "verbose": True},
        },
        backbone_finetune=True,
        max_epochs=100,
    )

    predict_model(
        dataloader=zamba_data_module.test_dataloader(),
        trainer=trainer,
    )


if __name__ == "__main__":
    app()
