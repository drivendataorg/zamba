import typer

from zamba_algorithms.data.video import VideoLoaderConfig
from zamba_algorithms.models.model_manager import train_model, predict_model
from zamba_algorithms.pytorch.transforms import zamba_image_model_transforms
from zamba_algorithms.pytorch_lightning.utils import ZambaDataModule


app = typer.Typer()


@app.command()
def single_frame_resnet():

    zamba_data_module = ZambaDataModule(
        batch_size=4,
        num_workers=8,
        video_loader_config=VideoLoaderConfig(
            frame_indices=[0], video_height=404, video_width=720
        ),
        transform=zamba_image_model_transforms(single_frame=True),
    )

    trainer = train_model(
        model_class="single_frame_resnet",
        model_name="first_frame_resnet",
        data_module=zamba_data_module,
    )

    predict_model(
        dataloader=zamba_data_module.test_dataloader(),
        trainer=trainer,
    )


@app.command()
def single_frame_mdlite_resnet():

    video_loader_config = VideoLoaderConfig(
        video_height=404,
        video_width=720,
        ensure_total_frames=True,
        megadetector_lite_config={"confidence": 0.9, "fill_mode": "score_sorted"},
        total_frames=1,
    )

    zamba_data_module = ZambaDataModule(
        batch_size=4,
        num_workers=0,
        video_loader_config=video_loader_config,
        transform=zamba_image_model_transforms(single_frame=True),
        prefetch_factor=2,  # default value so pytorch does not complain about not doing multiprocessing
    )

    trainer = train_model(
        model_class="single_frame_resnet",
        model_name="single_frame_mdlite_resnet",
        data_module=zamba_data_module,
        max_epochs=10,
    )

    predict_model(
        dataloader=zamba_data_module.test_dataloader(),
        trainer=trainer,
    )


@app.command()
def resnet_r2plus1d():

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
        transform=zamba_image_model_transforms(
            normalization_values=dict(
                mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
            ),
            channels_first=True,
        ),
    )

    trainer = train_model(
        model_class="resnet_r2plus1d",
        data_module=zamba_data_module,
    )

    predict_model(
        dataloader=zamba_data_module.test_dataloader(),
        trainer=trainer,
    )


if __name__ == "__main__":
    app()
