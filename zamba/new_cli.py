from pathlib import Path
from typing import Optional

import typer

from zamba.models.manager import TrainConfig, PredictConfig, ModelManager


app = typer.Typer()


@app.command()
def train(
    train_data: Path = typer.Option(
        Path("train_videos"), help="Path to folder containing training videos."
    ),
    val_data: Path = typer.Option(
        Path("val_videos"), help="Path to folder containing validation videos."
    ),
    labels: Path = typer.Option(Path("labels.csv"), help="Path to csv containing video labels."),
    config: Path = typer.Option(
        None,
        help="Specify options using yaml configuration file instead of through command line options.",
    ),
    model_path: Path = typer.Option(Path("custom_model.h5"), help="Path to model to train."),
    model_library: str = typer.Option("keras", help="Library to use for loading custom model."),
    n_epochs: Optional[int] = typer.Option(10, help="Number of epochs to train."),
    tempdir: Optional[Path] = typer.Option(
        None, help="Path to temporary directory. If not specified, OS temporary directory is used"
    ),
    model_class: str = typer.Option(
        "custom",
        help="[Not implemented] Model from model zoo to train. Currently only external custom models are supported.",
    ),
    height: Optional[int] = typer.Option(
        None, help="[Not implemented] Desired height for resized video."
    ),
    width: Optional[int] = typer.Option(
        None, help="[Not implemented] Desired width for resized video."
    ),
    augmentation: Optional[bool] = typer.Option(
        False, help="[Not implemented] If True, flip and rotate videos during training."
    ),
    early_stopping: Optional[bool] = typer.Option(
        False, help="[Not implemented] If True, use early stopping."
    ),
    save_path: Optional[Path] = typer.Option(
        None, help="[Not implemented] Save out trained model to this filepath."
    ),
):
    """Train a custom model using the provided data, labels, and path to preconfigured model architecture."""
    if config is not None:
        manager = ModelManager.from_config(config)

    else:
        manager = ModelManager(
            train_config=TrainConfig(
                train_data=train_data,
                val_data=val_data,
                labels=labels,
                config=config,
                model_path=model_path,
                model_class=model_class,
                tempdir=tempdir,
                n_epochs=n_epochs,
                height=height,
                width=width,
                augmentation=augmentation,
                early_stopping=early_stopping,
            )
        )

    typer.echo(f"Using train data_path:\t{manager.train_config.train_data}")
    typer.echo(f"Using val data_path:\t{manager.train_config.val_data}")
    typer.echo(f"Using labels:\t{manager.train_config.labels}")
    typer.echo(f"Using model:\t{manager.train_config.model_path}")
    typer.echo(f"Loading with:\t{manager.train_config.model_library}")

    # manager.train()


@app.command()
def predict(
    data_path: Path = typer.Option(Path("."), help="Path to videos."),
    model_path: Path = Path("."),
    model_class: str = "cnnensemble",
    pred_path: Optional[Path] = None,
    output_class_names: Optional[bool] = typer.Option(
        False,
        help="If True, we just return a video and the name of the most likely class. If False, we return a probability or indicator (depending on --proba_threshold) for every possible class.",
    ),
    proba_threshold: Optional[float] = None,
    tempdir: Optional[Path] = None,
    verbose: Optional[bool] = typer.Option(
        False, help="Displays additional logging information during processing."
    ),
    save: Optional[bool] = False,
    # model_kwargs: Optional[dict] = dict(),
    # predict_kwargs: Optional[dict] = dict(),
):
    # TODO: add profile and other params to predict and pass in accordingly
    print(PredictConfig(**kwargs))


if __name__ == "__main__":
    app()
