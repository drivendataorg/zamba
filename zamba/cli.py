import logging
from pathlib import Path
from typing import Optional

import typer

from zamba.models.config import (
    ModelClassEnum,
    FrameworkEnum,
    PredictConfig,
    TrainConfig,
    ModelConfig,
)
from zamba.models.manager import ModelManager
from zamba.models.cnnensemble_model import CnnModelProfileEnum


app = typer.Typer()


@app.command()
def train(
    train_data: Path = typer.Option(
        None, exists=True, help="Path to folder containing training videos."
    ),
    val_data: Path = typer.Option(
        None, exists=True, help="Path to folder containing validation videos."
    ),
    labels: Path = typer.Option(None, exists=True, help="Path to csv containing video labels."),
    model_path: Path = typer.Option(None, exists=True, help="Path to model to train."),
    framework: FrameworkEnum = typer.Option(
        FrameworkEnum.keras, help="Library to use for loading custom model."
    ),
    config: Path = typer.Option(
        None,
        exists=True,
        help="Specify options using yaml configuration file instead of through command line options.",
    ),
    tempdir: Optional[Path] = typer.Option(
        None,
        exists=True,
        help="Path to temporary directory. If not specified, OS temporary directory is used",
    ),
    n_epochs: Optional[int] = typer.Option(10, help="Number of epochs to train."),
    save_path: Optional[Path] = typer.Option(
        None, help="[Not implemented] Save out trained model to this filepath."
    ),
    model_class: str = typer.Option(
        "custom",
        help="[Not implemented] Model from model zoo to train. Currently only external custom models are supported.",
    ),
):
    """Train a custom model using the provided data, labels, and path to preconfigured model architecture."""
    if config is not None:
        typer.echo(f"Loading from config file: {config}. Any other arguments passed will be ignored.")
        manager = ModelManager.from_config(config)

    else:
        manager = ModelManager(
            model_config=ModelConfig(
                model_class=model_class,
            ),
            train_config=TrainConfig(
                train_data=train_data,
                val_data=val_data,
                labels=labels,
                model_path=model_path,
                framework=framework,
                tempdir=tempdir,
                n_epochs=n_epochs,
                save_path=save_path,
            )
        )

    typer.echo(f"Using train data_path:\t{manager.train_config.train_data}")
    typer.echo(f"Using val data_path:\t{manager.train_config.val_data}")
    typer.echo(f"Using labels:\t{manager.train_config.labels}")
    typer.echo(f"Using model:\t{manager.train_config.model_path}")
    typer.echo(f"Loading with:\t{manager.train_config.framework}")

    manager.train()


@app.command()
def predict(
    data_path: Path = typer.Option(
        None, exists=True, help="Path to folder containing videos for prediction."
    ),
    model_class: ModelClassEnum = typer.Option(
        "cnnensemble",
        help="Controls whether prodcution model, external custom model, or sample model is used.",
    ),
    model_profile: Optional[CnnModelProfileEnum] = typer.Option(
        CnnModelProfileEnum.full,
        help="If model_class is 'cnnensemble', options are 'full' which is slow and accurate or 'fast' \
which is faster and less accurate.",
    ),
    config: Path = typer.Option(
        None,
        exists=True,
        help="Specify options using yaml configuration file instead of through command line options.",
    ),
    pred_path: Optional[Path] = typer.Option(
        None,
        help="Filepath for predictions csv. If not specified, predictions will be saved to \
`predictions-{data-path}-{timestamp}.csv`.",
    ),
    proba_threshold: Optional[float] = typer.Option(
        None,
        help="Probability threshold for classification. If specified, binary predictions are \
returned with 1 being greater than the threshold, 0 being less than or equal to.If not specified, \
probabilities between 0 and 1 are returned.",
    ),
    output_class_names: Optional[bool] = typer.Option(
        False,
        help="If True, return a video and the name of the most likely class. If False, \
return a probability or indicator (depending on --proba_threshold) for every possible class.",
    ),
    resample: Optional[bool] = typer.Option(
        False,
        help="Resample videos to a consistent size and framerate (fps=15, width=448, height=252).",
    ),
    separate_blank: Optional[bool] = typer.Option(
        False,
        help="Do a second stage blank/non-blank that is more accurate; recommended with resample.",
    ),
    tempdir: Optional[Path] = typer.Option(
        None,
        exists=True,
        help="Path to temporary directory. If not specified, OS temporary directory is used.",
    ),
    verbose: Optional[bool] = typer.Option(
        False, help="Displays additional logging information during processing."
    ),
    weight_download_region: Optional[str] = typer.Option(
        "us", help="Server region for downloading weights. Options are 'us', 'eu', or 'asia'."
    ),
    save: Optional[bool] = typer.Option(True, help="Save predictions to csv file."),
    model_path: Path = typer.Option(
        None, exists=True, help="[Not implemented] Path to model to use for prediction."
    ),
):
    """Identify species in a video.

    This is a command line interface for prediction on camera trap footage. Given a path to camera
    trap footage, the predict function use a deep learning model to predict the presence or absense of
    a variety of species of common interest to wildlife researchers working with camera trap data.

    """

    if config is not None:
        typer.echo(f"Loading from config file: {config}. Any other arguments passed will be ignored.")
        manager = ModelManager.from_config(config)

    else:
        manager = ModelManager(
            model_config=ModelConfig(
                model_class=model_class,
                model_kwargs=dict(
                    profile=model_profile,
                    resample=resample,
                    seperate_blank_model=separate_blank,
                ),
            ),
            predict_config=PredictConfig(
                data_path=data_path,
                model_path=model_path,
                pred_path=pred_path,
                proba_threshold=proba_threshold,
                output_class_names=output_class_names,
                tempdir=tempdir,
                verbose=verbose,
                download_region=weight_download_region,
                save=save,
            )
        )

    typer.echo(f"Using data_path:\t{manager.predict_config.data_path}")
    typer.echo(f"Using model_class:\t{manager.model_config.model_class}")

    manager.predict()


@app.command()
def tune(
    data_path: Path = typer.Option(
        None, exists=True, help="Path to folder containing videos for finetuning."
    ),
    labels: Path = typer.Option(None, exists=True, help="Path to csv containing video labels."),
    config: Path = typer.Option(
        None,
        exists=True,
        help="Specify options using yaml configuration file instead of through command line options.",
    ),
    batch_size: Optional[int] = typer.Option(
        8, min=1, max=256, clamp=True, help="Batch size for model inputs."
    ),
    weights_output: Optional[Path] = typer.Option(None, help="Output path for saved weights."),
    tempdir: Optional[Path] = typer.Option(
        None,
        exists=True,
        help="Path to temporary directory. If not specified, OS temporary directory is used",
    ),
):
    """[NOT IMPLEMENTED] Update network with new data.

    Finetune the network for a different task by keeping the
    trained weights, replacing the top layer with one that outputs
    the new classes, and re-training for a few epochs to have the
    model output the new classes instead.

    """
    typer.echo("Finetuning an algorithm to new data is not yet implemented.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    app()
