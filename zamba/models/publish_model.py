from pathlib import Path
import shutil

import typer

from zamba.settings import ROOT_DIRECTORY, RESULTS_DIR


def publish_model(model_dir):
    """
    Copies local tensorboard model_dir with checkpoint, preds, and hparams to data/results/ on s3.
    """

    # get just portion of path within tb logs dir
    relative_model_dir = (
        Path(model_dir)
        .resolve()
        .relative_to(ROOT_DIRECTORY / "zamba" / "models" / "tensorboard_logs")
    )

    if (RESULTS_DIR / relative_model_dir).exists():
        raise ValueError(
            f"{relative_model_dir} already exists on s3. Please rename with a new version number."
        )
    else:
        shutil.copytree(model_dir, RESULTS_DIR / relative_model_dir)


if __name__ == "__main__":
    typer.run(publish_model)
