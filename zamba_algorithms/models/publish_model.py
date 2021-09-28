from pathlib import Path

from cloudpathlib import S3Path
import typer

from zamba_algorithms.settings import ROOT_DIRECTORY


def publish_model(model_dir):
    """
    Copies local tensorboard model_dir with checkpoint, preds, and hparams to data/results/ on s3.
    """
    if (ROOT_DIRECTORY / "zamba_algorithms" / "models" / "tensorboard_logs").exists():
        tb_root = ROOT_DIRECTORY / "zamba_algorithms" / "models" / "tensorboard_logs"

    else:
        tb_root = ROOT_DIRECTORY

    # get just portion of path within tb logs dir
    relative_model_dir = str(Path(model_dir).resolve().relative_to(tb_root))

    results_dir = S3Path("s3://drivendata-client-zamba/data/results")

    if (results_dir / relative_model_dir).exists():
        raise ValueError(
            f"{relative_model_dir} already exists on s3. Please rename with a new version number."
        )
    else:
        (results_dir / relative_model_dir).upload_from(model_dir)


if __name__ == "__main__":
    typer.run(publish_model)
