from pathlib import Path

import click

from djamba.models.manager import ModelManager

default_model_dir = Path('models', 'assets')


@click.group()
def main():
    pass


@main.command()
@click.argument('data_path',
                type=click.Path(exists=True,
                                file_okay=True,
                                dir_okay=True),
                default=Path('.'))
@click.argument('pred_path',
                type=click.Path(),
                default=Path('.', 'output.csv'))
@click.option('--tempdir',
              type=click.Path(exists=True),
              default=None)
@click.option('--proba_threshold',
              type=float,
              default=None)
@click.option('--model_path',
              type=click.Path(exists=True,
                              file_okay=True,
                              dir_okay=True),
              default=default_model_dir)
@click.option('--model_class',
              type=str,
              default="winning")
@click.option('--verbose', type=bool, default=True)
def predict(data_path, pred_path, tempdir, proba_threshold, model_path, model_class, verbose):
    """

    Args:
        data_path:
        pred_path:
        tempdir:
        proba_threshold:
        model_path:
        model_class:
        verbose:

    Returns:

    """

    if verbose:
        click.echo(f"Using data_path:\t{data_path}")
        click.echo(f"Using pred_path:\t{pred_path}")

    # Load the model into manager
    manager = ModelManager(model_path=model_path,
                           model_class=model_class,
                           data_path=data_path,
                           proba_threshold=proba_threshold,
                           tempdir=tempdir)

    # Make predictions, return a DataFrame
    manager.predict()


@main.command()
@click.argument('data_path',
                type=click.Path(exists=True,
                                file_okay=True,
                                dir_okay=True))
@click.argument('labels',
                type=click.File())
@click.option('--tempdir',
              type=click.Path(exists=True))
@click.option('--batch_size',
              default=8,
              type=click.IntRange(min=1,
                                  max=256,
                                  clamp=True))
@click.option('--weights_out', type=click.Path(exists=True))
def tune(data_path, labels, tempdir, batch_size, weights_out):
    """

    Args:
        data_path:
        labels:
        tempdir:
        batch_size:
        weights_out:

    Returns:

    """
    pass


@main.command()
@click.argument('data_path',
                type=click.Path(exists=True,
                                file_okay=True,
                                dir_okay=True))
@click.argument('labels',
                type=click.File())
@click.option('--tempdir', type=click.Path(exists=True))
def train(data_path, labels, tempdir):
    """

    Args:
        data_path:
        labels:
        tempdir:

    Returns:

    """
    pass
