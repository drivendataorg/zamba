import logging
from pathlib import Path

import click

from zamba.models.manager import ModelManager

default_model_dir = Path(__file__).parent / 'models' / 'assets'
default_model_dir.mkdir(exist_ok=True)


@click.group()
def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


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
              default=None,
              help="Path to temporary directory. If not specified, OS temporary directory is used.")
@click.option('--proba_threshold',
              type=float,
              default=None,
              help="Probability threshold for classification. if specified binary predictions are returned with 1 "
                   "being greater than the threshold, 0 being less than or equal to. If not specified, probabilities "
                   "between 0 and 1 are returned.")
@click.option('--output_class_names',
              is_flag=True,
              default=False,
              help="If True, we just return a video and the name of the most likely class. If False, "
                   "we return a probability or indicator (depending on --proba_threshold) for every "
                   "possible class.")
@click.option('--model_path',
              type=click.Path(exists=True,
                              file_okay=True,
                              dir_okay=True),
              default=default_model_dir,
              help="Path to model files to be loaded into model object.")
@click.option('--model_class',
              type=str,
              default="cnnensemble",
              help="Class of model, controls whether or not sample model is used.")
@click.option('--verbose',
              type=bool,
              default=True,
              help="Controls verbosity of the command line predict function.")
def predict(data_path, pred_path, tempdir, proba_threshold, output_class_names, model_path, model_class, verbose):
    """Identify species in a video.

      This is a command line interface for prediction on camera trap footage. Given a path to camera trap footage,
      the predict function use a deep learning model to predict the presence or absense of a variety of species of
      common interest to wildlife researchers working with camera trap data.

    """

    if verbose:
        click.echo(f"Using data_path:\t{data_path}")
        click.echo(f"Using pred_path:\t{pred_path}")

    # Load the model into manager
    manager = ModelManager(model_path=model_path,
                           model_class=model_class,
                           proba_threshold=proba_threshold,
                           tempdir=tempdir,
                           output_class_names=output_class_names)

    # Make predictions, return a DataFrame
    manager.predict(data_path, pred_path=pred_path, save=True)


@main.command()
@click.argument('data_path',
                type=click.Path(exists=True,
                                file_okay=True,
                                dir_okay=True))
@click.argument('labels',
                type=click.File())
@click.option('--tempdir',
              type=click.Path(exists=True),
              help="Path to temporary directory. If not specified, OS temporary directory is used")
@click.option('--batch_size',
              default=8,
              type=click.IntRange(min=1,
                                  max=256,
                                  clamp=True),
              help="Batch size for model inputs, defaults to 8 samples.")
@click.option('--weights_out',
              type=click.Path(exists=True),
              help="Output path for saved weights.")
def tune(data_path, labels, tempdir, batch_size, weights_out):
    """ [NOT IMPLEMENTED] Update network with new data.

        Finetune the network for a different task by keeping the
        trained weights, replacing the top layer with one that outputs
        the new classes, and re-training for a few epochs to have the
        model output the new classes instead.

    """
    click.echo("Finetuning an algorithm to new data is not yet implemented.")


@main.command()
@click.argument('data_path',
                type=click.Path(exists=True,
                                file_okay=True,
                                dir_okay=True))
@click.argument('labels',
                type=click.File())
@click.option('--tempdir',
              type=click.Path(exists=True),
              help="Path to temporary directory. If not specified, OS temporary directory is used")
def train(data_path, labels, tempdir):
    """ [NOT IMPLEMENTED] Retrain network from scratch.

        Train the weights from scratch using
        the provided data_path and labels.
    """
    click.echo("Training an algorithm from scratch on new data is not yet implemented.")
