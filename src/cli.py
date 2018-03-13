from pathlib import Path

import click
import pandas as pd

# from src.models.winning_model import WinningModel
from src import config
from src.src.models.io import load_model

# This is the main click group


@click.group()
def main():
    pass


# ######### PREDICT #########
# this is the predict command
@main.command()
@click.argument('datapath',
                type=click.Path(exists=True),
                default=Path('.').resolve())
@click.argument('predsout',
                type=click.Path(),
                default=Path('.', 'output.csv').resolve())
@click.option('--tmpdir',
              type=click.Path(exists=True),
              default=None)
@click.option('--proba_threshold',
              type=float,
              default=None)
@click.option('--modelpath',
              type=click.Path(exists=True,
                              file_okay=False,
                              dir_okay=True),
              default=config.default_model_dir)
@click.option('--verbose', type=bool, default=True)
def predict(datapath, predsout, tmpdir, proba_threshold, modelpath, verbose):

    datapath = Path(datapath)
    predsout = Path(predsout)

    if not predsout.exists():
        if predsout.suffix == '.csv':
            predsout.parent.mkdir(parents=True, exist_ok=True)
        else:
            predsout.mkdir(parents=True, exist_ok=True)
            predsout = Path(predsout, 'output.csv')
    else:
        if predsout.is_dir():
            predsout = Path(predsout, 'output.csv')

    if verbose:
        click.echo(f"Using datapath:\t{datapath}")
        click.echo(f"Using prepath:\t{predsout}")

    # Process the data
    # TODO process data in 'datapath', store results in 'tmpdir'

    # Load the model
    model = load_model(modelpath)

    # Make predictions, return a DataFrame
    if proba_threshold is not None:

        # binary labels if threshold given
        preds = model.predict_proba(tmpdir) >= proba_threshold

    else:
        # probability if threshold not given
        preds = model.predict_proba(tmpdir)

    # Save the result
    assert isinstance(preds, pd.DataFrame)
    # TODO check if predpath is file or dir and save appropriately
    preds.to_csv(
        Path(predsout).resolve(),
        index_label='id',
    )

    # Output for now
    click.echo(preds)


# ######### TUNE #########
# this is the tune command
# TODO still not sure how tune will differ from fit
@main.command()
@click.argument('datapath',
                type=click.Path(exists=True,
                                file_okay=True,
                                dir_okay=True))
@click.argument('labels',
                type=click.File())
@click.option('--tmpdir',
              type=click.Path(exists=True))
@click.option('--batchsize',
              default=8,
              type=click.IntRange(min=1,
                                  max=256,
                                  clamp=True))
@click.option('--weights_out', type=click.Path(exists=True))
def tune(datapath, labels):
    pass


# ######### TRAIN #########
# this is the train command
@main.command()
@click.argument('datapath',
                type=click.Path(exists=True,
                                file_okay=True,
                                dir_okay=True))
@click.argument('labels',
                type=click.File())
@click.option('--tmpdir', type=click.Path(exists=True))
def train():
    pass
