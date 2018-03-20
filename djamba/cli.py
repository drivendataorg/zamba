from pathlib import Path

import click

# from djamba.models.winning_model import WinningModel
from djamba.models.io import load_model

default_model_dir = Path('models', 'assets')


@click.group()
def main():
    pass


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
                              file_okay=True,
                              dir_okay=True),
              default=default_model_dir)
@click.option('--sample_model',
              type=bool,
              default=False)
@click.option('--verbose', type=bool, default=True)
def predict(datapath, predsout, tmpdir, proba_threshold, modelpath, sample_model, verbose):

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
        click.echo(f"Using predpath:\t{predsout}")

    # Process the data

    # Load the model
    model = load_model(modelpath, sample_model)

    # Make predictions, return a DataFrame
    preds = model.predict()

    # Output for now
    click.echo(preds)


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
