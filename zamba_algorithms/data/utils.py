from multiprocessing import cpu_count
import os
from typing import Dict, Union

from cloudpathlib import S3Path
import pandas as pd
from tqdm.contrib.concurrent import process_map
import typer

from zamba_algorithms.data.metadata import load_metadata
from zamba_algorithms.settings import DATA_DIRECTORY
from zamba_algorithms.data.video import num_frames

app = typer.Typer()


def get_n_frames(filepath: Union[str, os.PathLike]) -> Dict[str, str]:
    filepath = (
        S3Path(filepath)
        if isinstance(filepath, str) and filepath.startswith("s3://")
        else filepath
    )
    msg = None
    try:
        n = num_frames(filepath)
    except Exception as exc:
        n = None
        msg = str(exc)
    return {"filepath": str(filepath), "n": n, "message": msg}


@app.command()
def compute_num_frames(
    processes: int = cpu_count() - 1,
):
    metadata = load_metadata(zamba_label="original")
    results = process_map(get_n_frames, metadata.filepath, max_workers=processes)
    pd.DataFrame(results).to_csv(DATA_DIRECTORY / "interim" / "num_frames.csv", index=False)


@app.command()
def main():
    pass


if __name__ == "__main__":
    app()
