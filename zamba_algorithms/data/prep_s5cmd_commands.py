from cloudpathlib import S3Path
import pandas as pd


df = pd.read_csv("data/processed/unified_metadata.csv", low_memory=False)
df["local_path"] = df.filepath.apply(lambda x: S3Path(x).key)
df["cp"] = "cp"

# add strings around files to deal with spaces and commas in filenames
df["filepath"] = '"' + df.filepath + '"'
df["local_path"] = '"' + df.local_path + '"'

df[["cp", "filepath", "local_path"]].drop_duplicates().to_csv(
    "commands.txt",
    sep=" ",
    header=None,
    index=False,
)
