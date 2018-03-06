from pathlib import Path

# Path to project src
project_src = Path(__file__).absolute().parent.parent

# Path to model assets (graph, checkpoints)
models_dir = project_src / "src" / "models" / "assets"

# Name of test model
model_name = "test-model"

# Create new subdir for test model's files
model_subdir = Path(models_dir, model_name)
model_subdir.mkdir(exist_ok=True)

# Store global step, which will be baked into metagraph file name
global_step = 1000

# Store names of input layer tensors and operation (eg predict) to restore
# For the test model, we simply input w1, w2, add them, and multiply by bias:
# (w1 + w2) * b1
input_names = ["w1", "w2", "bias"]
op_to_restore_name = "op_to_restore"