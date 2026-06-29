import inspect
import os
from typing import Any, Dict, Optional, Tuple, Type

import pytorch_lightning as pl
import torch

from zamba.settings import INFERENCE_SEED


def filter_scheduler_params(
    scheduler_cls: Type,
    params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return scheduler kwargs supported by the scheduler constructor.

    Configs and checkpoints may include deprecated args (e.g. ``verbose`` was removed
    from PyTorch lr schedulers in 2.7).
    """
    if not params:
        return {}
    signature = inspect.signature(scheduler_cls.__init__)
    supported = set(signature.parameters) - {"self", "optimizer"}
    return {key: value for key, value in params.items() if key in supported}


def configure_inference_determinism(
    *,
    seed: Optional[int] = None,
    deterministic: bool = False,
) -> None:
    """Seed RNGs for inference and optionally enable strict GPU determinism.

    Seeding always runs so frame sampling and other stochastic preprocessing are
    reproducible. When ``deterministic`` is True, also request deterministic CUDA/cuDNN
    algorithms (best effort; may reduce GPU throughput).

    Args:
        seed: Random seed for Python, NumPy, and PyTorch. Defaults to ``INFERENCE_SEED``
            from settings (env ``INFERENCE_SEED``, default 55).
        deterministic: If True, enable strict deterministic CUDA/cuDNN algorithms where
            supported (best effort; some GPU ops may remain non-deterministic and will warn
            rather than error). Disables cuDNN benchmark mode. May reduce GPU throughput.
    """
    effective_seed = INFERENCE_SEED if seed is None else seed
    pl.seed_everything(effective_seed, workers=True)

    if deterministic:
        # cuBLAS needs this for reproducible GEMMs; must be set before CUDA init.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_multilayer_perceptron(
    input_size: int,
    hidden_layer_sizes: Optional[Tuple[int]],
    output_size: int,
    activation: Optional[torch.nn.Module] = torch.nn.ReLU,
    dropout: Optional[float] = None,
    output_dropout: Optional[float] = None,
    output_activation: Optional[torch.nn.Module] = None,
) -> torch.nn.Sequential:
    """Builds a multilayer perceptron.

    Args:
        input_size (int): Size of first input layer.
        hidden_layer_sizes (tuple of int, optional): If provided, size of hidden layers.
        output_size (int): Size of the last output layer.
        activation (torch.nn.Module, optional): Activation layer between each pair of layers.
        dropout (float, optional): If provided, insert dropout layers with the following dropout
            rate in between each pair of layers.
        output_dropout (float, optional): If provided, insert a dropout layer with the following
            dropout rate before the output.
        output_activation (torch.nn.Module, optional): Activation layer after the final layer.
    Returns:
        torch.nn.Sequential
    """
    if (hidden_layer_sizes is None) or len(hidden_layer_sizes) == 0:
        return torch.nn.Linear(input_size, output_size)

    layers = [torch.nn.Linear(input_size, hidden_layer_sizes[0])]
    if activation is not None:
        layers.append(activation())

    if (dropout is not None) and (dropout > 0):
        layers.append(torch.nn.Dropout(dropout))

    for in_size, out_size in zip(hidden_layer_sizes[:-1], hidden_layer_sizes[1:]):
        layers.append(torch.nn.Linear(in_size, out_size))
        if activation is not None:
            layers.append(activation())

        if (dropout is not None) and (dropout > 0):
            layers.append(torch.nn.Dropout(dropout))

    layers.append(torch.nn.Linear(hidden_layer_sizes[-1], output_size))

    if (output_dropout is not None) and (output_dropout > 0):
        layers.append(torch.nn.Dropout(dropout))

    if output_activation is not None:
        layers.append(output_activation())

    return torch.nn.Sequential(*layers)
