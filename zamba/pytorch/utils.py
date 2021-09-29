from typing import Optional, Tuple

import torch


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
