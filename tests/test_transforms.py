import torch
from zamba.pytorch.transforms import PadDimensions


def test_pad_dimensions():
    # do not change size of None dimensions
    pad = PadDimensions((None, 2))
    x = torch.randn(3, 1)
    padded_x = pad(x)
    assert padded_x.shape == torch.Size([3, 2])
    assert (pad(x)[:, 1:] == x).all()

    # pad a few more dimensions
    pad = PadDimensions((None, 5, None, 7))
    x = torch.randn(2, 3, 4, 5)
    padded_x = pad(x)
    assert padded_x.shape == torch.Size([2, 5, 4, 7])
    assert (padded_x[:, 1:-1, :, 1:-1] == x).all()

    # do not change sizes for if dimension is larger than requested
    pad = PadDimensions((None, 5, None, 4))
    x = torch.randn(2, 3, 4, 5)
    padded_x = pad(x)
    assert padded_x.shape == torch.Size([2, 5, 4, 5])
    assert (padded_x[:, 1:-1, :, :] == x).all()
