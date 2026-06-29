import torch

from zamba.pytorch.utils import filter_scheduler_params


def test_filter_scheduler_params_drops_unsupported_kwargs():
    params = {"milestones": [3], "gamma": 0.5, "verbose": True}
    filtered = filter_scheduler_params(torch.optim.lr_scheduler.MultiStepLR, params)

    assert filtered == {"milestones": [3], "gamma": 0.5}
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=0.1),
        **filtered,
    )
    assert scheduler.gamma == 0.5


def test_filter_scheduler_params_returns_empty_dict_for_none():
    assert filter_scheduler_params(torch.optim.lr_scheduler.MultiStepLR, None) == {}
