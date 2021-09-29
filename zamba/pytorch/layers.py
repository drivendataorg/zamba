import torch


def _stack_tups(tuples, stack_dim=1):
    """Stack tuple of tensors along `stack_dim`

    NOTE: vendored (with minor adaptations) from fastai:
    https://github.com/fastai/fastai/blob/4b0785254fdece1a44859956b6e54eedb167a97e/fastai/layers.py#L505-L507

    Updates:
        -  use `range` rather than fastai `range_of`
    """
    return tuple(torch.stack([t[i] for t in tuples], dim=stack_dim) for i in range(len(tuples[0])))


class TimeDistributed(torch.nn.Module):
    """Applies `module` over `tdim` identically for each step, use `low_mem` to compute one at a time.

    NOTE: vendored (with minor adaptations) from fastai:
    https://github.com/fastai/fastai/blob/4b0785254fdece1a44859956b6e54eedb167a97e/fastai/layers.py#L510-L544

    Updates:
     - super.__init__() in init
     - assign attributes in init
     - inherit from torch.nn.Module rather than fastai.Module
    """

    def __init__(self, module, low_mem=False, tdim=1):
        super().__init__()
        self.low_mem = low_mem
        self.tdim = tdim
        self.module = module

    def forward(self, *tensors, **kwargs):
        "input x with shape:(bs,seq_len,channels,width,height)"
        if self.low_mem or self.tdim != 1:
            return self.low_mem_forward(*tensors, **kwargs)
        else:
            # only support tdim=1
            inp_shape = tensors[0].shape
            bs, seq_len = inp_shape[0], inp_shape[1]
            out = self.module(*[x.view(bs * seq_len, *x.shape[2:]) for x in tensors], **kwargs)
        return self.format_output(out, bs, seq_len)

    def low_mem_forward(self, *tensors, **kwargs):
        "input x with shape:(bs,seq_len,channels,width,height)"
        seq_len = tensors[0].shape[self.tdim]
        args_split = [torch.unbind(x, dim=self.tdim) for x in tensors]
        out = []
        for i in range(seq_len):
            out.append(self.module(*[args[i] for args in args_split]), **kwargs)
        if isinstance(out[0], tuple):
            return _stack_tups(out, stack_dim=self.tdim)
        return torch.stack(out, dim=self.tdim)

    def format_output(self, out, bs, seq_len):
        "unstack from batchsize outputs"
        if isinstance(out, tuple):
            return tuple(out_i.view(bs, seq_len, *out_i.shape[1:]) for out_i in out)
        return out.view(bs, seq_len, *out.shape[1:])

    def __repr__(self):
        return f"TimeDistributed({self.module})"
