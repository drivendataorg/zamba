from datetime import datetime
from pathlib import Path
import os
from typing import Any, Optional, Tuple
import yaml

from loguru import logger
from pydantic import BaseModel
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import yolox
from yolox.exp import Exp
import yolox.exp as yolox_exp
import yolox.utils as utils


class YoloXArgs(BaseModel):
    """Args for commandline training of yolox from:
    train: https://github.com/Megvii-BaseDetection/YOLOX/blob/68408b4083f818f50aacc29881e6f97cd19fcef2/tools/train.py#L18-L96
    eval: https://github.com/Megvii-BaseDetection/YOLOX/blob/68408b4083f818f50aacc29881e6f97cd19fcef2/tools/eval.py#L29-L111
    """

    name: str  # model name to start from (e.g., yolox-s, yolox-nano, yolox-tiny)
    experiment_name: str = None
    dist_backend: str = "nccl"
    dist_url: str = None
    batch_size: int = 64
    devices: int = None
    # exp_file: str = None  # NOT CURRENTLY SUPPORTED
    resume: bool = False
    ckpt: Path = None
    start_epoch: int = None
    num_machines: int = 1
    machine_rank: int = 0
    fp16: bool = False
    cache: bool = False
    occupy: bool = False
    logger: str = "tensorboard"
    conf: float = None
    nms: float = None
    tsize: int = None  # TODO: use this for test size
    seed: int = None
    fuse: bool = False
    trt: bool = False
    legacy: bool = False
    test: bool = False
    speed: bool = False


class YoloXExp(BaseModel):
    # just the pieces that we want to be able to override from
    #   https://github.com/Megvii-BaseDetection/YOLOX/blob/68408b4083f818f50aacc29881e6f97cd19fcef2/yolox/exp/yolox_base.py
    #
    #  See the above link for more detail on these options
    #  Missing options were intentionally omitted because they are overridden when loading a yolo-* experiment from:
    #   https://github.com/Megvii-BaseDetection/YOLOX/tree/68408b4083f818f50aacc29881e6f97cd19fcef2/exps/default

    # model config
    num_classes: int

    # dataloader config
    data_num_workers: int = 4
    data_dir: str = None
    train_ann: str = None
    val_ann: str = None
    test_ann: str = None

    # training config
    warmup_epochs: int = 5
    max_epoch: int = 300
    warmup_lr: float = 0.0
    min_lr_ratio: float = 0.05
    basic_lr_per_img: float = 0.01 / 64.0  # TODO: can we use batch size dynamically? 64 is default
    scheduler: str = "yoloxwarmcos"
    no_aug_epochs: int = 15
    ema: bool = True

    weight_decay: float = 5e-4
    momentum: float = 0.9
    print_interval: int = 10
    eval_interval: int = 10
    save_history_ckpt: bool = True

    # test config
    test_conf: float = 0.01
    nmsthre: float = 0.65


# TODO: is this too much hardcoding?
class TinyExp(Exp):
    # default tiny exp
    def __init__(self):
        super(TinyExp, self).__init__()
        self.depth = 0.33
        self.width = 0.375
        self.scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (416, 416)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.enable_mixup = False


class YoloXModel:
    def __init__(self, exp: YoloXExp, args: YoloXArgs, gpus: int = None, image_size: int = None):
        utils.configure_module()

        # load base tiny exp
        base_exp = TinyExp()

        # override the yolo experiment default settings with any
        # ones that we pass in
        for k, v in exp.dict().items():
            setattr(base_exp, k, v)

        if not args.experiment_name:
            args.experiment_name = base_exp.exp_name

        self.exp = base_exp
        self.args = args

        gpus = gpus or args.devices
        self.num_gpu = utils.get_num_devices() if gpus is None else gpus
        assert self.num_gpu <= utils.get_num_devices()

        if image_size is not None:
            self.exp.input_size = (image_size, image_size)

        # TODO: do we need this?
        if self.args.tsize is not None:
            self.exp.test_size = (self.args.tsize, self.args.tsize)


    @classmethod
    def load(
        cls,
        checkpoint: os.PathLike,
        model_kwargs_path: os.PathLike,
        *args,
        **kwargs,
    ):
        model_kwargs = yaml.safe_load(Path(model_kwargs_path).read_text())
        model_kwargs["ckpt"] = checkpoint

        exp_dict = dict()
        args_dict = dict()

        # parse out which fields go to YoloXExp and which go to YoloXArgs
        if model_kwargs is not None:
            for k in model_kwargs.keys():
                if k in YoloXArgs.__fields__.keys():
                    args_dict[k] = model_kwargs[k]
                else:
                    exp_dict[k] = model_kwargs[k]

        return cls(
            YoloXExp(**exp_dict),
            YoloXArgs(
                name=model_kwargs["backbone"],
                **args_dict,
            ),
            *args,
            **kwargs,
        )
