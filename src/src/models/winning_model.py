from pathlib import Path
import tempfile


import pandas as pd
import tensorflow as tf

from .model import Model


class WinningModel(Model):
    def __init__(self, prop1=None):
        self.prop1 = prop1
