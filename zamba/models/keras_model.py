from pathlib import Path

from zamba.models.model import Model

try:
    from tensorflow import keras
except ImportError:
    msg = "Zamba must have tensorflow installed, run either `pip install zamba[cpu]` "\
          "or `pip install zamba[gpu]` depending on what is available on your system."
    raise ImportError(msg)


class KerasModel(Model):
    def __init__(self, tempdir=None, model=None, model_save_path=None):
        super().__init__(tempdir=tempdir)
        self.model = model
        self.model_save_path = model_save_path

    @classmethod
    def from_disk(cls, path):
        model = keras.models.load_model(Path(path))
        return cls(model=model)

    def to_disk(self, path=None):
        """Save the model to specified path.
        If no path is passed, tries to use model_save_path attribute.
        """

        # save to user-specified, or model's path
        if path is not None:
            save_path = Path(path)
        elif self.model_save_path is not None:
            save_path = Path(self.model_save_path)
        else:
            raise FileNotFoundError("Must provide save_path")

        # create if necessary
        save_path.parent.mkdir(exist_ok=True)
        # keras' save
        self.model.save(save_path, include_optimizer=False)

    def load_data(self):
        pass

    def train(self):
        self.model.fit()

    def predict(self, data):
        self.model.predict()
