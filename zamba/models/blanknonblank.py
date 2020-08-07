import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.utils import get_file

import zamba


class BlankNonBlank():
    MODEL_URL = "https://drivendata-public-assets.s3.amazonaws.com/zamba-and-obj-rec-0.859.joblib"
    FEATURE_NAMES = [
        "n_key_frames", "h", "w", "n_detections", "total_area", "bird", "cattle", "chimpanzee", "elephant",
        "forest buffalo", "gorilla", "hippopotamus", "human", "hyena", "large ungulate", "leopard", "lion",
        "other (non-primate)", "other (primate)", "pangolin", "porcupine", "reptile", "rodent", "small antelope",
        "small cat", "wild dog", "duiker", "hog", "blank",
    ]

    def __init__(
        self,
        model_path=None,
    ):
        if model_path is None:
            model_path = self._get_model()

        self.model = self.load_model(str(model_path))

    def fit(self, X, y):
        model = GridSearchCV(
            GradientBoostingClassifier(),
            {
                "loss": ["exponential", "deviance"],
                "learning_rate": [0.001, 0.01, 0.1, 0.5],
                "n_estimators": [100, 200, 300],
                "subsample": [0.10, 0.25, 0.5],
                "min_samples_split": [5, 10, 25],
                "max_depth": [3, 5, 10],
                "random_state": [1337],
            },
            verbose=2,
            n_jobs=-1,
            return_train_score=True,
        )

        model.fit(X, y)
        return model

    def prepare_features(self, features):
        """Produces a feature array with the correct column ordering

        Args:
            features (pd.DataFrame): A table of features containing the columns in `BlankNonBlank.FEATURE_NAMES`

        Returns:
            np.ndarray: An array of features with shape (num samples, num features)
        """
        return features[self.FEATURE_NAMES].values

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def load_model(self, model_path):
        """
        Load a model from a pickled file
        """
        with open(model_path, "rb") as f:
            model = joblib.load(f)

        return model

    def _get_model(self, file_name="zamba-and-obj-rec-0.859.joblib", model_url=None):
        if model_url is None:
            model_url = self.MODEL_URL

        cache_subdir = "blanknonblank"
        model_path = zamba.config.cache_dir / cache_subdir / file_name

        if not model_path.exists():
            model_path = get_file(
                fname=file_name,
                origin=model_url,
                cache_dir=zamba.config.cache_dir,
                cache_subdir=cache_subdir,
                extract=True,
            )

        return model_path
