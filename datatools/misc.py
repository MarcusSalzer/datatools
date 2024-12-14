import numpy as np
import polars as pl


class Scaler:
    def __init__(self, axis=0) -> None:
        self.axis = axis

    def standardize(self, X: np.ndarray | pl.DataFrame):
        self.mu = X.mean(axis=self.axis, keepdims=True)
        self.std = X.mean(axis=self.axis, keepdims=True)

        X_standard = (X - self.mu) / self.std

        return X_standard

    def transform(self):
        pass

    # TODO

    def inverse(self, X_standard: np.ndarray | pl.DataFrame):
        return X_standard * self.std + self.mu
