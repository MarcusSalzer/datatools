import numpy as np
import polars as pl
import regex as re


def normalize_names(
    names: list[str],
    wordlength: int | None = None,
    maxwords: int | None = None,
):
    result = []
    for n in names:
        words = re.split("[ \|,\._-]", n.lower().strip())

        if wordlength is not None:
            words = [w[:wordlength] for w in words]
        if maxwords is not None:
            words = words[:maxwords]

        n = "_".join(words)
        result.append(n)
    if len(set(result)) != len(result):
        raise ValueError("processed names not unique")

    return result


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
