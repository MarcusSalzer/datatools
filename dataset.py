from plotly import express as px
import polars as pl
from sklearn import preprocessing
import numpy as np
import scipy.sparse as sp

from typing import Literal


def plot_feature_pair(x_data, y_data, x=0, y=1, discrete=True):
    if discrete:
        labels = y_data.astype(str)
    else:
        labels = y_data

    fig = (
        px.scatter(
            x_data,
            x=x,
            y=y,
            color=labels,
            title="Feature space",
            width=500,
            height=400,
        )
        .update_traces(marker=dict(size=3))
        .update_layout(margin=dict(t=80, l=10, b=10, r=10))
    )
    return fig


class ScoreUtil:
    """A utility class for interpreting model performance."""

    # TODO
    def __init__(self) -> None:
        pass

    """IDEA: track performance for grid search.
    save all parameter combinations """


class DataSet:
    """Utility for organizing and preprocessing data."""

    def __init__(
        self,
        data: pl.DataFrame = None,
        targets: list[str] = [],
        ignore: list[str] = [],
        log: list[str] = [],
    ) -> None:
        self.df = data.select(pl.all().shrink_dtype())
        self._log = log
        self.features = self.df.drop("_split", *ignore, *targets).columns
        self.targets = targets
        self.ignore = ignore

        self.num_feats = list(
            filter(lambda k: self.df.schema[k] in pl.NUMERIC_DTYPES, self.features)
        )
        self.cat_feats = list(
            filter(lambda k: self.df.schema[k] == pl.String, self.features)
        )

    def __str__(self):
        return f"data: {self.df.shape}\nnum:{self.num_feats}\ncat:{self.cat_feats}"

    def print_log(self):
        """List operations performed on data"""
        print("\nDataset log:\n- " + "\n- ".join(self._log))

    def describe_features(self):
        schema = self.df.schema

        print("\nNumeric Features:")
        for f in self.num_feats:
            print(f"- {f} ({schema[f]})")
            print(f"    - {self.df[f].min()} -> {self.df[f].max()}\n")

        print("\nCategory Features:")
        for f in self.cat_feats:
            print(f"- {f} ({schema[f]})")
            cat_counts = self.df[f].value_counts()
            for v, c in cat_counts.iter_rows():
                print(f"    - {repr(v).ljust(12)} : {c}")
            print()

    def filter_categories(self, allowed: dict):
        """Remove categorical values not in allowed values (inplace).

        ## parameters
        - allowed (dict[list]): allowed categories for each feature ([] for allow all)
        """
        key_diff = set(self.cat_feats).symmetric_difference(allowed.keys())
        if key_diff:
            raise ValueError(f"Incorrect feature names {key_diff}")

        # remove non-allowed values
        self.df = self.df.with_columns(
            pl.when(pl.col(c).is_in(allowed[c]))
            .then(pl.col(c))
            .otherwise(None)
            .alias(c)
            for c in self.cat_feats
            if allowed[c]
        )
        self._log.append(
            f"filtered categories in {[c for c in allowed.keys() if allowed[c]]}"
        )

    def remove_rare_categories(self, threshold=None, quantile=0.05):
        """Remove rare categorical values (inplace)."""

        for c in self.cat_feats:
            value_counts = self.df[c].value_counts()
            if threshold:
                thr = threshold
            else:
                thr = int(self.df[c].value_counts()["count"].quantile(quantile))

            rare_values = value_counts.filter(pl.col("count") <= thr)[c].to_list()
            self.df = self.df.with_columns(
                pl.col(c).map_elements(
                    lambda x: None if x in rare_values else x, return_dtype=pl.String
                )
            )

            self._log.append(f"categories with <= {thr} samples removed from {c}")

    def get_encoded(
        self,
        verbose=True,
        encoding: Literal["ordinal", "one-hot"] = "ordinal",
        sample_ratio=1.0,
        min_frequency=10,
    ):
        """Encode all categoric features."""

        if len(self.targets) > 1:
            raise NotImplementedError("Only for 1d targets")

        if encoding == "ordinal":
            feat_encoder = preprocessing.OrdinalEncoder()
        elif encoding == "one-hot":
            feat_encoder = preprocessing.OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=min_frequency,
                sparse_output=False,
                drop="first",
            )

        feat_encoder.fit(self.df.select(self.cat_feats))

        self.target_enc = preprocessing.LabelEncoder().fit(self.df[self.targets[0]])

        splits = self.df["_split"].unique()

        if isinstance(sample_ratio, float):
            sample_ratio = dict.fromkeys(splits, sample_ratio)

        arrays = {}
        for s in splits:
            data_split = self.df.filter(pl.col("_split") == s)
            data_split = data_split[: int(sample_ratio[s] * data_split.shape[0])]
            cat_encoded = feat_encoder.transform(data_split.select(self.cat_feats))
            arrays[f"X_{s}"] = np.concatenate(
                [data_split.select(self.num_feats), cat_encoded], axis=1
            )
            arrays[f"Y_{s}"] = self.target_enc.transform(data_split[self.targets[0]])

        if verbose:
            print("\nArrays:")
            for k in arrays:
                print(f"- {k} ({arrays[k].dtype})")
                print(f"    - {arrays[k].shape}")

        return arrays

    def get_splits(self, verbose=True):
        """Return splits as separate X, Y dataframes"""
        splits = self.df["_split"].unique()
        frames = {}
        for s in splits:
            data_split = self.df.filter(pl.col("_split") == s)
            frames[f"X_{s}"] = data_split.select(self.features)
            frames[f"Y_{s}"] = data_split.select(self.targets)

        if verbose:
            print("\nDataframes:")
            for k in frames:
                print(f"- {k} {set(frames[k].dtypes)}")
                print(f"    - {frames[k].shape}")

        return frames

    @classmethod
    def from_splits(
        cls,
        splits: dict[str, pl.DataFrame],
        targets: list[str] = [],
        ignore: list[str] = [],
    ):
        log = []

        # fill missing columns
        for s in splits.keys():
            for t in targets:
                if t not in splits[s].columns:
                    splits[s] = splits[s].with_columns(pl.lit(None).alias(t))
                    log.append(f"added empty target: '{t}' in '{s}'")
            splits[s] = (
                splits[s]
                .select(sorted(splits[s].columns))
                .cast(
                    {
                        pl.INTEGER_DTYPES: pl.Int64,
                        pl.FLOAT_DTYPES: pl.Float64,
                    }
                )
            )

        # Ensure all provided splits have the same columns
        cols = list(splits.values())[0].columns
        for split_df in splits.values():
            if split_df.columns != cols:
                raise ValueError("All splits must have the same columns")

        # Add the split column to each DataFrame and concatenate them
        split_dfs = []
        for split_name, split_df in splits.items():
            split_df = split_df.with_columns(pl.lit(split_name).alias("_split"))
            split_dfs.append(split_df)

        log.append(f"create `DataSet` from splits: {list(splits.keys())}")

        return cls(
            data=pl.concat(split_dfs),
            targets=targets,
            ignore=ignore,
            log=log,
        )
