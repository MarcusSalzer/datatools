"""Tabular data analysis"""

import polars as pl


def value_counts_dict(series: pl.Series, verbose=False) -> dict:
    """Count occurences of each unique value in a pl.Series

    ## returns
    - a dict of `value : count` pairs, sorted descending
    """
    cc_name = "_count"
    vc = {
        r[0]: r[1]
        for r in series.value_counts(name=cc_name)
        .sort(cc_name, series.name, descending=True)
        .rows()
    }
    if verbose:
        print(
            f"{len(vc)} unique ({series.name}): ",
            ", ".join([repr(k) for k in list(vc.keys())[:5]]),
            ",...",
        )

    return vc
