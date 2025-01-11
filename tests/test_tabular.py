from unittest import TestCase, main
import polars as pl

import datatools.tabular as dttab


class TestValueCountsDict(TestCase):
    def test_a(self):
        s = pl.Series("name", ["c", "b", "b", "a"])
        self.assertDictEqual(
            {"b": 2, "a": 1, "c": 1},
            dttab.value_counts(s, sort_by="count", as_dict=True),
            "sorted by count",
        )
        self.assertDictEqual(
            {"a": 1, "b": 2, "c": 1},
            dttab.value_counts(s, sort_by="alpha", as_dict=True),
            "sorted alphabetically",
        )

    def test_ints(self):
        s = pl.Series("count", [4, 4, 7, 4])
        self.assertDictEqual(
            {4: 3, 7: 1},
            dttab.value_counts(s, sort_by="count", as_dict=True),
        )


if __name__ == "__main__":
    main()
