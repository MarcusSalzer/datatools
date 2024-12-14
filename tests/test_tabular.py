from unittest import TestCase, main
import polars as pl

import data_tools.tabular as tabular


class TestValueCountsDict(TestCase):
    def test_a(self):
        s = pl.Series("name", ["c", "b", "b", "a"])
        self.assertDictEqual(
            {"b": 2, "a": 1, "c": 1},
            tabular.value_counts_dict(s),
        )

    def test_ints(self):
        s = pl.Series("count", [4, 4, 7, 4])
        self.assertDictEqual(
            {4: 3, 7: 1},
            tabular.value_counts_dict(s),
        )


if __name__ == "__main__":
    main()
