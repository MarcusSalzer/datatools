import polars as pl

train = pl.read_csv("data/train.csv")
test = pl.read_csv("data/test.csv")

train.write_parquet("data/train.parquet")
test.write_parquet("data/test.parquet")

train2 = pl.read_parquet("data/train.parquet")
test2 = pl.read_parquet("data/test.parquet")

print(train.equals(train2))
print(test.equals(test2))
