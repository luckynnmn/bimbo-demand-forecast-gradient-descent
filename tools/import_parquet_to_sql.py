"""FOR VALIDATORS: create the two SQL tables from the Parquet files in data/export/ and check the row counts.

    python tools/import_parquet_to_sql.py
Needs: an empty database (README section 3.1) and a working .env (python test_connection.py must say 'Connection OK' - the
tables do not need to exist yet). Creates dbo.train_filtered and dbo.test_filtered (old copies are dropped), loads the
Parquet files in batches, then counts the rows IN SQL and compares them with the rows in the files.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq
from sqlalchemy import (BigInteger, Column, Float, Integer, MetaData, Table, inspect, text)

from bimbo_gd import config
from bimbo_gd.db import get_engine

DATA = config.ROOT / "data" / "export"
BATCH = 200_000

I, F, B = Integer, Float, BigInteger
COLUMNS = {
    "train": [("Week", I), ("Agency_ID", I), ("Channel_ID", I), ("Route_SAK", I), ("Customer_ID", B), ("Product_ID", I),
              ("Sales_Units_Today", I), ("Sales_Today", F), ("Returns_Units_Next", I), ("Returns_Next", F),
              ("Adjusted_Demand_Units", I)],
    "test": [("id", B), ("Week", I), ("Agency_ID", I), ("Channel_ID", I), ("Route_SAK", I), ("Customer_ID", B),
             ("Product_ID", I)],
}
JOBS = [("train", config.FILTERED_TRAIN_TABLE, DATA / "train_filtered.parquet"),
        ("test", config.FILTERED_TEST_TABLE, DATA / "test_filtered.parquet")]


def split_name(full):
    schema, _, name = full.rpartition(".")
    return (schema or None), name


def main():
    engine = get_engine(fast_executemany=True)
    ok = True
    for kind, full, path in JOBS:
        if not path.exists():
            sys.exit(f"File not found: {path}\nPut the two .parquet files in the data/export/ folder first (README section 3).")
        schema, name = split_name(full)
        expected = pq.ParquetFile(path).metadata.num_rows
        meta = MetaData()
        table = Table(name, meta, *[Column(c, t) for c, t in COLUMNS[kind]], schema=schema)
        with engine.begin() as con:
            table.drop(con, checkfirst=True)
            table.create(con)
        print(f"{full}: loading {expected:,} rows from {path.name} ...")
        t0, n = time.time(), 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH):
            df = batch.to_pandas()[[c for c, _ in COLUMNS[kind]]]
            df.to_sql(name, engine, schema=schema, if_exists="append", index=False, chunksize=50_000)
            n += len(df)
            print(f"   {n:,} / {expected:,} rows", end="\r")
        with engine.connect() as con:
            in_sql = con.execute(text(f"SELECT COUNT(*) FROM {full}")).scalar()
        status = "OK" if in_sql == expected else "MISMATCH"
        ok &= (in_sql == expected)
        print(f"\n{full}: {in_sql:,} rows in SQL, {expected:,} in the file -> {status} ({time.time() - t0:.0f} s)")
    if not ok:
        sys.exit("Row counts do not match - run the script again (it recreates the tables).")
    print("\nAll tables loaded. Next: python test_connection.py")


if __name__ == "__main__":
    main()
