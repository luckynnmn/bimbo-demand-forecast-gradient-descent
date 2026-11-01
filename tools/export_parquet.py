import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import text
from bimbo_gd.db import get_engine

OUT = Path("data/export"); OUT.mkdir(parents=True, exist_ok=True)
engine = get_engine()

for table, fname in [("dbo.train_filtered", "train_filtered.parquet"),
                     ("dbo.test_filtered",  "test_filtered.parquet")]:
    path, writer, n = OUT / fname, None, 0
    with engine.connect() as con:
        for chunk in pd.read_sql(text(f"SELECT * FROM {table}"), con, chunksize=500_000):
            t = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, t.schema, compression="zstd")
            writer.write_table(t)
            n += len(chunk)
            print(f"{table}: {n:,} rows written", end="\r")
    writer.close()
    print(f"\n{table} -> {path}  ({path.stat().st_size/1e6:.1f} MB)")