"""Extract data from SQL -> parquet (chunk by chunk so RAM is never exhausted)."""
import pandas as pd
from sqlalchemy import text

from . import config

_DOWNCAST = {
    "Week": "int8", "Agency_ID": "int32", "Channel_ID": "int16", "Route_SAK": "int32",
    "Customer_ID": "int64", "Product_ID": "int32", "id": "int32",
    "Sales_Units_Today": "int32", "Returns_Units_Next": "int32",
    "Adjusted_Demand_Units": "int32",
    "Sales_Today": "float32", "Returns_Next": "float32",
}


def build_query(table: str, columns: list[str], sample_mod: int) -> str:
    cols = ", ".join(columns)
    where = f"WHERE Customer_ID % {sample_mod} = 0" if sample_mod > 1 else ""
    return f"SELECT {cols} FROM {table} {where}"


def extract_table(engine, table, columns, out_path, sample_mod=1, chunksize=250_000):
    """Read a SQL table in chunks, downcast dtypes, save as parquet."""
    sql = build_query(table, columns, sample_mod)
    print(f"SQL: {sql}")
    parts, n = [], 0
    with engine.connect().execution_options(stream_results=True) as conn:
        for chunk in pd.read_sql(text(sql), conn, chunksize=chunksize):
            chunk = chunk.astype({c: t for c, t in _DOWNCAST.items() if c in chunk.columns})
            parts.append(chunk)
            n += len(chunk)
            print(f"  rows read: {n:,}", end="\r")
    df = pd.concat(parts, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df):,} rows -> {out_path}")
    return df


def load_train() -> pd.DataFrame:
    return pd.read_parquet(config.TRAIN_SAMPLE_PATH)


def load_test() -> pd.DataFrame:
    return pd.read_parquet(config.TEST_SAMPLE_PATH)
