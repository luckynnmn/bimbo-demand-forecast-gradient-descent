"""STEP 1: check the SQL connection and count the rows of both tables."""
from sqlalchemy import text

from bimbo_gd import config
from bimbo_gd.db import get_engine, ping


def main():
    engine = get_engine()
    print(f"Dialect: {config.DB_DIALECT} | Database: {config.DB_NAME or config.DB_SQLITE_PATH}")
    print("SELECT 1 ->", ping(engine), "=> Connection OK")
    with engine.connect() as conn:
        for t in (config.FILTERED_TRAIN_TABLE, config.FILTERED_TEST_TABLE):
            n = conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
            print(f"  {t}: {n:,} rows")


if __name__ == "__main__":
    main()
