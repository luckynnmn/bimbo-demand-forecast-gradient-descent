"""Create a SMALL SYNTHETIC SQLite database with the same tables and columns as the real project
(train_filtered, test_filtered). It lets anyone (e.g. a peer reviewer) run the whole pipeline end to end
WITHOUT the Kaggle data. The numbers it produces are NOT real results.

    python tools/make_demo_db.py            # creates data/demo.db
Then copy .env.demo to .env and run the notebooks.
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PRODUCTS = [2425, 1284, 1278, 43285, 36610, 1240, 1250, 2233, 1242, 45143]


def main(n_train=150_000, n_test=20_000, seed=0):
    rng = np.random.default_rng(seed)
    n_cust = 3000
    cust = rng.integers(1, n_cust, n_train)
    prod = rng.choice(PRODUCTS, n_train)
    week = rng.integers(3, 10, n_train)
    prod_level = dict(zip(PRODUCTS, np.exp(rng.normal(0.8, 0.5, len(PRODUCTS)))))
    cust_level = np.exp(rng.normal(0, 0.6, n_cust + 1))
    base = np.array([prod_level[p] for p in prod]) * cust_level[cust]
    y = np.maximum(0, rng.poisson(base * 3))
    sales = y + rng.poisson(0.4, n_train)
    train = pd.DataFrame(dict(Week=week, Agency_ID=cust % 40, Channel_ID=cust % 5 + 1, Route_SAK=cust % 300,
                              Customer_ID=cust, Product_ID=prod, Sales_Units_Today=sales,
                              Sales_Today=sales * rng.uniform(8, 30, n_train), Returns_Units_Next=sales - y,
                              Returns_Next=(sales - y) * 10.0, Adjusted_Demand_Units=y))
    c2 = rng.integers(1, n_cust + 200, n_test)
    test = pd.DataFrame(dict(id=range(n_test), Week=rng.integers(10, 12, n_test), Agency_ID=c2 % 40, Channel_ID=c2 % 5 + 1,
                             Route_SAK=c2 % 300, Customer_ID=c2, Product_ID=rng.choice(PRODUCTS, n_test)))
    path = ROOT / "data" / "demo.db"
    path.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(path)
    train.to_sql("train_filtered", con, index=False, if_exists="replace")
    test.to_sql("test_filtered", con, index=False, if_exists="replace")
    con.close()
    print(f"Created {path}: train_filtered {len(train):,} rows, test_filtered {len(test):,} rows (synthetic)")


if __name__ == "__main__":
    main()
