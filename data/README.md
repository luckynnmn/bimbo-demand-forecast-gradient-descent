# Data: source, access and limits

**Source.** Kaggle competition *Grupo Bimbo Inventory Demand* (https://www.kaggle.com/c/grupo-bimbo-inventory-demand). A free Kaggle account and acceptance of the competition rules are needed to download the original files (`train.csv` about 3 GB, `test.csv`). The data are used here only for this course assignment.

**What is in this repository.** Two small files, the **result** of the SQL cleaning and filtering done by the author (10 best-selling products):

| File | Rows | Content |
|---|---|---|
| `export/train_filtered.parquet` | 13,721,629 | weeks 3-9 with the target `Adjusted_Demand_Units` |
| `export/test_filtered.parquet` | 169,578 | weeks 10-11 without the target |

**What one row is.** The delivery of one product to one store in one week through one depot, route and sales channel.

**Columns.** `Week`, `Agency_ID`, `Channel_ID`, `Route_SAK`, `Customer_ID`, `Product_ID`, `Sales_Units_Today`, `Sales_Today`, `Returns_Units_Next`, `Returns_Next`, `Adjusted_Demand_Units` (target; training only). The test file has only the first six columns.

**How to use them.** Load both files into your own SQL Server with `python tools/import_parquet_to_sql.py` (it checks the row counts); see sections 2-3 of the main `README.md`. No credentials or private data are stored in the repository: the `.env` file with your server name is created locally and is not committed.

**Provenance.** The cleaning and filtering scripts are in `sql/` (`01_create_filtered_tables.sql`, `02_select_top10_products.sql`); `tools/export_parquet.py` shows how the Parquet files were made. You do not need to run them.

**Limits.** Identifiers are anonymized and there is no personal data. Only seven weeks of history exist (no seasonality), the test labels are hidden, and the data contain no promotions or weather. Use is restricted to the competition terms.
