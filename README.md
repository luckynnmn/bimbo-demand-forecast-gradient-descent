# Grupo Bimbo – Weekly Bakery Demand Forecast with Gradient Descent

MSBA 265 · University of the Pacific 
· Author: **NgocMaiNhi Nguyen (Lucky)** 
· Repository: **https://github.com/luckynnmn/bimbo-demand-forecast-gradient-descent**

**Question.** Can the weekly demand of a bakery product at a store be forecast from its own past orders, using a linear regression
whose weights are learned by **mini-batch Gradient Descent** (own NumPy implementation, `bimbo_gd/gradient_descent.py`)?
The target is `log1p(Adjusted_Demand_Units)`; the data come from the Kaggle competition *Grupo Bimbo Inventory Demand*.

## Result (validation week 9, never used to learn the weights)
| | Value |
|---|---|
| Final features (22 candidates → 4) | `mean3_cust_prod`, `te_product`, `te_customer`, `is_new_cust_prod` |
| R² training (weeks 4-8) / validation (week 9) | 0.656 / **0.691** (2 weeks ahead: 0.681) |
| Baseline (always the average) | R² ≈ 0.00 |
| RMSLE / MAE (units) on week 9 | 0.510 / 4.17 |
| Final equation | log1p(demand) = −0.657 + 0.648·mean3_cust_prod + 0.345·te_product + 0.329·te_customer − 0.340·is_new_cust_prod |

Scope: 10 best-selling products, **13,721,629 training rows** (weeks 3-9, all rows, no sampling) and 169,578 test rows (weeks 10-11, no target → forecasts only).

---

## 1. Why you start from two small data files (and do not download Kaggle)
The original Kaggle training table has more than 70 million rows (about 3 GB) and the test table about 7 million rows.
Downloading, loading, cleaning and filtering them takes hours and a lot of disk space, memory and SQL Server setup, and can overload a normal laptop.
I already did those steps in SQL Server (cleaning, then selecting the 10 best-selling products into two separate tables) and exported the
**result** to two Parquet files that are stored in this repository (`data/export/`, about 76 MB in total). You load them into **your own SQL Server** with one command,
and from then on the whole pipeline works through SQL exactly as in the author's project. There is **no Kaggle download and no cleaning to repeat**.
The scripts that created the tables are kept in `sql/` for transparency (section 6).

## 2. Setup (VS Code)
```powershell
git clone https://github.com/luckynnmn/bimbo-demand-forecast-gradient-descent
cd bimbo-demand-forecast-gradient-descent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
**If activation is blocked** (error says "running scripts is disabled"), run this command and then try the activation line again. It only changes the current PowerShell window, not your computer permanently:
 
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
 
When the environment is active, you see `(.venv)` at the start of the line.
**Requirements: Python 3.10+ and, for SQL Server, *ODBC Driver 17 or 18 for SQL Server* plus SSMS.**

## 3. Create the SQL tables from the data files
`data/export/train_filtered.parquet` (13,721,629 rows) and `data/export/test_filtered.parquet` (169,578 rows) come with the repository.

**3.1 Create an empty database.** In SSMS open a *New Query* and run:
```sql
CREATE DATABASE Bimbo;
```
**3.2 Connect from VS Code.** Create a file named `.env` in the project folder (copy `.env.example`) and set your server name:
```
DB_DIALECT=mssql
DB_HOST=localhost            # named instance: localhost\INSTANCE_NAME  (or localhost + DB_PORT=1433 if TCP/IP is enabled)
DB_PORT=
DB_NAME=Bimbo
DB_USER=
DB_PASSWORD=
DB_TRUSTED_CONNECTION=true   # Windows login; leave DB_USER / DB_PASSWORD empty
DB_ODBC_DRIVER=ODBC Driver 17 for SQL Server   # must match: python -c "import pyodbc; print(pyodbc.drivers())"
FILTERED_TRAIN_TABLE=dbo.train_filtered
FILTERED_TEST_TABLE=dbo.test_filtered
```
**3.3 Load the two tables.**
```powershell
python tools/import_parquet_to_sql.py
```
The script creates `dbo.train_filtered` and `dbo.test_filtered`, loads the files in batches, then counts the rows **in SQL** and compares them with the files.
It must print `OK` for both tables (13,721,629 and 169,578 rows); the training table takes several minutes. If it prints `MISMATCH` or stops, run it again (it recreates the tables).

**3.4 Check the tables in SSMS.** The import script already compares the row counts, but you can also see the tables yourself:
1. In SSMS, in *Object Explorer* click **Refresh** (the circular-arrow icon), then expand **Databases → Bimbo → Tables**.
   You must see `dbo.train_filtered` and `dbo.test_filtered`.
2. Click the **New Query** button (top-left toolbar). Make sure the database box on the toolbar says **Bimbo** (if it says `master`, pick `Bimbo` from the drop-down list).
3. Paste the commands below and press **Execute** (or **F5**):
```sql
SELECT COUNT(*) AS train_rows FROM dbo.train_filtered;
SELECT COUNT(*) AS test_rows  FROM dbo.test_filtered;
```
4. In the **Results** pane you must see **13,721,629** for `train_rows` and **169,578** for `test_rows`. If a table is missing or the numbers are different, go back to step 3.3 and run the import script again.
5. Optional: look at a few rows with `SELECT TOP 5 * FROM dbo.train_filtered;` (11 columns: `Week`, `Agency_ID`, `Channel_ID`, `Route_SAK`, `Customer_ID`, `Product_ID`, `Sales_Units_Today`, `Sales_Today`, `Returns_Units_Next`, `Returns_Next`, `Adjusted_Demand_Units`).

**3.5 Check the connection from VS Code.**
```powershell
python test_connection.py
```
It must print `Connection OK`, `dbo.train_filtered: 13,721,629 rows` and `dbo.test_filtered: 169,578 rows`.

The data come from the Kaggle competition *Grupo Bimbo Inventory Demand* and are shared only for this course assignment.
No SQL Server? Use the demo database in section 5.

## 4. Run order (P1: feasibility check)
| Step | What | Output |
|---|---|---|
| 1 | Run all cells of `notebooks/train_full_filtered.ipynb` (select the kernel of your virtual environment): EDA → skewness → Gradient Descent → feature ladder → final equation → validation on week 9 | tables and charts in the notebook; files in `outputs/final/` |

The notebook already contains the output of the Lucky's run, so you can read the results without running it.
Applying the frozen equation to the test weeks 10-11 and the saved-model demonstration belong to the next stage (P2).
The first run reads the data from SQL week by week (chunks of at most 1,000,000 rows) and caches it as parquet files in `data/processed/final/`;
the cache is removed automatically if the SQL tables change. Expect several minutes to tens of minutes depending on your computer.
Settings (learning rate, epochs, `MIN_GAIN`, `CORR_DROP`, ...) are at the top of the notebook.

## 5. Quick check without SQL Server (synthetic demo database)
```powershell
python tools/make_demo_db.py
copy .env.demo .env
jupyter nbconvert --to notebook --execute --inplace notebooks/train_full_filtered.ipynb
```
The demo data are synthetic: they only prove that the pipeline runs; the numbers are not real results.

## 6. Repository contents (P1)
```
bimbo_gd/        Python package: config, db connection, features, gradient_descent (own NumPy GD), metrics, product_scope, batch_pipeline, data_loader
notebooks/       train_full_filtered.ipynb (EDA, skewness, feature ladder, final equation, validation)
sql/             01_create_filtered_tables.sql (first filter with a fixed product list)
                 02_select_top10_products.sql (final filter: 10 best-selling products -> dbo.selected_products_v2, train_filtered, test_filtered)
data/export/     train_filtered.parquet, test_filtered.parquet (the two filtered tables exported from SQL)
tools/           import_parquet_to_sql.py (loads the data files into SQL), export_parquet.py (how the files were made), make_demo_db.py (synthetic demo)
report/          P1_Combined_Package.docx / .pdf (P1 report)
test_connection.py        checks the SQL connection and counts rows
requirements.txt, .env.demo, .env.example, .gitignore
```
`sql/01` and `sql/02` are the Lucky's scripts that created the filtered tables from the full Kaggle tables (`dbo.train`, `dbo.test`); you do **not** need to run them.
Files created when you run the notebook (`outputs/`, `data/processed/`) are not stored in the repository.

## 7. Method (flow of the training notebook)
1. **EDA** of the whole table (quality, volumes, per-product profile, distributions).
2. **Skewness**: demand is strongly right-skewed (24.1) → `log1p` (0.54); the extreme tail is capped in the training rows only; skewness is checked per feature and features with |r| ≥ 0.85 are merged.
3. **22 candidate features** built only from earlier weeks (no leakage): smoothed averages per product / customer / agency / route / channel / pair, popularity counts, last orders of the pair and their mean, weeks since last order, return rates, price, "never seen" flags.
4. **Feature ladder**: features enter one at a time; the table shows intercept, new coefficient, R² and MSE (training and validation) at every step.
5. **Pruning** (like choosing a best-fit model in JMP): features that add almost nothing to R² are dropped → 4 features.
6. **Gradient Descent** learns the final equation on all rows of weeks 4-8 (expanding window); the exact least-squares solution is a cross-check.
7. **Validation** on week 9 (1 and 2 weeks ahead) is compared with training R²; then the equation is frozen; applying it to test weeks 10-11 is the next stage (P2).

The workflow follows the MSBA 265 course materials (Solis, 2026; <https://github.com/shylasolis/machine-learning-pathway>):
per-feature skewness and multicollinearity audit (Module 1), Normal Equation vs Gradient Descent and standardisation (Module 2, Project 1).
The same approach for one and two features: <https://github.com/luckynnmn/linear-Regression-application-1-2-features>.
SQL cleaning and product selection apply knowledge from MSBA 230 (Database Management Systems with SQL and R).