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

**Two parts.** **Part 1 (P1)** is the feasibility check: data in SQL, features, Gradient Descent, validation (sections 1-5).
**Part 2 (P2)** applies the frozen model: forecast of test weeks 10-11, saved model + load-and-test demo, forecast API, charts for the paper and the slides (sections 6-10).
**Final project package** (report, HTML report, PowerPoint): see section 10. Follow the sections **in order**; each step tells you what you must see on the screen before you go on.

---

## 1. Why you start from two small data files (and do not download Kaggle)
The original Kaggle training table has more than 70 million rows (about 3 GB) and the test table about 7 million rows.
Downloading, loading, cleaning and filtering them takes hours and a lot of disk space, memory and SQL Server setup, and can overload a normal laptop.
I already did those steps in SQL Server (cleaning, then selecting the 10 best-selling products into two separate tables) and exported the
**result** to two Parquet files that are stored in this repository (`data/export/`, about 76 MB in total). You load them into **your own SQL Server** with one command,
and from then on the whole pipeline works through SQL exactly as in the author's project. There is **no Kaggle download and no cleaning to repeat**.
The scripts that created the tables are kept in `sql/` for transparency (section 9).

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
Applying the frozen equation to the test weeks 10-11, the saved-model demonstration and the API belong to Part 2 (sections 6-10); they need the files that this notebook saves in `outputs/final/`.
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

---

# Part 2 (P2): forecast, saved model, API and charts

**Before you start Part 2** you must have finished Part 1: the virtual environment is active (`(.venv)` at the start of the line), SQL Server is running, `python test_connection.py` prints `Connection OK`, and `notebooks/train_full_filtered.ipynb` has been run completely.
Check it with this command; the folder must contain `model_final.npz`, `history_stats_final.pkl`, `metrics_final.json` and `reference_check.npz`:
```powershell
dir outputs\final
```
If a file is missing, go back to section 4 and run the training notebook again.

## 6. Forecast the test weeks 10 and 11
This step applies the frozen equation to **every row** of `dbo.test_filtered` (169,578 rows). The test table has no demand column (the competition hides it), so the forecast itself is the result.

**6.1 Open the notebook.** In VS Code open `notebooks/test_full_filtered.ipynb` and choose the kernel of your virtual environment (top right, `.venv`).

**6.2 Run all cells.** Click **Run All** at the top of the notebook. It takes about one to two minutes.

**6.3 Check what you see**, section by section:
1. *Section 1 (model)*: a table with the intercept, the hyper-parameters and `R² on validation week 9 = 0.6906`, then the final equation.
2. *Section 2 (test rows)*: `Weeks in the test table: [10, 11] | total rows: 169,578` and `Products in test but NOT in the training table: none`.
3. *Section 3 (forecast)*: `169,578 test rows forecast -> outputs\final\test_predictions_rows.csv`.
4. *Section 4 (per product)*: a table with the expected demand of each product for week 10 and week 11, and the customer × product tables (top 25 customers).
5. *Section 5 (R²)*: `Expected for TEST WEEK 10: R² = 0.6906, RMSLE = 0.5098` and `Expected for TEST WEEK 11: R² = 0.6808, RMSLE = 0.5177`. These are not real test scores (labels are hidden); they come from forecasting the known week 9 one and two weeks ahead.
6. *Section 6 (compatibility)*: new customers 0.8% (week 10) and 1.3% (week 11), new products 0.0%, new customer-product pairs 4.7% and 6.3%; then the feature-drift table and one chart per product (actual weeks 3-9 and forecast weeks 10-11).

**6.4 Files created** in `outputs/final/`: `test_predictions_rows.csv` (id + forecast), `test_predictions_by_product_week.csv`, `customer_x_product_predicted_week10.csv`, `customer_x_product_predicted_week11.csv`, `test_summary.json`.
If the notebook stops with a connection error, go back to `python test_connection.py` (section 3.5).

*Reading the charts:* the forecast for week 10 is usually a little below the real average of week 9. This is mainly the known bias of converting a log-scale forecast back to units (the forecast is on average 4% to 26% below real demand per product), not a real drop in demand.

## 7. Load the saved model and test it (no database)
This step proves that the saved model can be loaded in a **fresh program**, without SQL and without training, and that it gives the same answers as at training time.

**7.1 Run it** from the project folder (the one that contains `load_and_test_demo.py`):
```powershell
python load_and_test_demo.py
```
**7.2 Check what you see:**
1. `Loaded model: 4 features | intercept -0.6570 | history weeks ...`
2. `Check 1 - same predictions as at training time on 2,000 stored rows: PASS`
3. `Check 2 - example forecasts ...`: a small table of three example customers; the third customer is unknown to the model, so the product average is used (the program does not fail).

If you see `Check 1 ... FAIL` or `Missing ...`, run the training notebook again (section 4).

## 8. Start the forecast API and test it
The frozen equation is also served by a small local API (FastAPI). It uses compact lookup files in `api/artifacts/` (13 MB), so it needs **no SQL Server and no training**. You need **two terminals**.

**8.1 Install the packages (only once).** In the first terminal, with `(.venv)` active:
```powershell
cd api
pip install -r requirements-api.txt
```
**8.2 Start the server (terminal 1).**
```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
You must see `Application startup complete.` and `Uvicorn running on http://127.0.0.1:8000`. **Leave this terminal open**; it is the running server.

**8.3 Run the smoke check (terminal 2).** In VS Code click the **+** of the terminal panel to open a second terminal, activate the environment again (`.\.venv\Scripts\Activate.ps1` from the project folder), then:
```powershell
cd api
python smoke_check_api.py
```
The script (a) calls `GET /health` and checks `{"status": "ready", "model_id": "..."}`, (b) sends the example request stored in `artifacts/model_metadata.json` to `POST /predict`, (c) compares the answer with the reference computed by the full original pipeline when the model was frozen (tolerance 1e-6 on the log scale). The last line must be:
```
PASS: health and forecast match the supplied reference.
```
**8.4 Try the API yourself (optional).**
- In a browser open `http://127.0.0.1:8000/health`: you see `{"status":"ready","model_id":"bimbo-gd-linear-1fddf012c1b6"}` (the address `http://127.0.0.1:8000/` alone shows "Not Found"; this is normal).
- Open `http://127.0.0.1:8000/docs`, click `POST /predict` → **Try it out**, enter `{"customer_id": 1502792, "product_id": 1242}` and click **Execute**. The answer is about **5.7 units**.
- Or in PowerShell:
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body '{"customer_id":1502792,"product_id":1242}'
```
The answer contains `predicted_units`, `predicted_log_demand`, the four feature values and the flags `known_customer`, `known_product`, `known_pair`. A customer or product that the model has never seen receives the overall average level instead of an error, and the flags say so.

**8.5 Stop the server.** Click the terminal that runs the server and press `Ctrl + C`.

| Problem | What to do |
|---|---|
| `No module named uvicorn` | `(.venv)` is not active or step 8.1 was skipped |
| `Address already in use` | close the old server (`Ctrl + C`) or use `--port 8001` (then change the URL in `smoke_check_api.py`) |
| The smoke check cannot connect | the server of step 8.2 is not running |

`api/build_artifacts.py` (optional) rebuilds the artifacts from `history_stats_final.pkl`; the 200 MB history file is not stored in the repository.

## 9. Draw the charts of the paper and the slides
`summarize_results.py` reads the files saved by the two notebooks (it trains and tests nothing, and needs no database) and draws all the charts used in `report/P2_Draft_Paper.docx` and `report/Stakeholder_Presentation.pptx`. Run it from the project folder:
```powershell
python summarize_results.py
```
You must see one line `saved report\figures\... <- fig_...` for each chart and `Done: figures are in report\figures`:
`pipeline_compact.png` (icon-and-arrow pipeline), `eda.png`, `batch_r2.png`, `ladder.png`, `coefficients.png`, `product_scatter.png`, `inventory.png` and `forecast_vs_history.png`.
If `forecast_vs_history` is skipped, run the test notebook (section 6) first.

## 10. Where to find the results and the final deliverables
| What | File |
|---|---|
| Final paper (PDF, 10-section outline) | `report/Final_Report.pdf` |
| Rendered report (HTML, open it in a browser) | `report/Final_Report.html` |
| Stakeholder PowerPoint (with speaker notes) | `report/Stakeholder_Presentation.pptx` |
| Data source and access notes | `data/README.md` |
| Saved model, reload test and forecast API (FastAPI extension) | `outputs/final/model_final.npz` + `load_and_test_demo.py` (sections 6-7), `api/` (section 8) |
| Completed Assignment Validation form | `Assignment_Validation_Form.pdf` |
| Charts | `report/figures/*.png` |
| Forecast of the test weeks | `outputs/final/test_predictions_rows.csv` and the other files listed in 6.4 |
| Saved model and demo | `outputs/final/model_final.npz`, `load_and_test_demo.py` |
| Forecast API | `api/` |

Main results: expected R² on the log scale **0.691** (week 10) and **0.681** (week 11); the forecast is 25.7% below and 14.1% above real demand in total on validation week 9 (stockout and excess-stock risk).

## 11. Repository contents
Part 1 files are marked **P1**, Part 2 files **P2**, final-stage files **Final**.

```text
bimbo-demand-forecast-gradient-descent/
│
├── bimbo_gd/                   Python package (own NumPy Gradient Descent, features, SQL access)
│
├── notebooks/
│   ├── train_full_filtered.ipynb   P1  EDA, features, Gradient Descent, validation
│   └── test_full_filtered.ipynb    P2  forecast of test weeks 10-11
│
├── api/                        P2  forecast API
│   ├── app.py                      FastAPI service
│   ├── smoke_check_api.py          health + forecast check (prints PASS)
│   ├── requirements-api.txt
│   ├── build_artifacts.py          optional: rebuilds the artifacts
│   ├── README.md
│   └── artifacts/                  model.npz, history_compact.npz, model_metadata.json
│
├── data/
│   ├── README.md                   source, access and licence notes
│   └── export/                     P1  train_filtered.parquet, test_filtered.parquet
├── sql/                        P1  01_create_filtered_tables.sql, 02_select_top10_products.sql
│
├── tools/
│   ├── import_parquet_to_sql.py    loads the data files into SQL Server
│   ├── export_parquet.py           how the data files were made
│   └── make_demo_db.py             synthetic demo database
│
├── report/
│   ├── P1_Combined_Package.docx / .pdf     P1  report
│   ├── P2_Draft_Paper.docx                 P2  draft paper
│   ├── Final_Report.pdf / .html            Final  paper and rendered report
│   ├── Stakeholder_Presentation.pptx       P2 / Final  slides with speaker notes
│   └── figures/                            charts of the paper and the slides
│
├── load_and_test_demo.py       P2  loads the saved model and tests it (no database)
├── summarize_results.py        P2  draws the charts from outputs/final/
├── test_connection.py          P1  checks the SQL connection
├── Assignment_Validation_Form.pdf   completed form with the peer's signature
└── requirements.txt, .env.example, .env.demo, .gitignore
```

| Folder or file | What you do with it |
|---|---|
| `data/export/`, `tools/import_parquet_to_sql.py` | load the data into your SQL Server (section 3) |
| `notebooks/` | run in order: train (section 4), then test (section 6) |
| `load_and_test_demo.py` | section 7 |
| `api/` | section 8 |
| `summarize_results.py` | section 9 |
| `sql/` | for reading only: the scripts that created the filtered tables from the full Kaggle tables; you do **not** need to run them |

`outputs/` and `data/processed/` are created when you run the notebooks and are not stored in the repository; `api/artifacts/` is stored so the API runs right after cloning.

## 12. Method (flow of the training notebook)
1. **EDA** of the whole table (quality, volumes, per-product profile, distributions).
2. **Skewness**: demand is strongly right-skewed (24.1) → `log1p` (0.54); the extreme tail is capped in the training rows only; skewness is checked per feature and features with |r| ≥ 0.85 are merged.
3. **22 candidate features** built only from earlier weeks (no leakage): smoothed averages per product / customer / agency / route / channel / pair, popularity counts, last orders of the pair and their mean, weeks since last order, return rates, price, "never seen" flags.
4. **Feature ladder**: features enter one at a time; the table shows intercept, new coefficient, R² and MSE (training and validation) at every step.
5. **Pruning** (like choosing a best-fit model in JMP): features that add almost nothing to R² are dropped → 4 features.
6. **Gradient Descent** learns the final equation on all rows of weeks 4-8 (expanding window); the exact least-squares solution is a cross-check.
7. **Validation** on week 9 (1 and 2 weeks ahead) is compared with training R²; then the equation is frozen.
8. **P2:** the frozen equation is applied to test weeks 10-11 (`test_full_filtered.ipynb`), saved and reloaded (`load_and_test_demo.py`) and served by the API (`api/`).

The workflow follows the MSBA 265 course materials (Solis, 2026; <https://github.com/shylasolis/machine-learning-pathway>):
per-feature skewness and multicollinearity audit (Module 1), Normal Equation vs Gradient Descent and standardisation (Module 2, Project 1).
The same approach for one and two features: <https://github.com/luckynnmn/linear-Regression-application-1-2-features>.
SQL cleaning and product selection apply knowledge from MSBA 230 (Database Management Systems with SQL and R).
