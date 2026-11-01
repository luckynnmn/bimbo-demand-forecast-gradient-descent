-- Select the 10 best-selling products that (a) are sold in EVERY training week,
-- (b) appear in BOTH test weeks, and (c) have enough rows each week.
-- Ranking uses weeks 3-8 only (validation week 9 is never used for selection).
-- Result: dbo.selected_products_v2, dbo.train_filtered, dbo.test_filtered
USE Bimbo;
GO
DECLARE @TopN int = 10, @MinTrainRowsWeek int = 1000, @MinTestRowsWeek int = 100,
        @FirstWeek int = 3, @LastRankWeek int = 8;

DROP TABLE IF EXISTS dbo.selected_products_v2;
DROP TABLE IF EXISTS dbo.train_filtered;
DROP TABLE IF EXISTS dbo.test_filtered;

WITH train_week AS (
    SELECT Product_ID, Week, COUNT(*) AS n_rows, SUM(CAST(Adjusted_Demand_Units AS bigint)) AS units
    FROM dbo.train GROUP BY Product_ID, Week),
train_stats AS (
    SELECT Product_ID, COUNT(DISTINCT Week) AS weeks_in_train, MIN(n_rows) AS min_rows_week,
           SUM(CASE WHEN Week BETWEEN @FirstWeek AND @LastRankWeek THEN units END) AS units_w3_8
    FROM train_week GROUP BY Product_ID),
test_week AS (
    SELECT Product_ID, Week, COUNT(*) AS n_rows FROM dbo.test GROUP BY Product_ID, Week),
test_stats AS (
    SELECT Product_ID, COUNT(DISTINCT Week) AS weeks_in_test, MIN(n_rows) AS min_test_rows
    FROM test_week GROUP BY Product_ID)
SELECT TOP (@TopN) a.Product_ID, a.units_w3_8, a.weeks_in_train, a.min_rows_week, b.weeks_in_test, b.min_test_rows
INTO dbo.selected_products_v2
FROM train_stats a JOIN test_stats b ON a.Product_ID = b.Product_ID
WHERE a.weeks_in_train = (SELECT COUNT(DISTINCT Week) FROM dbo.train)
  AND a.min_rows_week >= @MinTrainRowsWeek
  AND b.weeks_in_test = (SELECT COUNT(DISTINCT Week) FROM dbo.test)
  AND b.min_test_rows >= @MinTestRowsWeek
ORDER BY a.units_w3_8 DESC;

SELECT t.* INTO dbo.train_filtered FROM dbo.train t JOIN dbo.selected_products_v2 s ON s.Product_ID = t.Product_ID;
SELECT t.* INTO dbo.test_filtered  FROM dbo.test  t JOIN dbo.selected_products_v2 s ON s.Product_ID = t.Product_ID;

SELECT * FROM dbo.selected_products_v2 ORDER BY units_w3_8 DESC;
SELECT COUNT(*) AS train_rows FROM dbo.train_filtered;
SELECT COUNT(*) AS test_rows  FROM dbo.test_filtered;
