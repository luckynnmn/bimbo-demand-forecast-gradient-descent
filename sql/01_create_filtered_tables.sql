-- Run in SSMS, database [Bimbo]. Creates the two tables used by the notebooks.
-- (If the test table has the typo column "Angency_ID", rename it first - see section 0.)

-- 0) one-time fix of a typo in the test table (skip if the column is already Agency_ID)
-- EXEC sp_rename 'dbo.test.Angency_ID', 'Agency_ID', 'COLUMN';

-- 1) the 10 products used in this project
SELECT *
INTO   dbo.train_filtered
FROM   dbo.train
WHERE  Product_ID IN (2233, 1242, 1250, 35305, 1240, 1146, 1109, 1232, 1125, 1278);

SELECT *
INTO   dbo.test_filtered
FROM   dbo.test
WHERE  Product_ID IN (2233, 1242, 1250, 35305, 1240, 1146, 1109, 1232, 1125, 1278);

-- 2) sanity checks
SELECT COUNT(*) AS train_rows FROM dbo.train_filtered;
SELECT COUNT(*) AS test_rows  FROM dbo.test_filtered;
SELECT Week, COUNT(*) AS rows_in_week FROM dbo.train_filtered GROUP BY Week ORDER BY Week;
SELECT Week, COUNT(*) AS rows_in_week FROM dbo.test_filtered  GROUP BY Week ORDER BY Week;
