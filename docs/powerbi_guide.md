# Power BI Dashboard — Build Guide

Build a 3-page Power BI dashboard from this project's exported data layer in
about 30–60 minutes. The result mirrors `results/dashboard.html` but as a real
Power BI report — a portfolio artifact in its own right.

**Prerequisite:** run `python main.py train` first. It exports the data layer to
`results/powerbi/`:

| File | Grain | Role |
|------|-------|------|
| `fact_forecast.csv` | Store × Dept × Week | actual vs forecast + errors |
| `fact_inventory.csv` | Store × Dept | ABC-XYZ policy, stockout, restock value |
| `dim_store.csv` | Store | type, size, region, cluster |
| `dim_date.csv` | Week | calendar attributes, holiday flag |
| `measures.dax` | — | measures to paste (one per `New measure`) |

## 1. Import & model (10 min)

1. **Power BI Desktop → Get Data → Text/CSV** — import the four CSVs
   (or *Get Data → Folder* pointed at `results/powerbi/`, then expand).
2. In **Model view**, create relationships (all one-to-many, single direction):
   - `dim_store[Store]` 1—* `fact_forecast[Store]`
   - `dim_store[Store]` 1—* `fact_inventory[Store]`
   - `dim_date[Date]` 1—* `fact_forecast[Date]`
3. Mark `dim_date` as the **date table** (Table tools → Mark as date table).
4. Open `measures.dax` in a text editor; for each measure: **Modeling → New
   measure**, paste one block, Enter. Format the `%` measures as percentage,
   the $ ones as currency (0 decimals).

## 2. Page 1 — Executive Overview (15 min)

| Element | Visual | Fields / notes |
|---------|--------|----------------|
| KPI row | 5 Card visuals | `Total Actual`, `Forecast Accuracy %`, `Forecast Bias %`, `Avg Service Level %`, `Series Needing Restock` |
| Trend | Line chart | X = `dim_date[Date]`, Y = `Total Actual` and `Total Forecast` — two lines. Optional: add `IsHoliday` to tooltips |
| Store map/rank | Bar chart | X = `Total Actual`, Y = `dim_store[Store]`, sorted desc, top 10 filter |
| Slicers (one row, top) | Slicer ×3 | `dim_store[Type]`, `dim_store[Cluster]`, `dim_date[Month_Name]` |

Design tips that make it look professional: one accent color (`#2A78D6`) for
actuals, a muted second color for forecast; titles in sentence case; remove
gridline clutter (Format → Gridlines off); left-align page title text box.

## 3. Page 2 — Forecast Accuracy (10 min)

| Element | Visual | Fields |
|---------|--------|--------|
| Error by week | Column chart | X = `Date`, Y = `MAE`. Holiday weeks stand out if you add `dim_date[IsHoliday]` to Legend |
| Error by department | Bar chart (top N) | Y = `fact_forecast[Dept]`, X = `Total Abs Error`, Top N = 10 |
| Holiday vs regular | Clustered column | X = `fact_forecast[IsHoliday]`, Y = `MAE` |
| Scatter | Scatter chart | X = `Total Actual`, Y = `Total Forecast`, Details = `Store` — dots below the diagonal are under-forecast stores |

## 4. Page 3 — Inventory Actions (10 min)

| Element | Visual | Fields |
|---------|--------|--------|
| ABC-XYZ matrix | Matrix visual | Rows = `ABC`, Columns = `XYZ`, Values = count of Store + `Total Restock Value` |
| Priority table | Table | Store, Dept, `ABC_XYZ`, `Stockout_Rate`, `Service_Level`, `Restock_Value`; filter `Restock_Recommended = 1`; conditional formatting (data bars) on `Stockout_Rate` |
| KPI row | Cards | `Series Needing Restock`, `Total Restock Value`, `Avg Stockout Rate %` |

## 5. Finish

- **File → Save as** `powerbi/walmart_forecasting.pbix` (folder is gitignored
  except the .pbix if you choose to commit it — it's typically a few MB).
- Take a screenshot of Page 1 for the README / your CV.
- Optional: publish to Power BI Service (free tier) and add the public link
  to the README.

## Notes

- All values are **dollars** — the dataset has no unit prices.
- `WMAE` here matches the project's primary metric (holiday weeks ×5).
- Re-running `python main.py train` refreshes the CSVs; hit **Refresh** in
  Power BI to pull the new data.
