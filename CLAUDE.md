# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state (check before assuming anything is built)

Nothing beyond raw data and dependencies exists yet: `data/raw/` has the 3 SMARD
CSVs, `requirements.txt` is defined, and `src/__init__.py` is empty. There is no
`notebooks/` directory, no other `src/` module, and no `data/interim/` output.
Start at Phase 1 (`config.py` → `data_loader.py`) per Build order below. Do not
assume later-phase files exist without checking.

## Commands

Environment is a local venv at `.SMARD/` (gitignored).

```powershell
# create + activate (PowerShell)
python -m venv .SMARD
.SMARD\Scripts\Activate.ps1
pip install -r requirements.txt

# run the notebooks
jupyter lab   # open notebooks/SMARD_analysis.ipynb (phases 1-4, 6) and/or
              # notebooks/SMARD_machine_Learning.ipynb (phase 5)
```

There is no lint config, formatter, or test suite in this repo, so don't invent
commands for them.

## What this is

Analysis of hourly German electricity market data from SMARD (Bundesnetzagentur),
1 Jan 2024 – 31 Dec 2025. Three source CSVs, ~17,540 rows each, joined on
`Start date`.

Pipeline: clean → EDA → visualizations → insights → ML → Power BI.

**Architecture: importable `.py` modules in `src/`, orchestrated from notebooks.**
Notebooks are the only files that run: `SMARD_analysis.ipynb` for phases 1-4
and 6, `SMARD_machine_Learning.ipynb` for phase 5. Modules provide functions;
the notebooks call them, display results, and hold all written interpretation.

Python on Windows. pandas, numpy, scikit-learn, matplotlib/seaborn/plotly.

## Layout

```
data/raw/          READ-ONLY. Never write here. The 3 SMARD CSVs.
data/interim/      clean_hourly.csv: written once, read by every later phase
data/processed/    Power BI star schema (phase 6 only)

src/
  __init__.py
  config.py          paths, column groups, constants, plot theme
  data_loader.py     phase 1: parse, merge, DST dedup, null profiling
  features.py        phase 1: engineered columns, lags, calendar
  eda.py             phase 2: profiling + correlation functions
  visualize.py       phase 3: chart builders
  models.py          phase 5: negative-price classifier, attribution GBM
  evaluate.py        phase 5: metrics, regime splits, baseline comparison
  powerbi_export.py  phase 6: star schema builder

notebooks/
  SMARD_analysis.ipynb          phases 1-4 and 6: clean, EDA, viz, insights, Power BI export
  SMARD_machine_Learning.ipynb  phase 5: ML, standalone; reads data/interim/clean_hourly.csv directly

reference/         throwaway prior run. Read for reference; NOT a pipeline input.
powerbi/           .pbix dashboard
```

Two notebooks, not one. Each is the file that is run for its phases:
`SMARD_analysis.ipynb` for 1-4 and 6, `SMARD_machine_Learning.ipynb` for phase
5. `SMARD_machine_Learning.ipynb` does not import anything from
`SMARD_analysis.ipynb`; it resolves its own project root and reads
`clean_hourly.csv` directly, so it runs standalone from either notebook.

There is no `outputs/` directory. Charts render inline in the notebook, tables
print inline, insights live in notebook markdown cells. Only two things are
written to disk: `data/interim/clean_hourly.csv` and, in phase 6, the
`data/processed/` star schema.

Module filenames are role-based, not phase-numbered, because Python module names
cannot begin with a digit: `import 01_clean` is a syntax error.

## Module contract: follow this exactly

Every function in `src/` is **pure**: takes arguments, returns a value.

- No `print()`: return the object and let the notebook display it
- No `plt.show()` / `fig.show()`: **return the figure object**
- No file writes, except explicitly-named export functions
- No module-level side effects; nothing executes on import
- Type hints and a one-line docstring on every public function
- All paths come from `src/config.py`. No hardcoded user directories anywhere.

Chart functions return a matplotlib or plotly `Figure`:

```python
def plot_merit_order(df: pd.DataFrame, color_by: str = "hour") -> go.Figure:
    """Scatter of net residual load vs price. Returns a plotly Figure."""
    ...
    return fig
```

The notebook does `fig = viz.plot_merit_order(df); fig.show()`.

**`visualize.py` contains no interpretation of results.** It builds figures only.
Conclusions are written by hand in notebook markdown cells, after the chart has
been rendered and reviewed. Visualizations come before insights, deliberately.

## Build order

Build one phase at a time. Do not scaffold ahead: later phases depend on what
the data actually looks like.

**Phase 1: clean.** `config.py` first (everything imports it), then
`data_loader.py`, then `features.py`, then the phase-1 section of the notebook.
Ends by writing `data/interim/clean_hourly.csv`.

*Acceptance:* 17,544 rows on a complete hourly spine; 2 nulls in `price`;
`Nuclear` flagged at 95.96% null; `net_residual_load` correlates 1.000 with
SMARD's own `Residual load`. If any of these is off, stop and say so.

**Phase 2: EDA.** `eda.py`: null profile, distributions, pairwise correlations
(Pearson + Spearman + mutual information), correlations split by year,
merit-order deciles. Reproduce the benchmarks below.

**Phase 3: visualizations.** `visualize.py`: fuel mix stacked area, price
duration curve, merit-order scatter coloured by hour, renewable-share heatmap
(hour × month), negative-price calendar heatmap.

**Phase 4: insights.** Notebook markdown only. No code.

**Phase 5: ML.** `models.py` + `evaluate.py`, orchestrated from its own
notebook, `notebooks/SMARD_machine_Learning.ipynb`, not from
`SMARD_analysis.ipynb`. Primary: binary classifier for negative-price hours.
Secondary: GBM on actuals with permutation importance, framed as attribution.

**Phase 6: Power BI.** `powerbi_export.py`. Star schema, described at the end.

## Parsing the SMARD files

- Separator `;`
- **English locale numbers**: `3,920.75` = comma thousands, period decimal.
  Strip commas, then `astype(float)`. Do NOT use `decimal=','`.
- Date format `%b %d, %Y %I:%M %p` (e.g. `Jan 1, 2024 12:00 AM`)
- `-` means missing → NaN
- Column headers carry verbose suffixes (`Biomass [MWh] Calculated resolutions`).
  Split on `[` and strip.
- Both the generation and consumption files contain a `Hydro pumped storage`
  column. Rename on load: `Pumped storage generation` / `Pumped storage consumption`.
- DST: October repeats an hour under an identical label.
  `df[~df.index.duplicated(keep="first")]`
- Reindex onto a complete hourly spine. Only 2 genuinely missing hours exist in
  the whole series. If you end up with materially fewer than 17,540 rows,
  something is wrong.

## Traps: these have already caused real bugs

**`Nuclear` is 95.96% null.** Germany's phase-out means it is `0.00` until
2024-01-30 11:00 and `-` for every hour after. It carries zero information
(`n_unique == 1`).

- Include it in generation SUMS via `.fillna(0)`: 0 MWh nuclear is genuinely true
- NEVER use it as a standalone model feature: it is constant, so correlation is
  undefined
- NEVER let it reach a `.dropna()`. A global dropna including this column silently
  cuts 17,542 rows to 540, with no error raised, producing plausible-looking but
  worthless results

**General rule this implies:** profile nulls BEFORE any dropna, drop dead columns
explicitly (screen on null % and `n_unique <= 1`, don't hardcode names), use
pairwise correlations rather than listwise deletion, and assert row counts after
every filtering step.

**Other traps**
- `net_residual_load` (load − wind − PV) correlates 1.000 with SMARD's own
  `Residual load`. Keep one, not both.
- `month` flips correlation sign between years (+0.249 → −0.101). Do not use it as
  a model feature. `hour` and `dayofweek` are stable.
  Quantified: adding `month` to the attribution GBM *improves* same-year
  validation MAE (18.95 vs 19.52 on an Oct–Dec 2024 holdout) but *degrades*
  2025 test MAE by 4.09 (19.48 vs 15.39). A feature can look good in
  validation and still fail across a regime shift. See
  `reference/phase5_diagnostic.py` section 9.
- `Photovoltaics` is non-monotonic vs price (Pearson −0.474 vs Spearman −0.227),
  confounded by hour-of-day because of the mass of night-time zeros. Never
  interpret its effect without conditioning on hour.
- 24 predictor pairs exceed |r| 0.8 in the load / residual-load / conventional
  cluster. Fine for trees; drop to one per cluster for linear or coefficient models.
- `shap` is unavailable in this environment: `import shap` pulls in numba, whose
  compiled extension is blocked by a Windows Application Control policy (OS-level,
  not a package problem). Do not reinstall numba/llvmlite/shap or suggest disabling
  the policy. Attribution uses `sklearn.inspection` (permutation importance +
  partial dependence) instead.

## Required in every data-handling function

```python
assert len(df) > 17000, f"row count collapsed: {len(df)}"
```

Return a null profile rather than printing one: the notebook displays it.

Guard the disk writes; files are often open in Excel on Windows:

```python
try:
    df.to_csv(path)
except PermissionError:
    df.to_csv(str(path).replace(".csv", "_new.csv"))
```

## Modelling rules: non-negotiable

The data contains **actual** generation, not day-ahead **forecasts**. A model
using actual wind/PV/load to predict the same hour's price is not forecasting:
at real prediction time those values are unknown.

- Models using actual generation → label **explanatory / attribution**, never
  forecasting, in code, docstrings, and notebook text
- Only models restricted to lags + calendar may be called forecasting
- Split chronologically: train 2024, test 2025. **Never shuffle.** No
  `train_test_split` without an explicit time cutoff
- Always report the naive `price_lag24` baseline alongside any model
- Report error separately for negative (<0), normal (0–150), and spike (>150) regimes

Primary model: binary classifier for negative-price hours (`price < 0`).
Secondary: GBM on actuals with permutation importance, framed as attribution.

Use `sklearn.ensemble.HistGradientBoosting*`. Do not add LightGBM or XGBoost:
at 17.5k rows they gain nothing and complicate the Windows install.

Open experiments, not yet run: delta target (`price − price_lag24`) to neutralise
level drift; pulling ENTSO-E day-ahead forecasts to enable genuine forecasting.

## Known benchmarks (train 2024 / test 2025)

Reproduce these. If your numbers differ materially, you have a bug: say so
rather than moving on.

| model | MAE | R² | notes |
|---|---|---|---|
| naive `price_lag24` | 25.97 | 0.397 | baseline |
| GBM, lags + calendar (honest) | 23.70 | 0.540 | **beats naive** |
| GBM, actual generation | 15.39 | 0.789 | attribution, not forecasting |
| Ridge, actual generation | 16.92 | 0.772 | attribution, not forecasting |

Negative-price classifier (primary model), same validation methodology,
selected on PR-AUC rather than MAE: **average_precision 0.784, ROC-AUC 0.981**
on 2025 (base rate 6.5%; precision 0.625, recall 0.813 at the default 0.5
threshold).

Hyperparameters (`max_iter` for both GBMs and the classifier, `alpha` for
Ridge) were selected on a chronological validation split *within* 2024 (train
Jan–Sep, validate Oct–Dec); 2025 was used exactly once, for final evaluation,
after selection. See `reference/phase5_diagnostic.py`. They are pinned in
`src/models.py` (`FORECAST_GBM_MAX_ITER`, `ATTRIBUTION_GBM_MAX_ITER`,
`RIDGE_ALPHA`, `CLASSIFIER_MAX_ITER`) rather than left at sklearn defaults.

GBM figures are sklearn-version-dependent (measured on scikit-learn 1.8.0);
naive and Ridge are stable across versions. Treat the GBM numbers as
approximate targets, not exact assertions.

Target stats: mean 83.91, median 86.36, std 52.69, range −250.32 to 936.28,
skew 1.86, kurtosis 20.33. 1,030 negative hours (5.9%), 996 above €150 (5.7%).
2024 mean 78.51 → 2025 mean 89.33 (regime shift, expect out-of-sample degradation).

The spike regime is defined as price > 150 EUR/MWh, per `config.PRICE_REGIMES`,
the single source of truth for negative/normal/spike boundaries, used for
all per-regime error reporting.

Top correlations with price: `Residual load` 0.827, `renewable_share` −0.752,
`conventional_gen` 0.748, `vre_share` −0.731, `price_lag24` 0.676.

Merit order, mean price by net-residual-load decile (monotonic across all ten):
−0.77 → 37.34 → 61.67 → 75.65 → 85.06 → 92.56 → 99.24 → 106.98 → 118.99 → 162.43.
Decile 0 is 52.9% negative-price hours.

## Power BI target (phase 6)

Two fact tables, not one: unpivoting generation into a single fact table would
duplicate price and load 12× per hour and break SUM aggregations.

- `fact_market`, one row per hour: price, load, residual load, shares, model output
- `fact_generation`, one row per hour per fuel: timestamp, fuel_id, generation_mwh
- `dim_date`, shared calendar, marked as a date table for time intelligence
- `dim_fuel_type`, 12 rows: fuel_name, category, is_renewable, is_dispatchable,
  display_order, color_hex

Single-direction relationships, dimension filtering fact. No bidirectional filtering.
Pages: Overview KPIs → Generation Mix → Price Analysis → Model Results.

## Style

- Keep functions small and single-purpose. One concern per function.
- Fail loudly. Prefer an assertion that stops execution over a silent fallback.
- Flag data-quality problems immediately rather than working around them.
- Never inflate findings. If a model underperforms, lead with that.
- The notebook uses `%load_ext autoreload` / `%autoreload 2`, so modules can be
  edited without restarting the kernel. Avoid patterns that break under autoreload
  (module-level state, cached singletons).
- `nbformat` must be installed or plotly figures render blank in the notebook.