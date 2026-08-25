"""Profiling and correlation functions for phase 2. Returns only — no printing, no plotting, no writes."""

from itertools import combinations

import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from src import config


def screen_candidates(
    df: pd.DataFrame, candidates: list[str], max_null_pct: float = 20.0
) -> tuple[list[str], pd.DataFrame]:
    """Screen candidate features for high nullity or zero variance; return survivors and a drop log."""
    dropped_rows = []
    surviving = []

    for col in candidates:
        null_pct = df[col].isna().mean() * 100
        n_unique = df[col].nunique()

        if null_pct > max_null_pct:
            dropped_rows.append(
                {"feature": col, "reason": "high null pct", "null_pct": null_pct, "n_unique": n_unique}
            )
        elif n_unique <= 1:
            dropped_rows.append(
                {"feature": col, "reason": "constant", "null_pct": null_pct, "n_unique": n_unique}
            )
        else:
            surviving.append(col)

    dropped_df = pd.DataFrame(dropped_rows, columns=["feature", "reason", "null_pct", "n_unique"])
    return surviving, dropped_df


def describe_target(df: pd.DataFrame, col: str = "price") -> pd.Series:
    """Distribution summary and negative/spike-hour counts for the target column."""
    assert len(df) > 17000, f"row count collapsed: {len(df)}"

    valid = df[col].dropna()
    n = len(valid)
    n_negative = int((valid < 0).sum())
    n_spike = int((valid > config.SPIKE_THRESHOLD).sum())

    return pd.Series(
        {
            "n": n,
            "missing": int(df[col].isna().sum()),
            "mean": valid.mean(),
            "median": valid.median(),
            "std": valid.std(),
            "min": valid.min(),
            "max": valid.max(),
            "p01": valid.quantile(0.01),
            "p99": valid.quantile(0.99),
            "skew": valid.skew(),
            "kurtosis": valid.kurt(),
            "n_negative": n_negative,
            "pct_negative": n_negative / n * 100,
            "n_spike": n_spike,
            "pct_spike": n_spike / n * 100,
        }
    )


def target_by_year(df: pd.DataFrame, col: str = "price") -> pd.DataFrame:
    """Per-year target summary — exposes the 2024 -> 2025 regime shift."""
    assert len(df) > 17000, f"row count collapsed: {len(df)}"

    rows = []
    for year in sorted(set(df.index.year)):
        valid = df.loc[df.index.year == year, col].dropna()
        n = len(valid)
        n_negative = int((valid < 0).sum())
        n_spike = int((valid > config.SPIKE_THRESHOLD).sum())
        rows.append(
            {
                "year": year,
                "n": n,
                "mean": valid.mean(),
                "median": valid.median(),
                "pct_negative": n_negative / n * 100,
                "pct_spike": n_spike / n * 100,
            }
        )

    return pd.DataFrame(rows).set_index("year")


def correlation_table(df: pd.DataFrame, candidates: list[str], target: str = "price") -> pd.DataFrame:
    """Pairwise pearson/spearman/mutual-info of each candidate against the target."""
    assert len(df) > 17000, f"row count collapsed: {len(df)}"

    rows = []
    for col in candidates:
        pair = df[[col, target]].dropna()
        pearson = pair[col].corr(pair[target], method="pearson")
        spearman = pair[col].corr(pair[target], method="spearman")
        rows.append(
            {
                "feature": col,
                "n": len(pair),
                "pearson": pearson,
                "spearman": spearman,
                "abs_pearson": abs(pearson),
                "nonlinearity": abs(spearman) - abs(pearson),
            }
        )

    table = pd.DataFrame(rows).set_index("feature")

    complete = df[candidates + [target]].dropna()
    mi = mutual_info_regression(complete[candidates], complete[target], random_state=0)
    table["mutual_info"] = pd.Series(mi, index=candidates)

    return table.sort_values("abs_pearson", ascending=False)


def correlations_by_year(df: pd.DataFrame, candidates: list[str], target: str = "price") -> pd.DataFrame:
    """Pairwise pearson correlation of each candidate against the target, split by year, with year-over-year drift."""
    assert len(df) > 17000, f"row count collapsed: {len(df)}"

    years = sorted(set(df.index.year))
    rows = []
    for col in candidates:
        row = {"feature": col}
        for year in years:
            year_pair = df.loc[df.index.year == year, [col, target]].dropna()
            row[str(year)] = year_pair[col].corr(year_pair[target])
            row[f"{year}_mean_price"] = year_pair[target].mean()
            row[f"{year}_n"] = len(year_pair)
        rows.append(row)

    table = pd.DataFrame(rows).set_index("feature")

    prev_year, last_year = str(years[-2]), str(years[-1])
    table["drift"] = (table[last_year] - table[prev_year]).abs()

    return table.sort_values("drift", ascending=False)


def multicollinearity_pairs(df: pd.DataFrame, candidates: list[str], threshold: float = 0.8) -> pd.DataFrame:
    """All candidate predictor pairs whose pairwise pearson |r| exceeds threshold."""
    assert len(df) > 17000, f"row count collapsed: {len(df)}"

    rows = []
    for a, b in combinations(candidates, 2):
        pair = df[[a, b]].dropna()
        r = pair[a].corr(pair[b])
        if abs(r) > threshold:
            rows.append({"feature_a": a, "feature_b": b, "r": r})

    pairs = pd.DataFrame(rows, columns=["feature_a", "feature_b", "r"])
    return pairs.reindex(pairs["r"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def merit_order_deciles(
    df: pd.DataFrame, load_col: str = "net_residual_load", price_col: str = "price"
) -> pd.DataFrame:
    """Mean/median price by net-residual-load decile — the merit-order curve."""
    assert len(df) > 17000, f"row count collapsed: {len(df)}"

    pair = df[[load_col, price_col]].dropna()
    decile = pd.qcut(pair[load_col], 10, labels=False)

    grouped = pair.groupby(decile)
    result = grouped.agg(
        resid_load_mean=(load_col, "mean"),
        price_mean=(price_col, "mean"),
        price_median=(price_col, "median"),
        price_std=(price_col, "std"),
        n=(price_col, "size"),
    )
    result["pct_negative"] = grouped[price_col].apply(lambda s: (s < 0).mean() * 100)
    result.index.name = "decile"

    return result[["resid_load_mean", "price_mean", "price_median", "price_std", "pct_negative", "n"]]