"""Parsing, merging, and null profiling for the raw SMARD exports."""

from pathlib import Path

import pandas as pd

from src import config


def _clean_column_name(name: str) -> str:
    name = name.lstrip("﻿")
    name = name.replace("∅ ", "avg ")
    return name.split("[")[0].strip()


def _to_float(series: pd.Series) -> pd.Series:
    na_tokens = set(config.NA_VALUES) | {"", "nan"}
    series = series.where(~series.isin(na_tokens), None)
    series = series.str.replace(",", "", regex=False)
    return series.astype(float)


def load_smard_csv(path: Path) -> pd.DataFrame:
    """Parse one raw SMARD export into a float DataFrame indexed by timestamp."""
    df = pd.read_csv(path, sep=config.SEP, encoding="utf-8", dtype=str)
    df.columns = [_clean_column_name(c) for c in df.columns]

    df.index = pd.to_datetime(df["Start date"], format=config.DATE_FORMAT)
    df.index.name = "timestamp"
    df = df.drop(columns=["Start date", "End date"])

    df = df.apply(_to_float)

    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def load_and_merge() -> pd.DataFrame:
    """Load and outer-join the three SMARD exports onto one complete hourly spine."""
    generation = load_smard_csv(config.GENERATION_FILE).rename(columns=config.GENERATION_RENAME)
    consumption = load_smard_csv(config.CONSUMPTION_FILE).rename(
        columns={**config.CONSUMPTION_RENAME, "grid load": "load"}
    )
    price = load_smard_csv(config.PRICE_FILE)[[config.PRICE_COLUMN]].rename(
        columns={config.PRICE_COLUMN: "price"}
    )

    df = generation.join(consumption, how="outer").join(price, how="outer")

    spine = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(spine)
    df.index.name = "timestamp"

    assert len(df) > 17000, f"row count collapsed: {len(df)}"
    return df


def null_profile(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: null_count, null_pct, n_unique, first_valid, last_valid."""
    profile = pd.DataFrame(
        {
            "null_count": df.isna().sum(),
            "null_pct": df.isna().mean() * 100,
            "n_unique": df.nunique(),
            "first_valid": [df[c].first_valid_index() for c in df.columns],
            "last_valid": [df[c].last_valid_index() for c in df.columns],
        }
    )
    return profile.sort_values("null_pct", ascending=False)
