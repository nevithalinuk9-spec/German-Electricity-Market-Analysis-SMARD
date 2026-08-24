"""Feature engineering on the merged SMARD frame."""

import holidays
import pandas as pd

from src import config


def add_generation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add generation sums, shares, and net_residual_load.

    net_residual_load correlates 1.000 with SMARD's own "Residual load" column.
    Both are kept here as a cross-check; only one should reach a model.
    """
    df = df.copy()

    df["renewable_gen"] = df[config.RENEWABLE].sum(axis=1)
    df["conventional_gen"] = df[config.CONVENTIONAL].fillna(0).sum(axis=1)
    df["total_gen"] = df["renewable_gen"] + df["conventional_gen"]

    df["wind_total"] = df["Wind onshore"] + df["Wind offshore"]
    df["vre_gen"] = df["wind_total"] + df["Photovoltaics"]

    df["renewable_share"] = df["renewable_gen"] / df["total_gen"]
    df["vre_share"] = df["vre_gen"] / df["total_gen"]
    df["net_residual_load"] = df["load"] - df["vre_gen"]

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, dayofweek, month, is_weekend, and German public-holiday flag."""
    df = df.copy()

    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    de_holidays = holidays.Germany(years=sorted(set(df.index.year)))
    df["is_holiday"] = pd.Series(df.index.date, index=df.index).isin(de_holidays).astype(int)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price_lag24, price_lag168, load_lag24 — the only day-ahead-knowable features."""
    df = df.copy()

    df["price_lag24"] = df["price"].shift(24)
    df["price_lag168"] = df["price"].shift(168)
    df["load_lag24"] = df["load"].shift(24)

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_negative_price, the phase 5 classification target; NaN price stays NaN."""
    df = df.copy()

    df["is_negative_price"] = (df["price"] < 0).astype(float).mask(df["price"].isna())

    return df


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    """Apply generation, calendar, lag, and target features in order."""
    df = add_generation_features(df)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_target(df)

    assert len(df) > 17000, f"row count collapsed: {len(df)}"
    return df