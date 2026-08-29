"""Metrics, splits, and baselines for phase 5. Scores predictions only — no model training."""

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.config import PRICE_REGIMES, TEST_YEAR, TRAIN_YEAR


def chronological_split(
    df: pd.DataFrame, train_year: int | None = None, test_year: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into train/test by calendar year on its DatetimeIndex. Never shuffles."""
    train_year = TRAIN_YEAR if train_year is None else train_year
    test_year = TEST_YEAR if test_year is None else test_year

    train = df.loc[df.index.year == train_year]
    test = df.loc[df.index.year == test_year]

    assert train.index.max() < test.index.min(), (
        f"train max ({train.index.max()}) is not before test min ({test.index.min()})"
    )
    assert len(train) > 1000, f"train split has {len(train)} rows, expected > 1000"
    assert len(test) > 1000, f"test split has {len(test)} rows, expected > 1000"

    return train, test


def naive_baseline(test: pd.DataFrame, price_col: str = "price", lag_col: str = "price_lag24") -> pd.Series:
    """The naive forecast: yesterday's price at the same hour — the bar every model must clear."""
    return test[lag_col]


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """MAE, RMSE, R2, and mean bias between y_true and y_pred; pairwise-complete, never a global dropna."""
    combined = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    n_total = len(combined)
    valid = combined.dropna()
    n = len(valid)

    if n == 0:
        return pd.Series(
            {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "mean_error": float("nan"), "n": 0, "n_dropped": n_total}
        )

    mae = mean_absolute_error(valid["y_true"], valid["y_pred"])
    rmse = float(np.sqrt(mean_squared_error(valid["y_true"], valid["y_pred"])))
    r2 = r2_score(valid["y_true"], valid["y_pred"]) if n >= 2 else float("nan")
    mean_error = (valid["y_pred"] - valid["y_true"]).mean()

    return pd.Series({"mae": mae, "rmse": rmse, "r2": r2, "mean_error": mean_error, "n": n, "n_dropped": n_total - n})


def metrics_by_regime(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """Regression metrics bucketed on y_true by config.PRICE_REGIMES, plus a final 'all' row."""
    combined = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()

    rows = []
    for regime, (low, high) in PRICE_REGIMES.items():
        mask = combined["y_true"] > low if math.isinf(high) else (combined["y_true"] >= low) & (combined["y_true"] < high)
        subset = combined.loc[mask]
        m = regression_metrics(subset["y_true"], subset["y_pred"])
        rows.append({"regime": regime, "n": int(m["n"]), "mae": m["mae"], "rmse": m["rmse"], "mean_error": m["mean_error"]})

    m_all = regression_metrics(combined["y_true"], combined["y_pred"])
    rows.append({"regime": "all", "n": int(m_all["n"]), "mae": m_all["mae"], "rmse": m_all["rmse"], "mean_error": m_all["mean_error"]})

    return pd.DataFrame(rows)


def compare_models(results: dict[str, pd.Series], y_true: pd.Series) -> pd.DataFrame:
    """Score each {model_name: predictions} entry against y_true, sorted by MAE. Requires a 'naive' entry."""
    if "naive" not in results:
        raise ValueError("results must include a 'naive' baseline entry")

    rows = []
    for name, y_pred in results.items():
        m = regression_metrics(y_true, y_pred)
        rows.append({"model": name, **m.to_dict()})

    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)


def classification_metrics(y_true: pd.Series, y_prob: pd.Series, threshold: float = 0.5) -> pd.Series:
    """Classification metrics for the negative-price classifier; pairwise-complete.

    Base rate is ~5.9% (negative-price hours), so accuracy is misleading here —
    lead with average_precision (PR-AUC) instead.
    """
    combined = pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).dropna()
    y_t = combined["y_true"].astype(int)
    y_p = combined["y_prob"]
    y_label = (y_p >= threshold).astype(int)

    n = len(combined)
    n_positive = int(y_t.sum())
    both_classes_present = y_t.nunique() > 1

    tn, fp, fn, tp = confusion_matrix(y_t, y_label, labels=[0, 1]).ravel()

    return pd.Series(
        {
            "n": n,
            "n_positive": n_positive,
            "base_rate": n_positive / n if n else float("nan"),
            "accuracy": accuracy_score(y_t, y_label),
            "precision": precision_score(y_t, y_label, zero_division=0),
            "recall": recall_score(y_t, y_label, zero_division=0),
            "f1": f1_score(y_t, y_label, zero_division=0),
            "roc_auc": roc_auc_score(y_t, y_p) if both_classes_present else float("nan"),
            "average_precision": average_precision_score(y_t, y_p) if both_classes_present else float("nan"),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
    )


def threshold_sweep(y_true: pd.Series, y_prob: pd.Series, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Precision/recall/f1 of the negative-price classifier across candidate thresholds."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.05)

    combined = pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).dropna()
    y_t = combined["y_true"].astype(int)
    y_p = combined["y_prob"]

    rows = []
    for t in thresholds:
        y_label = (y_p >= t).astype(int)
        rows.append(
            {
                "threshold": t,
                "precision": precision_score(y_t, y_label, zero_division=0),
                "recall": recall_score(y_t, y_label, zero_division=0),
                "f1": f1_score(y_t, y_label, zero_division=0),
            }
        )

    return pd.DataFrame(rows)
