"""Model training for phase 5. Fits estimators only — scoring lives in src/evaluate.py, not here.

Models trained on ACTUAL generation are explanatory/attribution: they use information
(realised wind, PV, load) that is unknown at day-ahead prediction time, so they explain
a price after the fact rather than forecasting it. Only models restricted to lags and
calendar features may be called forecasting. That distinction is load-bearing throughout
this module's naming and docstrings.

Hyperparameters below (max_iter for both GBMs, alpha for Ridge) are pinned to values
selected on a chronological validation split within 2024 (train Jan-Sep, validate
Oct-Dec), with 2025 held out and touched only once, for final evaluation, after
selection — see reference/phase5_diagnostic.py sections 5-11 for that selection run.
They are hardcoded rather than left at sklearn's defaults so results do not silently
shift if a future sklearn version changes what those defaults are.
"""

import inspect
from itertools import combinations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FORECAST_FEATURES = [
    "price_lag24",
    "price_lag168",
    "load_lag24",
    "hour",
    "dayofweek",
    "is_weekend",
    "is_holiday",
]

# Validation-selected on Jan-Sep 2024 train / Oct-Dec 2024 validate; 2025 untouched
# until final evaluation. See reference/phase5_diagnostic.py sections 5-12.
# The two regressors and Ridge were selected on validation MAE; the classifier
# was selected on validation average_precision (PR-AUC), appropriate for its
# ~5.9% base rate.
FORECAST_GBM_MAX_ITER = 30
ATTRIBUTION_GBM_MAX_ITER = 200
RIDGE_ALPHA = 10.0
CLASSIFIER_MAX_ITER = 200

DEFAULT_PRUNE_PRIORITY = [
    "Residual load",
    "renewable_share",
    "conventional_gen",
    "wind_total",
    "Photovoltaics",
    "load",
    "price_lag24",
    "hour",
    "dayofweek",
]


def build_feature_sets(df: pd.DataFrame, surviving: list[str]) -> dict[str, list[str]]:
    """Return {'forecast': day-ahead-knowable features, 'explanatory': surviving candidates}.

    'month' is excluded from both sets: its correlation with price flips sign
    between years (+0.249 -> -0.101), so it is not a stable predictor in either
    the forecasting or the explanatory regime. Run prune_collinear on the
    'explanatory' set before training the attribution GBM.
    """
    missing = [c for c in FORECAST_FEATURES if c not in df.columns]
    assert not missing, f"forecast feature set references missing columns: {missing}"

    explanatory_features = [c for c in surviving if c != "month"]

    return {"forecast": list(FORECAST_FEATURES), "explanatory": explanatory_features}


def prune_collinear(
    df: pd.DataFrame,
    features: list[str],
    priority: list[str] | None = None,
    threshold: float = 0.8,
) -> tuple[list[str], pd.DataFrame]:
    """Greedily drop one feature from each collinear pair (|r| > threshold), keeping the higher-priority one.

    Permutation importance splits importance between correlated twins, understating
    both of them individually — pruning collinear duplicates before training
    the attribution GBM keeps per-feature importance interpretable instead of
    smeared across near-identical columns (e.g. net_residual_load vs Residual
    load, r=1.000).
    """
    if priority is None:
        priority = DEFAULT_PRUNE_PRIORITY

    rank = {name: i for i, name in enumerate(priority)}
    fallback_rank = {name: len(priority) + i for i, name in enumerate(features)}

    def _rank(name: str) -> int:
        return rank.get(name, fallback_rank[name])

    pairs = []
    for a, b in combinations(features, 2):
        pair = df[[a, b]].dropna()
        r = pair[a].corr(pair[b])
        if pd.notna(r) and abs(r) > threshold:
            pairs.append((a, b, r))

    pairs.sort(key=lambda row: abs(row[2]), reverse=True)

    kept = set(features)
    dropped_rows = []
    for a, b, r in pairs:
        if a not in kept or b not in kept:
            continue
        loser, winner = (a, b) if _rank(a) > _rank(b) else (b, a)
        kept.discard(loser)
        dropped_rows.append({"dropped": loser, "kept_instead": winner, "r": r})

    kept_features = [f for f in features if f in kept]
    dropped_df = pd.DataFrame(dropped_rows, columns=["dropped", "kept_instead", "r"])

    return kept_features, dropped_df


def train_negative_price_classifier(
    train: pd.DataFrame, features: list[str], **kwargs
) -> HistGradientBoostingClassifier:
    """PRIMARY MODEL: HistGradientBoostingClassifier for is_negative_price (~5.9% base rate).

    Explanatory, not forecasting, whenever `features` includes actual generation —
    naming responsibility for that lies with the caller's choice of feature set.
    Class imbalance is handled with class_weight="balanced" if the installed
    sklearn supports it on this estimator, else with a manually computed
    sample_weight passed to fit — detected via inspect, not assumed.
    max_iter defaults to CLASSIFIER_MAX_ITER (validation-selected on PR-AUC;
    module docstring) unless overridden via kwargs.
    """
    assert len(train) > 1000, f"training set has {len(train)} rows, expected > 1000"

    valid = train.dropna(subset=["is_negative_price"])
    X = valid[features]
    y = valid["is_negative_price"].astype(int)
    assert y.nunique() == 2, f"training target has {y.nunique()} class(es), expected 2"

    params = {"random_state": 0, "max_iter": CLASSIFIER_MAX_ITER, **kwargs}

    if "class_weight" in inspect.signature(HistGradientBoostingClassifier.__init__).parameters:
        params.setdefault("class_weight", "balanced")
        model = HistGradientBoostingClassifier(**params)
        model.fit(X, y)
    else:
        class_counts = y.value_counts()
        weight_per_class = len(y) / (2 * class_counts)
        sample_weight = y.map(weight_per_class)
        model = HistGradientBoostingClassifier(**params)
        model.fit(X, y, sample_weight=sample_weight)

    return model


def train_attribution_gbm(
    train: pd.DataFrame, features: list[str], target: str = "price"
) -> HistGradientBoostingRegressor:
    """SECONDARY MODEL: HistGradientBoostingRegressor on actual generation. Attribution, not forecasting.

    `features` is expected to include actual generation/load columns, which are
    unknown at day-ahead prediction time — this model explains what happened to
    price in a given hour, it does not predict price ahead of that hour.
    max_iter is pinned to ATTRIBUTION_GBM_MAX_ITER (validation-selected; module docstring).
    """
    assert len(train) > 1000, f"training set has {len(train)} rows, expected > 1000"

    valid = train.dropna(subset=[target])
    X = valid[features]
    y = valid[target]

    model = HistGradientBoostingRegressor(random_state=0, max_iter=ATTRIBUTION_GBM_MAX_ITER)
    model.fit(X, y)

    return model


def train_forecast_gbm(
    train: pd.DataFrame, features: list[str], target: str = "price"
) -> HistGradientBoostingRegressor:
    """The honest forecasting configuration: HistGradientBoostingRegressor restricted to lags + calendar.

    This is the only regressor in this module allowed to be called a forecast —
    only if `features` is build_feature_sets(...)["forecast"] or an equivalent
    day-ahead-knowable subset. Under validation-selected hyperparameters (see
    module docstring) it beats the naive price_lag24 baseline (MAE ~23.7 vs
    ~25.97 on 2025) — CLAUDE.md's older "loses to naive" figure was measured
    with a feature set that included `month`, which CLAUDE.md itself forbids.
    max_iter is pinned to FORECAST_GBM_MAX_ITER (validation-selected).
    """
    assert len(train) > 1000, f"training set has {len(train)} rows, expected > 1000"

    valid = train.dropna(subset=[target])
    X = valid[features]
    y = valid[target]

    model = HistGradientBoostingRegressor(random_state=0, max_iter=FORECAST_GBM_MAX_ITER)
    model.fit(X, y)

    return model


def train_ridge_baseline(train: pd.DataFrame, features: list[str], target: str = "price") -> Pipeline:
    """Linear baseline: StandardScaler + Ridge, returned as a Pipeline so scaling travels with the model.

    Unlike the HistGradientBoosting models, Ridge cannot handle missing values,
    so rows with any NaN in `features` or `target` are dropped here — scoped to
    exactly those columns, not a global dropna. alpha is pinned to RIDGE_ALPHA
    (validation-selected; module docstring).
    """
    assert len(train) > 1000, f"training set has {len(train)} rows, expected > 1000"

    valid = train.dropna(subset=[target] + list(features))
    X = valid[features]
    y = valid[target]

    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=RIDGE_ALPHA, random_state=0))])
    model.fit(X, y)

    return model


def permutation_feature_importance(
    model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 10, random_state: int = 0
) -> pd.DataFrame:
    """Permutation importance, sorted descending with a share_of_total column.

    Pass the TEST set (X, y) here, not train — this measures the increase in
    prediction error (MAE) when a feature's values are independently shuffled,
    computed ON THE TEST SET, so it reflects out-of-sample attribution across
    the 2024->2025 regime shift, not training-set fit. Rows with a NaN target
    are dropped (paired with X via index) before scoring.
    """
    valid = y.notna()
    X_valid, y_valid = X.loc[valid], y.loc[valid]

    result = permutation_importance(
        model, X_valid, y_valid, n_repeats=n_repeats, random_state=random_state, scoring="neg_mean_absolute_error"
    )

    table = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    table["share_of_total"] = table["importance_mean"] / table["importance_mean"].sum()

    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def partial_dependence_data(
    model, X: pd.DataFrame, features: list[str], grid_resolution: int = 50
) -> dict[str, pd.DataFrame]:
    """Partial dependence of model predictions on each of features, one at a time.

    Returns {feature: DataFrame(grid_value, avg_prediction)} — data only, so the
    notebook/visualize.py decide how to plot direction of effect.
    """
    result = {}
    for feature in features:
        pdp = partial_dependence(model, X, [feature], grid_resolution=grid_resolution)
        result[feature] = pd.DataFrame(
            {
                "grid_value": pdp["grid_values"][0],
                "avg_prediction": pdp["average"][0],
            }
        )

    return result
