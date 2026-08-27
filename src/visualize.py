"""Chart builders for phase 3. Every function returns a Figure — no interpretation, no display, no writes."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from src import config


def plot_fuel_mix(df: pd.DataFrame, freq: str = "W") -> go.Figure:
    """Stacked area of generation by fuel, resampled to freq. Nuclear excluded — it is 0 across the whole period."""
    assert len(df) > 17000, f"row count collapsed: {len(df)}"

    fuel_order = [f for f in (config.RENEWABLE + config.CONVENTIONAL) if f not in config.DEAD_COLUMNS]
    resampled = df[fuel_order].resample(freq).mean()

    fig = go.Figure()
    for fuel in fuel_order:
        fig.add_trace(
            go.Scatter(
                x=resampled.index,
                y=resampled[fuel],
                name=fuel,
                mode="lines",
                stackgroup="one",
                line={"width": 0.5, "color": config.FUEL_COLORS[fuel]},
            )
        )
    fig.update_layout(
        title="Generation by fuel over time",
        xaxis_title="time",
        yaxis_title="generation (MWh)",
        width=config.PLOTLY_WIDTH,
        height=config.PLOTLY_HEIGHT,
    )
    return fig


def plot_price_duration_curve(df: pd.DataFrame, price_col: str = "price") -> plt.Figure:
    """Price sorted descending against percentage of hours, with reference lines at 0 and the spike threshold."""
    assert len(df) > 17000, f"row count collapsed: {len(df)}"

    prices = df[price_col].dropna().sort_values(ascending=False).reset_index(drop=True)
    pct_hours = np.linspace(0, 100, len(prices))

    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    ax.plot(pct_hours, prices.values, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(config.SPIKE_THRESHOLD, color="black", linewidth=0.8, linestyle="--")
    ax.text(100, 0, "0", va="bottom", ha="right", fontsize=8)
    ax.text(100, config.SPIKE_THRESHOLD, f"{config.SPIKE_THRESHOLD}", va="bottom", ha="right", fontsize=8)
    ax.set_xlabel("% of hours")
    ax.set_ylabel("price (EUR/MWh)")
    ax.set_title("Price duration curve")
    return fig


def plot_merit_order(df: pd.DataFrame, color_by: str = "hour") -> go.Figure:
    """Scatter of net_residual_load vs price, coloured by hour, month, or renewable_share."""
    assert len(df) > 17000, f"row count collapsed: {len(df)}"
    assert color_by in {"hour", "month", "renewable_share"}, f"unsupported color_by: {color_by}"

    pair = df[["net_residual_load", "price", color_by]].dropna()

    fig = go.Figure(
        go.Scattergl(
            x=pair["net_residual_load"],
            y=pair["price"],
            mode="markers",
            marker={
                "size": 4,
                "opacity": 0.3,
                "color": pair[color_by],
                "colorscale": "Viridis",
                "colorbar": {"title": color_by},
            },
            customdata=pair.index.astype(str).to_numpy().reshape(-1, 1),
            hovertemplate=(
                "timestamp=%{customdata[0]}<br>"
                "net_residual_load=%{x}<br>"
                "price=%{y}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line={"color": "black", "width": 1})
    fig.update_layout(
        title="Net residual load vs price",
        xaxis_title="net residual load (MWh)",
        yaxis_title="price (EUR/MWh)",
        width=config.PLOTLY_WIDTH,
        height=config.PLOTLY_HEIGHT,
    )
    return fig


def plot_renewable_share_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Mean renewable_share by hour x month. Month is display-only — its price correlation flips sign between years."""
    assert len(df) > 17000, f"row count collapsed: {len(df)}"

    pair = df[["renewable_share"]].dropna().copy()
    pair["hour"] = pair.index.hour
    pair["month"] = pair.index.month
    pivot = pair.pivot_table(index="hour", columns="month", values="renewable_share", aggfunc="mean") * 100

    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={"label": "renewable share (%)"}, ax=ax)
    ax.set_xlabel("month")
    ax.set_ylabel("hour")
    ax.set_title("Mean renewable share by hour and month (%)")
    return fig


def plot_negative_price_calendar(df: pd.DataFrame) -> plt.Figure:
    """Calendar heatmap of negative-price hour counts per day (0-24), one panel per year."""
    assert len(df) > 17000, f"row count collapsed: {len(df)}"

    valid = df[["price"]].dropna().copy()
    valid["year"] = valid.index.year
    valid["month"] = valid.index.month
    valid["day"] = valid.index.day
    valid["is_negative"] = (valid["price"] < 0).astype(int)

    daily = valid.groupby(["year", "month", "day"])["is_negative"].sum().reset_index()
    years = sorted(daily["year"].unique())

    fig, axes = plt.subplots(len(years), 1, figsize=(config.FIGSIZE[0], config.FIGSIZE[1] * len(years)))
    axes = np.atleast_1d(axes)

    for ax, year in zip(axes, years):
        pivot = (
            daily.loc[daily["year"] == year]
            .pivot(index="month", columns="day", values="is_negative")
            .reindex(index=range(1, 13), columns=range(1, 32))
        )
        mask = pivot.isna() | (pivot == 0)
        sns.heatmap(pivot, mask=mask, cmap="Reds", ax=ax, linewidths=0.5, linecolor="lightgrey")
        ax.set_title(str(year))
        ax.set_xlabel("day")
        ax.set_ylabel("month")

    fig.tight_layout()
    return fig


def plot_price_distribution(df: pd.DataFrame, price_col: str = "price") -> plt.Figure:
    """Histogram of price on a log-scale y axis, with vertical lines at 0 and the spike threshold."""
    assert len(df) > 17000, f"row count collapsed: {len(df)}"

    prices = df[price_col].dropna()

    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    ax.hist(prices, bins=100)
    ax.set_yscale("log")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(config.SPIKE_THRESHOLD, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("price (EUR/MWh)")
    ax.set_ylabel("count (log scale)")
    ax.set_title("Price distribution")
    return fig
