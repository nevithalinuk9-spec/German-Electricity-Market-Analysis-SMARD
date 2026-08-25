"""Paths, column groups, and constants shared by every module in src/."""

from pathlib import Path

# --- paths ---------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

GENERATION_FILE = RAW / "Actual_generation_202401010000_202601010000_Hour.csv"
CONSUMPTION_FILE = RAW / "Actual_consumption_202401010000_202601010000_Hour.csv"
PRICE_FILE = RAW / "Day-ahead_prices_202401010000_202601010000_Hour.csv"

CLEAN_FILE = INTERIM / "clean_hourly.csv"

# --- parsing ---------------------------------------------------------------

SEP = ";"
DATE_FORMAT = "%b %d, %Y %I:%M %p"
NA_VALUES = ["-"]

# Both raw files use this column name for different things; disambiguate on load.
PUMPED_STORAGE_RAW = "Hydro pumped storage"
GENERATION_RENAME = {PUMPED_STORAGE_RAW: "Pumped storage generation"}
CONSUMPTION_RENAME = {PUMPED_STORAGE_RAW: "Pumped storage consumption"}

PRICE_COLUMN = "Germany/Luxembourg"

# --- column groups ---------------------------------------------------------

RENEWABLE = [
    "Biomass",
    "Hydropower",
    "Wind offshore",
    "Wind onshore",
    "Photovoltaics",
    "Other renewable",
]

# Nuclear is included here for generation-sum purposes only (see DEAD_COLUMNS).
CONVENTIONAL = [
    "Nuclear",
    "Lignite",
    "Hard coal",
    "Fossil gas",
    "Pumped storage generation",
    "Other conventional",
]

VRE = ["Wind offshore", "Wind onshore", "Photovoltaics"]

# 95.96% null, n_unique == 1 after the phase-out. Fine inside a fillna(0) sum;
# never usable as a standalone model feature.
DEAD_COLUMNS = ["Nuclear"]

# --- modelling ---------------------------------------------------------

TRAIN_YEAR = 2024
TEST_YEAR = 2025

PRICE_REGIMES = {
    "negative": (float("-inf"), 0),
    "normal": (0, 150),
    "spike": (150, float("inf")),
}

SPIKE_THRESHOLD = PRICE_REGIMES["spike"][0]

# --- plotting ---------------------------------------------------------

FUEL_COLORS = {
    "Biomass": "#4C9A2A",
    "Hydropower": "#1F77B4",
    "Wind offshore": "#0B5394",
    "Wind onshore": "#6FA8DC",
    "Photovoltaics": "#F1C232",
    "Other renewable": "#93C47D",
    "Nuclear": "#B4A7D6",
    "Lignite": "#7F5539",
    "Hard coal": "#3C3C3C",
    "Fossil gas": "#E06666",
    "Pumped storage generation": "#45818E",
    "Other conventional": "#999999",
}
