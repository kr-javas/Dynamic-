"""
data_loader.py — Load, clean, and merge Indian Railway data files.

This module reads the supplied CSV/JSON files from the data/ folder,
parses timestamps, merges route info with delay records, and returns
clean DataFrames ready for the app and model.
"""

import pandas as pd
import numpy as np
import json
import os

# ---------------------------------------------------------------------------
# Paths — relative so the project is portable
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DELAYS_CSV   = os.path.join(DATA_DIR, "train_routes_delays_Sep2024.csv")
ROUTES_CSV   = os.path.join(DATA_DIR, "train_routes_Sep2024.csv")
EDGES_CSV    = os.path.join(DATA_DIR, "IRN_edges.csv")
ZONES_JSON   = os.path.join(DATA_DIR, "stations_zones_mapping.json")


def load_zones() -> dict:
    """Load station → zone mapping (e.g. 'NDLS' → 'NR')."""
    with open(ZONES_JSON, "r") as f:
        return json.load(f)


def load_routes() -> pd.DataFrame:
    """
    Load train route data: station order, distance, scheduled times.
    Returns DataFrame with columns:
      trainNumber, trainName, station_code, station_name,
      stnSerialNumber, distance, arrivalTime, departureTime
    """
    df = pd.read_csv(ROUTES_CSV)
    # Ensure correct types
    df["trainNumber"]     = df["trainNumber"].astype(str)
    df["stnSerialNumber"] = df["stnSerialNumber"].astype(int)
    df["distance"]        = pd.to_numeric(df["distance"], errors="coerce").fillna(0)
    return df


def _parse_time_12h(series: pd.Series) -> pd.Series:
    """
    Parse '08:00 AM' style strings into datetime.time objects.
    Returns NaT-equivalent (None) for unparseable values.
    """
    return pd.to_datetime(series, format="%I:%M %p", errors="coerce").dt.time


def load_delays() -> pd.DataFrame:
    """
    Load the main delay dataset and parse time columns.
    Returns DataFrame with columns:
      train, date, station,
      sch_arr (time), act_arr (time), arr_delay (float, minutes),
      sch_dep (time), act_dep (time), dep_delay (float, minutes)
    """
    df = pd.read_csv(DELAYS_CSV)

    # --- Basic type cleaning ---
    df["train"]   = df["train"].astype(str)
    df["date"]    = pd.to_datetime(df["date"], errors="coerce")

    # Parse delay columns as floats (some may be missing)
    for col in ["arr_delay", "dep_delay"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse 12-hour time strings
    for col in ["sch_arr", "act_arr", "sch_dep", "act_dep"]:
        df[col] = _parse_time_12h(df[col])

    # Drop rows with no date (unusable)
    df = df.dropna(subset=["date"])

    return df


def merge_delays_with_routes(delays: pd.DataFrame,
                              routes: pd.DataFrame) -> pd.DataFrame:
    """
    Join delay records with route info to get station sequence number
    and distance for each delay record.

    Join keys: train number + station code.
    After join, sort by (train, date, stnSerialNumber) to guarantee
    correct route order within each journey.
    """
    merged = delays.merge(
        routes[["trainNumber", "station_code", "station_name",
                "trainName", "stnSerialNumber", "distance"]],
        left_on=["train", "station"],
        right_on=["trainNumber", "station_code"],
        how="left"
    )

    # Fill missing serial numbers with a large value so they sort last
    merged["stnSerialNumber"] = merged["stnSerialNumber"].fillna(9999).astype(int)
    merged["distance"]        = merged["distance"].fillna(0)

    # Sort: each journey in route order
    merged = merged.sort_values(
        ["train", "date", "stnSerialNumber"]
    ).reset_index(drop=True)

    return merged


def load_edges() -> pd.DataFrame:
    """Load the railway network edges (station connectivity graph)."""
    return pd.read_csv(EDGES_CSV)


def get_journey(merged_df: pd.DataFrame,
                train_number: str,
                journey_date: str) -> pd.DataFrame:
    """
    Extract a single journey: one train on one date, in route order.

    Parameters
    ----------
    merged_df    : The merged delays+routes DataFrame.
    train_number : e.g. '12303'
    journey_date : e.g. '2024-09-02'

    Returns
    -------
    DataFrame of that journey's station-by-station records, sorted by
    station sequence number.
    """
    mask = (
        (merged_df["train"] == str(train_number)) &
        (merged_df["date"]  == pd.Timestamp(journey_date))
    )
    journey = merged_df.loc[mask].copy()
    journey = journey.sort_values("stnSerialNumber").reset_index(drop=True)
    return journey


def get_available_trains(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary of available trains with their names and journey
    counts, useful for the dropdown selector.
    """
    summary = (
        merged_df
        .groupby("train")
        .agg(
            trainName=("trainName", "first"),
            journey_count=("date", "nunique"),
        )
        .reset_index()
        .sort_values("train")
    )
    # Clean up missing names
    summary["trainName"] = summary["trainName"].fillna("Unknown")
    return summary


def get_dates_for_train(merged_df: pd.DataFrame,
                        train_number: str) -> list:
    """Return sorted list of journey dates available for a given train."""
    dates = (
        merged_df
        .loc[merged_df["train"] == str(train_number), "date"]
        .dropna()
        .dt.date
        .unique()
    )
    return sorted(dates)


# ---------------------------------------------------------------------------
# Convenience: load everything in one call
# ---------------------------------------------------------------------------
def load_all():
    """
    Load and merge all data. Returns (merged_df, routes_df, zones_dict).

    merged_df : delay records enriched with route info, sorted by journey.
    routes_df : raw route table.
    zones     : station → zone mapping dict.
    """
    delays = load_delays()
    routes = load_routes()
    zones  = load_zones()

    merged = merge_delays_with_routes(delays, routes)
    return merged, routes, zones
