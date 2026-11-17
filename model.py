"""
model.py — Baseline and Random Forest delay-prediction logic.

Three prediction strategies:
  1. Timetable baseline  — predicts zero delay (train is always on time).
  2. Persistence baseline — the latest observed delay continues unchanged.
  3. Random Forest model  — uses features available up to the current step.

All predictions output *delay in minutes*; the app converts that to ETA.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit


# ============================================================================
# 1.  BASELINES
# ============================================================================

def timetable_baseline(n_upcoming: int) -> np.ndarray:
    """
    Timetable baseline: predict zero delay for every upcoming station.
    (Assumes the train always runs exactly on schedule.)
    """
    return np.zeros(n_upcoming)


def persistence_baseline(last_known_delay: float,
                         n_upcoming: int) -> np.ndarray:
    """
    Persistence baseline: the most-recently observed delay will continue
    unchanged for all remaining stations.
    """
    return np.full(n_upcoming, last_known_delay)


# ============================================================================
# 2.  FEATURE ENGINEERING  (for Random Forest)
# ============================================================================

def _time_to_minutes(t) -> float:
    """Convert a datetime.time object to minutes since midnight."""
    if pd.isna(t) or t is None:
        return np.nan
    return t.hour * 60 + t.minute


def build_features_for_training(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from the full merged dataset for model training.

    For each station row we compute:
      - stn_seq        : station sequence number on the route
      - distance       : route distance from origin (km)
      - sch_arr_min    : scheduled arrival as minutes since midnight
      - prev_delay     : arrival delay at the *previous* station in the
                         same journey (NaN for the first station)
      - cumul_delay    : running average of delays seen so far
      - journey_id     : (train, date) used for group-based splitting

    Target:
      - arr_delay      : arrival delay in minutes (what we predict)

    IMPORTANT: prev_delay uses only past information — no data leakage.
    """
    df = merged_df.copy()

    # Journey identifier for group splitting
    df["journey_id"] = df["train"] + "_" + df["date"].astype(str)

    # Scheduled arrival in minutes since midnight
    df["sch_arr_min"] = df["sch_arr"].apply(_time_to_minutes)

    # Previous station's delay (shift within each journey group)
    df["prev_delay"] = df.groupby("journey_id")["arr_delay"].shift(1)

    # Cumulative mean delay up to (but NOT including) the current station
    # We use expanding().mean() on the shifted series
    df["cumul_delay"] = (
        df.groupby("journey_id")["arr_delay"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    # Station sequence (renamed for clarity)
    df["stn_seq"] = df["stnSerialNumber"]

    return df


FEATURE_COLS = ["stn_seq", "distance", "sch_arr_min", "prev_delay", "cumul_delay"]
TARGET_COL   = "arr_delay"


# ============================================================================
# 3.  TRAIN / TEST SPLIT  (by journey, not by row)
# ============================================================================

def split_by_journey(df: pd.DataFrame, test_size: float = 0.2,
                     random_state: int = 42):
    """
    Split data by journey_id so that all stations from one journey stay
    together in either train or test set.  This prevents data leakage
    from future stations leaking into the training set.
    """
    # Remove rows where target or key features are missing
    clean = df.dropna(subset=[TARGET_COL, "prev_delay"]).copy()

    if len(clean) < 20:
        return clean, pd.DataFrame(), None, None

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size,
                           random_state=random_state)
    groups = clean["journey_id"]

    train_idx, test_idx = next(gss.split(clean, groups=groups))
    return clean.iloc[train_idx], clean.iloc[test_idx], train_idx, test_idx


# ============================================================================
# 4.  RANDOM FOREST MODEL
# ============================================================================

class DelayPredictor:
    """
    Wraps a scikit-learn RandomForestRegressor.

    Usage:
        predictor = DelayPredictor()
        predictor.train(merged_df)        # one-time training
        preds = predictor.predict(X_new)  # predict delay in minutes
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 12,
                 random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,            # use all CPU cores
        )
        self.is_trained = False
        self.train_mae  = None
        self.test_mae   = None
        self.n_train    = 0
        self.n_test     = 0

    def train(self, merged_df: pd.DataFrame):
        """
        Build features, split by journey, and fit the Random Forest.
        Stores train/test MAE for display in the app.
        """
        feat_df = build_features_for_training(merged_df)
        train_df, test_df, _, _ = split_by_journey(feat_df)

        if len(train_df) < 10:
            # Not enough data to train
            self.is_trained = False
            return

        X_train = train_df[FEATURE_COLS].fillna(0)
        y_train = train_df[TARGET_COL]

        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.n_train    = len(X_train)

        # Training MAE
        self.train_mae = np.mean(np.abs(self.model.predict(X_train) - y_train))

        # Test MAE (if we have test data)
        if len(test_df) > 0:
            X_test = test_df[FEATURE_COLS].fillna(0)
            y_test = test_df[TARGET_COL]
            self.test_mae = np.mean(np.abs(self.model.predict(X_test) - y_test))
            self.n_test   = len(X_test)

    def predict_upcoming(self, journey_df: pd.DataFrame,
                         revealed_count: int) -> np.ndarray:
        """
        Predict arrival delay for stations *after* the revealed ones.

        Parameters
        ----------
        journey_df     : Full journey DataFrame (all stations, in order).
        revealed_count : How many stations have been revealed so far.

        Returns
        -------
        Array of predicted delays (minutes) for each upcoming station.
        """
        if not self.is_trained:
            # Fallback: return zeros (same as timetable baseline)
            n_upcoming = len(journey_df) - revealed_count
            return np.zeros(max(n_upcoming, 0))

        upcoming = journey_df.iloc[revealed_count:].copy()
        if len(upcoming) == 0:
            return np.array([])

        # Build features for upcoming stations using only revealed info
        # Last known delay from the most-recently revealed station
        if revealed_count > 0:
            last_delay = journey_df.iloc[revealed_count - 1]["arr_delay"]
            if pd.isna(last_delay):
                last_delay = 0.0

            # Average delay of all revealed stations
            revealed_delays = journey_df.iloc[:revealed_count]["arr_delay"].dropna()
            avg_delay = revealed_delays.mean() if len(revealed_delays) > 0 else 0.0
        else:
            last_delay = 0.0
            avg_delay  = 0.0

        upcoming["prev_delay"]  = last_delay
        upcoming["cumul_delay"] = avg_delay
        upcoming["stn_seq"]     = upcoming["stnSerialNumber"]
        upcoming["sch_arr_min"] = upcoming["sch_arr"].apply(_time_to_minutes)

        X = upcoming[FEATURE_COLS].fillna(0)
        return self.model.predict(X)
