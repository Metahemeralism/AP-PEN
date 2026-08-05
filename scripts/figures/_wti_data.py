"""
Shared WTI panel loader for the Kalman / real-data-PINN figure scripts.

Mirrors `get_data_wti()` in notebooks/AP-PEN.ipynb exactly
(same CSV, same 8-fixed-maturity-slot pivot) so that the maturity grid here
lines up one-to-one with both the Kalman filter's `taus` and the trained
AP-PEN models' `taus` -- without this, a "PINN vs Kalman" or "observed vs
fitted" comparison would silently be comparing curves sampled at different
maturities. Not imported from the notebook itself (notebooks aren't import-
able modules here); duplicated deliberately, same rationale the rest of this
package already documents for mirroring notebook logic in figure scripts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
WTI_CSV = REPO / "data" / "input" / "real" / "wti_analysis_ready.csv"


def load_wti_panel(path: Path = WTI_CSV, n_maturities: int = 8):
    """Return (dates, t_train, S, taus, log_F_obs, r, is_crisis).

    dates       (n,)      np.datetime64
    t_train     (n,)      years since dates[0]
    S           (n,)      spot price
    taus        (n_mat,)  per-slot mean time-to-maturity
    log_F_obs   (n, n_mat)
    r           (n,)      risk-free rate
    is_crisis   (n,)      bool, 2015/2016 or April 2020
    """
    df = pd.read_csv(path)
    df = df.dropna(subset=["spot", "rate"])  # drops the 2020-04-20 negative-price day
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "tau"])
    df["slot"] = df.groupby("date").cumcount()
    df = df[df["slot"] < n_maturities]
    df = df[df.groupby("date")["slot"].transform("count") == n_maturities]

    tau_matrix = df.pivot(index="date", columns="slot", values="tau").sort_index()
    settle_matrix = df.pivot(index="date", columns="slot", values="settle").sort_index()
    spot_by_date = df.groupby("date")["spot"].first().sort_index()
    rate_by_date = df.groupby("date")["rate"].first().sort_index()

    dates = tau_matrix.index.to_numpy()
    t_train = ((dates - dates[0]) / np.timedelta64(365, "D")).astype(float)
    S = spot_by_date.to_numpy()
    taus = tau_matrix.to_numpy().mean(axis=0)
    log_F_obs = np.log(settle_matrix.to_numpy())
    r = rate_by_date.to_numpy()

    dt = pd.DatetimeIndex(dates)
    is_crisis = np.asarray((dt.year == 2015) | (dt.year == 2016) | ((dt.year == 2020) & (dt.month == 4)))

    return dates, t_train, S, taus, log_F_obs, r, is_crisis
