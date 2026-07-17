
# Setup ------------------------------------------------------------------------

import os
import sys
if os.path.basename(os.getcwd()) == "src": os.chdir("..")
if os.getcwd() not in sys.path: sys.path.insert(0, os.getcwd())

import re
import random

import polars as pl
import numpy as np

from numpy.typing import NDArray

from src.parameters import PARAMETERS as PARS
from src.parameters import _counts, _trading_days



# Datetime functions -----------------------------------------------------------

def format_days_byyear(days: NDArray[np.datetime64], fmt: str = "%m-%d") -> str:
    days_fmt = ""

    for y in np.unique(days.astype("datetime64[Y]")):
        days_y = days[days.astype("datetime64[Y]") == y]

        if days_y.size > 0:
            items_str = f"\n- {days_y[0].item().strftime("%Y")}: "
            lsize = len(items_str)
            i = 0

            while i < days_y.size:
                if lsize > 80:
                    items_str = items_str[:-2] + "\n    "
                    lsize = 0
                item = f"{days_y[i].item().strftime(fmt)}, "
                items_str += item
                lsize += len(item)
                i += 1

            days_fmt += items_str
    return days_fmt


def fill_missing_days(data: pl.DataFrame) -> pl.DataFrame:
    stock_lifetimes = (
        data
        .group_by("permno")
        .agg([
            pl.col("d").min().alias("first_obs"),
            pl.col("d").max().alias("last_obs")
        ])
    )

    data_filled = (
        # Grid of all permnos X trading days:
        data.select(pl.col("permno").unique())
        .join(pl.DataFrame({"d": PARS.trading_days_final.astype("int64")}), how = "cross")
        # Remove rows outside the permno lifetime:
        .join(stock_lifetimes, on = "permno", how = "left")
        .filter(
            (pl.col("d") >= pl.col("first_obs"))
            & (pl.col("d") <= pl.col("last_obs"))
        )
        # Add obs_count data, filling missing days with 0:
        .join(data, on = ["permno", "d"], how = "left")
        .with_columns(pl.col("obs_count").fill_null(0))
        .select(["permno", "d", "obs_count"])
        .sort(["permno", "d"])
    )

    return data_filled



# PCA helpers ------------------------------------------------------------------

def pca_high_freq_validate(
    X: NDArray, k: int | None, u: NDArray | None,
    evectors_start: NDArray | None = None
) -> None:
    if not isinstance(X, np.ndarray):
        raise TypeError("`X` must be a numpy `ndarray`.")
    if X.ndim != 2:
        raise ValueError("`X` must be a 2D array (matrix).")
    if not X.dtype in [np.float16, np.float32, np.float64]:
        raise TypeError("`X` must have a float dtype.")
    if not np.all(np.isfinite(X)):
        raise ValueError("`X` must not contain NaN or Inf values.")

    T, d = X.shape

    if k is not None:
        if not isinstance(k, int):
            raise ValueError("`k` must be an `int`.")
        if k <= d:
            raise ValueError("`k` must be bigger than `X.shape[1]`.")
        if k < T:
            raise ValueError("`k` must be bigger or equal to `X.shape[0]`.")

    if u is not None:
        if not isinstance(u, np.ndarray):
            raise TypeError("`u` must be a numpy `ndarray`.")
        if not X.dtype in [np.float16, np.float32, np.float64]:
            raise TypeError("`u` must have a float dtype.")
        if u.ndim != 1 or u.shape[0] != d:
            raise ValueError("`u` must have (1 x `X.shape[1]`) dimensions.")
        if not np.all(np.isfinite(u)):
            raise ValueError("`u` must not contain NaN or Inf values.")
        if not np.all(u > 0):
            raise ValueError("`u` must contain only strictly positive values.")

    if evectors_start is not None:
        if not isinstance(evectors_start, np.ndarray):
            raise TypeError("`evectors_start` must be a numpy `ndarray`.")
        if not X.dtype in [np.float16, np.float32, np.float64]:
            raise TypeError("`evectors_start` must have a float dtype.")
        es = evectors_start.shape
        if evectors_start.ndim != 2 or es != (d, d):
            raise ValueError(
                "`evectors_start` must have (`X.shape[1]` x "
                "`X.shape[1]`) dimensions."
            )

    return None

# TODO: Warn if u is too big given X general level
# TODO: Implement k and u bounds from proposition 3
# varsigma = np.log(k) / np.log(T)
# k_conforms = (gamma / 2) < varsigma < 0.5
# C = 3.0 * np.sqrt((np.pi / 2) * np.sum(np.abs(X[1:] * X[:-1]), axis=0))
# varpi = np.log(np.maximum(u / C, 1e-12)) / np.log(1 / T)
# u_conforms = np.all((varpi >= (1 - varsigma) / (2 - gamma)) & (varpi < 0.5))



# Debugging --------------------------------------------------------------------

if __name__ == "__main__":
    ...
