
# Setup ------------------------------------------------------------------------

import os
import sys
if os.path.basename(os.getcwd()) == "src": os.chdir("..")
if os.getcwd() not in sys.path: sys.path.insert(0, os.getcwd())

import re
import random
import numpy as np

from numpy.typing import NDArray

from src.parameters import PARAMETERS as PARS



# Datetime functions -----------------------------------------------------------

def format_days_byyear(days: NDArray[np.datetime64], fmt: str = "%m-%d") -> str:
    days_fmt = ""
    for y in np.unique(days.astype("datetime64[Y]")):
        days_y = days[days.astype("datetime64[Y]") == y]
        if days_y.size > 0:
            days_fmt += (
                f"\n- {days_y[0].item().strftime("%Y")}: "
                + ", ".join(map(lambda x: x.item().strftime(fmt), days_y))
            )
    return days_fmt


def obs_per_stock_day(
    n: int | None = None
) -> NDArray[np.int_]: # shape: (n, len(days))
    key_errors = {"out_bounds": set(), "in_bounds": set()}
    folder = "data/stocks_1min_returns"
    paths = {
        path[4:9]: os.path.join(folder, path)
        for path in os.listdir(folder)
        if re.match(r"perm[0-9]{5}.csv", path)
    }

    n = len(paths) if n is None else min(n, len(paths))
    paths = dict(random.sample(list(paths.items()), k = n))
    if not paths:
        raise Exception(f"No matching files found in folder {folder}")
    
    days = tuple(str(d) for d in PARS.trading_days)
    stocks_day_obs = np.zeros((len(paths), len(days)), dtype = float)
    day_obs_base = {day: 0 for day in days}

    for idx, path in enumerate(paths.values()):
        if idx % 100 == 0:
            print(f"Processing file {idx + 1}/{len(paths)}: {path}")
        day_obs = day_obs_base.copy()
        # Assumes 1 line per minute. If not, create a set for the minutes, then
        # format them it days before passing to Counter
        range = ["", ""]

        with open(path, "rb") as f: # Binary for speed
            f.readline() # Skip header
            line = f.readline()
            # First characters are the datetime in YYYY-MM-DD hh:mm:ss format
            day = line[:10].decode("ascii")
            range[0] = day
            while line:
                day = line[:10].decode("ascii")
                try:
                    day_obs[day] += 1
                except KeyError:
                    if day < days[0] or day > days[-1]:
                        key_errors["out_bounds"].add(day)
                    else:
                        key_errors["in_bounds"].add(day)

                line = f.readline()
            else:
                range[1] = day

        day_arr = (
            np.array(list(day_obs.values()), dtype = float)
            / PARS.blocks_1min
        )
        day_arr[:np.searchsorted(days, range[0], side = "left")] = np.nan
        day_arr[np.searchsorted(days, range[1], side = "right"):] = np.nan

        stocks_day_obs[idx, :] = day_arr

    if key_errors["in_bounds"]:
        raise KeyError(
            f"\nWarning: {len(key_errors['in_bounds'])} in-bounds days not " +
            "found in trading days list. They were:" +
            format_days_byyear(
                np.array(list(key_errors["in_bounds"]), dtype = "datetime64[D]")
            )
        )


    if key_errors["out_bounds"]:
        raise KeyError(
            f"Error: {len(key_errors['out_bounds'])} out-of-bounds days not "
            "found in trading days list. They were:"
            "\n- ".join(sorted(set(key_errors["out_bounds"])))
        )

    return stocks_day_obs



# Debugging --------------------------------------------------------------------

if __name__ == "__main__":
    obs = obs_per_stock_day(2)
