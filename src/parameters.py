
# Setup ------------------------------------------------------------------------

import numpy as np
import exchange_calendars as ecals
import pandas_market_calendars as mcal

from typing import Final, NamedTuple
from numpy.typing import NDArray



# Dates ------------------------------------------------------------------------

_date_start = np.datetime64("2001-01-01 00:00:00", "m")
_date_end = np.datetime64("2017-12-29 23:59:59", "m")

_time_start = np.datetime64("0000-01-01T09:30", "m")
_time_end = np.datetime64("0000-01-01T16:00", "m")

_blocks_1min = (_time_end - _time_start).astype("int") + 1 # 391
_blocks_5min = ((_time_end - _time_start).astype("int") + 1) // 5 # 78



# Trading days -----------------------------------------------------------------

# Factor data:
_days_factor = np.loadtxt(
    "data/ff6_1min_returns.csv",
    delimiter = ",", skiprows = 1, usecols = 0, dtype = "datetime64[D]"
)

_trading_days, _counts = np.unique(
    _days_factor[(_days_factor >= _date_start) & (_days_factor <= _date_end)],
    return_counts = True
)


# Only full trading days (probably removes the added calendar days):
_trading_days_full = _trading_days[_counts == _blocks_1min]
_trading_days_full.flags.writeable = False


# Calendars seem to have extra days, lets accept them in:
_days_ecals = ecals.get_calendar(
    "XNYS"
).sessions_in_range( # Reduce range available
    "2006-07-11", _date_end.astype("datetime64[D]").item()
)
_days_mcal = mcal.get_calendar(
    "NYSE"
).valid_days(
    start_date = _date_start, end_date = _date_end
)

_days_cals = np.union1d(
    np.array(_days_ecals, dtype = "datetime64[D]"),
    np.array(_days_mcal, dtype = "datetime64[D]")
)

_trading_days = np.sort(np.union1d(_trading_days, _days_cals))
_trading_days.flags.writeable = False



# Parameters object ------------------------------------------------------------

class ProjectParameters(NamedTuple):
    date_start: np.datetime64
    date_end: np.datetime64
    time_start: np.datetime64
    time_end: np.datetime64
    blocks_1min: np.int64
    blocks_5min: np.int64
    trading_days: NDArray[np.datetime64]
    trading_days_full: NDArray[np.datetime64]

PARAMETERS: Final[ProjectParameters] = ProjectParameters(
    date_start = _date_start,
    date_end = _date_end,
    time_start = _time_start,
    time_end = _time_end,
    blocks_1min = _blocks_1min,
    blocks_5min = _blocks_5min,
    trading_days = _trading_days,
    trading_days_full = _trading_days_full
)
