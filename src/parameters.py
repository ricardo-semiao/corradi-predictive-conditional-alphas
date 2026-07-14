
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
_time_end = np.datetime64("0000-01-01T15:59", "m")

_blocks_1min = (_time_end - _time_start).astype("int") + 1 # 390
_blocks_5min = ((_time_end - _time_start).astype("int") + 1) // 5 # 78



# Trading days -----------------------------------------------------------------

# Factor data:
_days_factor = np.loadtxt(
    "data/ff6_1min_returns_raw.csv",
    delimiter = ",", skiprows = 1, usecols = 0, dtype = "datetime64[D]"
)

_trading_days, _counts = np.unique(
    _days_factor[(_days_factor >= _date_start) & (_days_factor <= _date_end)],
    return_counts = True
)


# Only full trading days (probably removes the added calendar days):
_trading_days_factors = _trading_days[_counts == (_blocks_1min + 1)]
_trading_days_factors.flags.writeable = False
# Factor data has +1 minute observation


# Calendars seem to have extra days, lets accept them in:
_days_ecals = ecals.get_calendar(
    "XNYS"
)
_days_ecals = _days_ecals.sessions_in_range( # Reduce range available
    max(
        _days_ecals.sessions[0].to_numpy().astype("datetime64[D]").item(),
        _date_start.astype("datetime64[D]").item()
    ),
    min(
        _days_ecals.sessions[0].to_numpy().astype("datetime64[D]").item(),
        _date_end.astype("datetime64[D]").item()
    )
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

_trading_days_cal = np.sort(np.union1d(_trading_days, _days_cals))
_trading_days_cal.flags.writeable = False


# Stock data showed other non-full trading days, lets remove them
_trading_days_out_stocks = np.array(
    [
        '2001-06-08', '2001-07-03', '2001-11-23', '2001-12-24', '2002-07-05',
        '2002-09-11', '2002-11-29', '2002-12-24', '2003-07-03', '2003-08-15',
        '2003-11-28', '2003-12-24', '2003-12-26', '2004-11-26', '2005-11-25',
        '2005-12-23', '2006-07-03', '2006-11-24', '2007-07-03', '2007-11-23',
        '2007-12-24', '2008-07-03', '2008-11-28', '2008-12-24', '2008-12-26',
        '2009-11-27', '2009-12-24', '2010-11-26', '2010-12-27', '2011-08-15',
        '2011-11-25', '2011-12-23', '2011-12-27', '2012-07-03', '2012-11-12',
        '2012-11-23', '2012-12-24', '2013-07-03', '2013-08-22', '2013-11-29',
        '2013-12-24', '2014-07-03', '2014-11-28', '2014-12-24', '2014-12-26',
        '2015-10-12', '2015-11-27', '2015-12-24', '2016-11-25', '2016-12-23',
        '2017-07-03', '2017-11-24'
    ],
    dtype = "datetime64[D]"
)
_trading_days_final = np.setdiff1d(_trading_days_factors, _trading_days_out_stocks)
_trading_days_final.flags.writeable = False



# Parameters object ------------------------------------------------------------

class ProjectParameters(NamedTuple):
    date_start: np.datetime64
    date_end: np.datetime64
    time_start: np.datetime64
    time_end: np.datetime64
    blocks_1min: int
    blocks_5min: int
    trading_days: NDArray[np.datetime64]
    trading_days_factors: NDArray[np.datetime64]
    trading_days_final: NDArray[np.datetime64]

PARAMETERS: Final[ProjectParameters] = ProjectParameters(
    date_start = _date_start,
    date_end = _date_end,
    time_start = _time_start,
    time_end = _time_end,
    blocks_1min = _blocks_1min.item(),
    blocks_5min = _blocks_5min.item(),
    trading_days = _trading_days_cal,
    trading_days_factors = _trading_days_factors,
    trading_days_final = _trading_days_final
)
