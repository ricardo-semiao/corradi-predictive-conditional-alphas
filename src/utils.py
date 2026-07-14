
# Setup ------------------------------------------------------------------------

import os
import sys
if os.path.basename(os.getcwd()) == "src": os.chdir("..")
if os.getcwd() not in sys.path: sys.path.insert(0, os.getcwd())

import re
import random

import polars as pl
import numpy as np

import plotnine as gg
from plotnine import ggplot, aes

from numpy.typing import NDArray

from src.parameters import PARAMETERS as PARS
from src.parameters import _counts, _trading_days



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



# Graphs -----------------------------------------------------------------------

def plot_day_fullness_river(data: pl.DataFrame, n: int | str = "") -> gg.ggplot:
    g_fill_colors = {
        "10% - 90%": "#ffb399",
        "20% - 80%": "#ff6699",
        "30% - 70%": "#ff0000",
        "40% - 60%": "#ffffff",
        "median": "black",
        "non-full factors": "blue",
        "non-full others": "gray"
    }

    data_factor = (
        pl.DataFrame({"d": _trading_days, "n": _counts})
        .filter(pl.col("n") != (PARS.blocks_1min + 1)) # Factor data +1 minute
        .with_columns(pl.col("n") / (PARS.blocks_1min + 1))
    )
    days_low_obs = (
        data
        .with_columns(median_rw = pl.col("median").rolling_mean(5))
        .filter(pl.col("median") - pl.col("median_rw") <= - 0.1)
    )

    g = (
        ggplot(data, aes("d", group = 1)) +
        gg.geom_ribbon(aes(ymin = "q1", ymax = "q9", fill = "'10% - 90%'")) +
        gg.geom_ribbon(aes(ymin = "q2", ymax = "q8", fill = "'20% - 80%'")) +
        gg.geom_ribbon(aes(ymin = "q3", ymax = "q7", fill = "'30% - 70%'")) +
        gg.geom_ribbon(aes(ymin = "q4", ymax = "q6", fill = "'40% - 60%'")) +
        gg.geom_line(aes(y = "median", color = "'median'"), size = 0.25) +
        gg.geom_point(aes("d", "n", color = "'non-full factors'"), data_factor, size = 1.5) +
        gg.geom_point(aes("d", "median", color = "'non-full others'"), days_low_obs, size = 1.5) +
        gg.scale_x_date(date_labels = "%Y", date_breaks = "1 year") +
        gg.scale_fill_manual(values = g_fill_colors) +
        gg.scale_color_manual(values = g_fill_colors) +
        gg.scale_y_continuous(labels = lambda bs: ["{:.0%}".format(b) for b in bs]) +
        gg.labs(
            title = "Proportion of minutes with data within each day",
            subtitle = (
                f"Quantiles calculated each day across all {n}"
                " stocks"
            ),
            caption = "Non-full others: median $0.1$ points below its 5-day rolling average.",
            y = "Proportion of minutes with data", x = "Day",
            fill = "Quantiles", color = " "
        ) +
        gg.theme_bw() +
        gg.theme(
            axis_text_x = gg.element_text(rotation = 45, hjust = 1),
            figure_size = (6.5, 5),
            legend_position = "bottom",
            legend_box = "vertical"
        )
    )

    return g


def plot_day_fullness_box(
    data: pl.DataFrame, min: int = 1, n: int | str = ""
) -> gg.ggplot:
    def mean_perc_low_obs(x: int) -> pl.Series:
        res = (
            data
            .group_by("permno")
            .agg(((pl.col("obs_count") >= x) / pl.len()).sum())
            ["obs_count"]
        )
        return res

    if min == 1:
        k = (PARS.blocks_5min // 2) # 36
        divs = range(PARS.blocks_1min // k)
    elif min == 5:
        k = 6
        divs = range(PARS.blocks_5min // k)
    else:
        raise ValueError("min must be 1 or 5")

    gdata = (
        pl.DataFrame({
            f"below_{(i + 1) * k}": mean_perc_low_obs((i + 1) * k)
            for i in divs
        })
        .unpivot(on = None, variable_name = "n_obs", value_name = "perc_low_obs")
        .with_columns(
            pl.col("n_obs").str.replace_all("below_", "")
            .cast(pl.Enum(str((i + 1) * k) for i in divs))
        )
    )

    g = (
        gg.ggplot(gdata, gg.aes("n_obs", "perc_low_obs")) +
        gg.geom_boxplot() +
        gg.geom_hline(yintercept = 0.975, linetype = "dashed", color = "red") +
        gg.scale_y_continuous(labels = lambda bs: ["{:.0%}".format(b) for b in bs]) +
        gg.labs(
            title = "Distribution of days' observation count ($n$)",
            subtitle = f"Across all {n} stocks",
            caption = "Red line: $97.5\\%$.",
            y = "Fraction of days with $n \\geq \\tau$",
            x = "$\\tau$"
        ) +
        gg.theme_bw()
    )

    return g

def plot_day_fullness_hist(data: pl.DataFrame) -> gg.ggplot:
    g = (
        ggplot(data, aes("obs_count", gg.after_stat("density"))) +
        gg.geom_histogram(bins = 40) +
        gg.labs(
            title = "Distribution of days' observation count ($n$)",
            subtitle = f"Across all {data.shape[0]:,} stocks-day pairs",
            x = "$n$", y = "Density"
        ) +
        gg.theme_bw()
    )

    return g


def plot_prices_returns(
    data_prices: pl.DataFrame, data_returns: pl.DataFrame
) -> gg.ggplot:
    datas = {
        key: (
            df
            .group_by(["permno", pl.col("ts_min_ny") // (60 * 24 * 30)])
            .agg(pl.col(key).last() if key == "price" else pl.col(key).sum())
            .with_columns(
                pl.from_epoch(pl.col("ts_min_ny") * 30, time_unit = "d"),
                variable = pl.lit(key.replace("_", " ")).cast(pl.Enum(["price", "log return"]))
            )
        )
        for key, df in {"price": data_prices, "log_return": data_returns}.items()
    }

    g = (
        ggplot(mapping = aes(x = "ts_min_ny")) +
        gg.geom_line(aes(y = "price"), datas["price"]) +
        gg.geom_segment(
            aes(
                x = "ts_min_ny", xend = "ts_min_ny", y = 0, yend = "log_return",
                color = "log_return > 0"
            ),
            datas["log_return"]
        ) +
        gg.facet_grid("variable", "permno", scales = "free") +
        gg.scale_x_datetime(date_labels = "%Y", date_breaks = "1 year") +
        gg.scale_color_manual(values = ["red", "green"]) +
        gg.labs(
            title = "Price and log returns across time",
            subtitle = f"Each column is a randomly selected PERMNOs",
            y = "Value", x = "Time"
        ) +
        gg.theme_bw() +
        gg.theme(
            axis_text_x = gg.element_text(rotation = 45, hjust = 1),
            legend_position = "none"
        )
    )

    return g



# Debugging --------------------------------------------------------------------

if __name__ == "__main__":
    data_days = pl.read_csv(
        "data/day_counts.csv"
    ).with_columns(
        pl.from_epoch(pl.col("d"), time_unit = "d")
    )

    data_days_fill = (
        data_days.with_columns(pl.col("obs_count") / PARS.blocks_1min)
        .group_by("d")
        .agg(
            median = pl.median("obs_count").alias("median"),
            *[
                pl.quantile("obs_count", q / 10).alias(f"q{q}")
                for q in [4, 6, 3, 7, 2, 8, 1, 9]
            ]
        )
    )

    plot_day_fullness(data_days_fill)
