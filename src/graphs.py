
# Setup ------------------------------------------------------------------------

import polars as pl
import numpy as np

import matplotlib as mpl
import plotnine as gg
from plotnine import ggplot, aes

from collections.abc import Collection

from src.parameters import PARAMETERS as PARS
from src.parameters import _counts, _trading_days



# Helpers ----------------------------------------------------------------------

# Create a color palette from a mlp colormap with specified limits
def palette_cmap(
    cmap: str, n: int, lims: tuple[float, float] = (0, 1),
    rev: bool = False
) -> list[str]:
    base_cmap = mpl.colormaps[cmap]
    color_points = np.linspace(lims[0], lims[1], n)
    res = [mpl.colors.to_hex(color) for color in base_cmap(color_points)]
    return res[::-1] if rev else res

def palette_rescaler(to = (0, 1)):
    def rescaler(x, _from = None):
        __from = (np.min(x), np.max(x)) if _from is None else _from
        return np.interp(x, __from, to)
    return rescaler



# Stocks -----------------------------------------------------------------------

# Calculate stock day fullness
def mean_perc_low_obs(data: pl.DataFrame, x: int) -> pl.Series:
    res = (
        data
        .group_by("permno")
        .agg(((pl.col("obs_count") >= x) / pl.len()).sum())
        ["obs_count"]
    )
    return res


def day_fullness_river(data: pl.DataFrame, n: int | str = "") -> ggplot:
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
        gg.scale_y_continuous(labels = lambda bs: [f"{b:.0%}" for b in bs]) +
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


def day_fullness_box(
    data: pl.DataFrame, min: int = 1, n: int | str = ""
) -> ggplot:
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
            f"below_{(i + 1) * k}": mean_perc_low_obs(data, (i + 1) * k)
            for i in divs
        })
        .unpivot(on = None, variable_name = "n_obs", value_name = "perc_low_obs")
        .with_columns(
            pl.col("n_obs").str.replace_all("below_", "")
            .cast(pl.Enum(str((i + 1) * k) for i in divs))
        )
    )

    g = (
        ggplot(gdata, gg.aes("n_obs", "perc_low_obs")) +
        gg.geom_boxplot() +
        gg.geom_hline(yintercept = 0.975, linetype = "dashed", color = "red") +
        gg.scale_y_continuous(labels = lambda bs: [f"{b:.0%}" for b in bs]) +
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


def day_fullness_matrix(
    data: pl.DataFrame, min_obs: Collection[int], min_days: Collection[float],
    n: int | str = ""
) -> ggplot:
    res = np.full((len(min_obs), len(min_days)), np.nan)
    for i, min in enumerate(min_obs):
        for j, days in enumerate(min_days):
            res[i, j] = (
                data
                .group_by("permno")
                .agg(((pl.col("obs_count")  >= min).sum() / pl.len()) >= days)
                ["obs_count"].sum()
            )

    cols = [str(d) for d in min_days]
    gdata = (
        pl.DataFrame(res, schema = cols)
        .with_columns(pl.Series("min_obs", min_obs))
        .unpivot(
            on = cols, index = "min_obs",
            variable_name = "min_days", value_name = "n_stocks"
        )
        .with_columns(pl.col("min_days").cast(pl.Float64))
    )

    g = (
        ggplot(gdata, gg.aes("min_obs", "min_days", fill = "n_stocks")) +
        gg.geom_raster() +
        gg.geom_text(aes(label = "n_stocks"), size = 6) +
        gg.scale_y_continuous(labels = lambda bs: [f"{b:.1%}" for b in bs], breaks = list(min_days)) +
        gg.scale_x_continuous(breaks = list(min_obs)) +
        gg.scale_fill_continuous(rescaler = palette_rescaler((0.2, 1))) +
        gg.labs(
            title = "Stocks with $p$% days where $\\tau$ minutes had trades",
            subtitle = f"Across all {n} stocks",
            fill = "Stocks count",
            y = "$p$", x = "$\\tau$"
        ) +
        gg.theme_bw()
    )

    return g


def day_fullness_hist(data: pl.DataFrame) -> ggplot:
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


def prices_returns(
    data_prices: pl.DataFrame, data_returns: pl.DataFrame
) -> ggplot:
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
            aes(xend = "ts_min_ny", y = 0, yend = "log_return", color = "log_return > 0"),
            datas["log_return"]
        ) +
        gg.facet_grid("variable", "permno", scales = "free") +
        gg.scale_x_datetime(date_labels = "%Y", date_breaks = "1 year") +
        gg.scale_color_manual(values = ["red", "green"]) +
        gg.labs(
            title = "Price and log returns across time",
            subtitle = "Each column is a randomly selected PERMNOs",
            y = "Value", x = "Time"
        ) +
        gg.theme_bw() +
        gg.theme(
            axis_text_x = gg.element_text(rotation = 45, hjust = 1),
            legend_position = "none"
        )
    )

    return g



# Factors ----------------------------------------------------------------------

def factors(
    data: pl.DataFrame, cols: dict[str, str],
    segment_color: str | None = "value > 0",
    title: str = "Minute log returns of the Fama-French 6 factors"
) -> ggplot:
    cols_extra = set(data.columns) - set(cols.keys())
    gdata = (
        data
        .group_by(datetime = pl.col("datetime").dt.date())
        .agg(
            [pl.col(col).sum() for col in cols]
            + [pl.col(col).first() for col in cols_extra - {"datetime"}]
        )
        .unpivot(list(cols.keys()), index = list(cols_extra))
        .with_columns(
            pl.col("variable").replace(cols).cast(pl.Enum(list(cols.values())))
        )
    )

    g = (
        ggplot(gdata, aes("datetime", "value")) +
        gg.geom_segment(
            gg.aes(xend = "datetime", y = 0, yend = "value", color = segment_color),
            size = 0.1, alpha = 0.5
        ) +
        gg.facet_wrap("variable") +
        gg.scale_x_datetime(date_labels = "%Y", date_breaks = "2 year") +
        gg.scale_color_manual(values = ["red", "green"]) +
        gg.labs(
            title = title,
            x = "Date", y = "Log return"
        ) +
        gg.theme_bw() +
        gg.theme(
            axis_text_x = gg.element_text(rotation = 45, hjust = 1),
            legend_position = "none"
        )
    )

    return g



# PCA --------------------------------------------------------------------------

def factor_loadings(
    data: pl.DataFrame,
    cols: list[str],
    block_size: float,
    exp: float = 1
) -> ggplot:
    gdata = (
        data
        .unpivot(cols, index = ["block", "fct"], variable_name = "pc")
        .with_columns(pl.col("value").abs())
        .with_columns(
            (pl.col("value") ** exp) / (pl.col("value") ** exp).sum().over(["block", "fct"])
        )
        .group_by(pl.col("block") // 20, "fct", "pc")
        .agg(pl.col("value").sum())
        .with_columns(
            pl.col("block") * 20,
            pl.col("pc").str.replace("evec_", "PC ")
        )
    )

    g = (
        ggplot(gdata, aes("block", "value", fill = "pc")) +
        gg.geom_area(position = gg.position_stack(reverse = True)) +
        gg.facet_wrap("fct") +
        gg.scale_fill_manual(values = palette_cmap("viridis", 6, (0.2, 1), True)) +
        gg.labs(
            title = "Factors' absolute loadings on PCs across time",
            x = f"Block ($\\approx$ {block_size} trading days)", y = "Absolute loading",
            fill = "",
            caption = "PC: principal component, ordered by variance explained"
        ) +
        gg.theme_bw() +
        gg.theme(legend_position = "bottom")
    )

    return g


def principal_components(
    data_evals: pl.DataFrame, data_pcs: pl.DataFrame,
    blocks_size: int,
    cols_evals: list[str], cols_pc: list[str]
) -> ggplot:
    evalues_repeated = (
        data_evals
        .with_columns(
            (pl.col(c) / pl.sum_horizontal(cols_evals)) for c in cols_evals
        )
        .unpivot(
            cols_evals, index = ["block"],
            variable_name = "evalue_name", value_name = "evalue"
        )
        ["evalue"]
        .repeat_by(blocks_size)
        .explode()
    )

    gdata_pcs = (
        data_pcs
        .unpivot(
            cols_pc, index = ["datetime", "block"],
            variable_name = "pc", value_name = "pc_value"
        )
        .with_columns(
            pl.col("pc").str.replace("pc_", "PC "),
            evalue = evalues_repeated # Join is too expensive
        )
        .group_by(pl.row_index() // 50, "pc")
        .agg(pl.col("pc_value").sum(), pl.col("evalue").mean(), pl.col("datetime").last())
    )

    g = (
        ggplot(gdata_pcs, aes("datetime", "pc_value")) +
        gg.geom_segment(
            aes(xend = "datetime", y = 0, yend = "pc_value", color = "evalue"),
            size = 0.1, alpha = 0.5
        ) +
        gg.facet_wrap("pc") +
        gg.scale_x_datetime( date_labels = "%Y", date_breaks = "2 year") +
        gg.labs(
            title = "Principal Components across time",
            x = "Date", y = "Value", color = "Var. explained",
            caption = "PC: principal component, ordered by variance explained"
        ) +
        gg.theme_bw() +
        gg.theme(
            axis_text_x=gg.element_text(rotation = 45, hjust = 1)
        )
    )

    return g
