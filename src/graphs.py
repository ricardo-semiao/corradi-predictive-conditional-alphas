
# Setup ------------------------------------------------------------------------

import polars as pl
import numpy as np

import matplotlib as mpl
import plotnine as gg
from plotnine import ggplot, aes

from collections.abc import Collection

from src.parameters import PARAMETERS as PARS
from src.parameters import _counts, _trading_days
import src.utils as ut

ut.set_seed()


# Parameters:
FILL_COLORS = {
    "10% - 90%": "#ffb399",
    "20% - 80%": "#ff6699",
    "30% - 70%": "#ff0000",
    "40% - 60%": "#ffffff",
    "median": "black",
    "non-full factors": "blue",
    "non-full others": "gray"
}


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


# Partials ---------------------------------------------------------------------

def riverplot(
    data: pl.DataFrame, x: str,
    median_size: float = 0.25, date_breaks: str = "1 year"
) -> ggplot:
    g = (
        ggplot(data, aes(x, group = 1)) +
        gg.geom_ribbon(aes(ymin = "q1", ymax = "q9", fill = "'10% - 90%'")) +
        gg.geom_ribbon(aes(ymin = "q2", ymax = "q8", fill = "'20% - 80%'")) +
        gg.geom_ribbon(aes(ymin = "q3", ymax = "q7", fill = "'30% - 70%'")) +
        gg.geom_ribbon(aes(ymin = "q4", ymax = "q6", fill = "'40% - 60%'")) +
        gg.geom_line(aes(y = "median", color = "'median'"), size = median_size) +
        gg.scale_x_date(date_labels = "%Y", date_breaks = date_breaks) +
        gg.scale_fill_manual(values = FILL_COLORS) +
        gg.scale_color_manual(values = FILL_COLORS, name = "") +
        gg.theme_bw() +
        gg.theme(
            axis_text_x = gg.element_text(rotation = 45, hjust = 1),
            figure_size = (6.5, 5),
            legend_position = "bottom",
            legend_box = "vertical",
            legend_key = gg.element_rect(colour = "black", linewidth = 0.5)
        )
    )
    return g



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


def returns_river(
    data: pl.DataFrame | pl.LazyFrame,
    x: str, y: str,
    median_size: float = 0.01, labs: dict[str, str] = {}
) -> ggplot:
    data = (
        data
        .group_by(x)
        .agg(
            median = pl.median(y),
            **{
                f"q{q}": pl.quantile(y, q / 10)
                for q in [4, 6, 3, 7, 2, 8, 1, 9]
            }
        )
        .with_columns(pl.from_epoch("ts_day_ny", time_unit = "d"))
    )

    if isinstance(data, pl.LazyFrame):
        data = data.collect()

    guides = gg.guides(color = False) if median_size == 0 else None

    g = (
        riverplot(data, x, median_size) +
        gg.scale_y_continuous(labels = lambda bs: [f"{b:.0%}" for b in bs]) +
        gg.labs(fill = "Quantiles", **labs) +
        guides
    )

    return g


def day_fullness_river(data: pl.DataFrame, n: int | str = "") -> ggplot:
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
        riverplot(data, "d") +
        gg.geom_point(aes("d", "n", color = "'non-full factors'"), data_factor, size = 1.5) +
        gg.geom_point(aes("d", "median", color = "'non-full others'"), days_low_obs, size = 1.5) +
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
            subtitle = "Each column is a randomly selected PERMNO",
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



# Betas ------------------------------------------------------------------------

BETA_LABELS = {
    f"beta_{i}": f"$\\beta_{i}$" for i in range(1, 7)
}

def adjusted_returns(
    vis_permnos: Collection[int],
    path_adjusted: str = "data/stocks_1min_adjusted.parquet",
    path_returns: str = "data/stocks_1min_returns.parquet"
):
    data = (
        pl.scan_parquet(path_adjusted)
        .filter(pl.col("permno").is_in(vis_permnos))
        .join(
            pl.scan_parquet(path_returns)
            .filter(pl.col("permno").is_in(vis_permnos))
            .group_by("permno", ts_day_ny = pl.col("ts_min_ny") // 1440)
            .agg(pl.sum("log_return")),
            on = ["permno", "ts_day_ny"],
        )
        .with_columns(
            pl.from_epoch("ts_day_ny", time_unit = "d"),
            adjustment = pl.col("log_return") - pl.col("adjusted_return")
        )
        .with_columns(adjusted_up = pl.col("adjustment") > 0)
        .unpivot(index = ["permno", "ts_day_ny", "adjusted_up"])
        .with_columns(
            pl.col("variable").str.replace("_", " ")
            .cast(pl.Enum(("adjustment", "adjusted return", "log return"))),
            pl.col("adjusted_up").cast(pl.String).cast(pl.Enum(("true", "false")))
        )
        .collect()
    )

    g = (
        ggplot(data, aes(x = "ts_day_ny", xend = "ts_day_ny", y = 0, yend = "value")) +
        gg.geom_segment(aes(color = "adjusted_up"), alpha = 0.1) +
        gg.facet_grid("variable", "permno", scales = "free_x") +
        gg.scale_x_datetime(date_labels = "%Y", date_breaks = "2 year") +
        gg.scale_color_manual(values = ["green", "red"]) +
        gg.labs(
            title = "Returns and adjustment across time",
            subtitle = "Each column is a randomly selected PERMNOs",
            y = "Value", x = "Time",
            color = "Adjusted up"
        ) +
        gg.theme_bw() +
        gg.theme(
            axis_text_x = gg.element_text(rotation = 45, hjust = 1),
            legend_position = "bottom",
            legend_direction = "horizontal"
        ) +
        gg.guides(color = gg.guide_legend(override_aes = {"alpha": 1}))
    )

    return g


def adjusted_returns_river(
    path_1min: str = "data/stocks_1min_adjusted.parquet",
    path_5min: str = "data/stocks_5min_adjusted.parquet"
) -> ggplot:
    datas = (
        pl.scan_parquet(path)
        .group_by("ts_day_ny")
        .agg(
            median = pl.median("adjusted_return"),
            **{
                f"q{q}": pl.quantile("adjusted_return", q / 10)
                for q in [4, 6, 3, 7, 2, 8, 1, 9]
            }
        )
        .with_columns(
            pl.from_epoch("ts_day_ny", time_unit = "d"),
            frequency = pl.lit(f"{i} minute")
        )
        for i, path in [(1, path_1min), (5, path_5min)]
    )
    data = pl.concat(datas).collect()

    g = (
        riverplot(data, "ts_day_ny", 0) +
        gg.facet_wrap("frequency", nrow = 2) +
        gg.scale_y_continuous(labels = lambda bs: [f"{b:.0%}" for b in bs]) +
        gg.labs(
            title = "Adjusted returns' distribution across time",
            subtitle = "For each frequency aggregation",
            fill = "Quantiles", y = "Adjusted return", x = "Time"
        ) +
        gg.guides(color = False)
    )

    return g


def betas(
    vis_permnos: Collection[str],
    data_path: str = "data/betas_1min.parquet"
) -> ggplot:
    data = (
        pl.scan_parquet(data_path)
        .filter(pl.col("permno").is_in(vis_permnos))
        .unpivot(index = ["permno", "ts_day_ny"])
        .with_columns(
            pl.from_epoch("ts_day_ny", time_unit = "d"),
            pl.col("variable").replace(BETA_LABELS)
        )
        .collect()
    )

    g = (
        ggplot(data, aes("ts_day_ny", "value")) +
        gg.geom_line(alpha= 0.5) +
        gg.facet_wrap("variable", nrow = 2, scales = "free_x") +
        gg.scale_x_date(date_labels = "%Y", date_breaks = "2 years") +
        gg.labs(
            title = "Realized betas across time", subtitle = f"For PERMNO {vis_permnos}",
            x = "Time", y = "Value"
        ) +
        gg.theme_bw() +
        gg.theme(
            axis_text_x = gg.element_text(rotation = 45, hjust = 1)
        )
    )

    return g


def betas_river(
    data_path: str = "data/betas_1min.parquet",
    D: int = 6
) -> ggplot:
    datas = []

    for i in range(1, D + 1):
        datas.append(
            pl.scan_parquet(data_path)
            .group_by("ts_day_ny")
            .agg(
                median = pl.median(f"beta_{i}"),
                **{
                    f"q{q}": pl.quantile(f"beta_{i}", q / 10)
                    for q in [4, 6, 3, 7, 2, 8, 1, 9]
                }
            )
            .with_columns(
                pl.from_epoch("ts_day_ny", time_unit = "d"),
                beta = pl.lit(f"$\\beta_{i}$")
            )
            .collect()
        )

    g = (
        riverplot(pl.concat(datas), "ts_day_ny", 0, "2 years") +
        gg.facet_wrap("beta", nrow = 2, scales = "free_y") +
        gg.labs(
            title = "Realized betas' distribution across time",
            x = "Time", y = "Value", fill = "Quantiles"
        ) +
        gg.guides(color = False)
    )

    return g


def frequency_comparison(
    path_1min: str = "data/stocks_1min_adjusted.parquet",
    path_5min: str = "data/stocks_5min_adjusted.parquet",
    col: str = "adjusted_return"
) -> pl.DataFrame:
    data = (
        pl.scan_parquet(path_1min)
        .join(
            pl.scan_parquet(path_5min),
            on = ["permno", "ts_day_ny"], how = "full"
        )
        .with_columns(
            pl.col(col).alias(f"{col}_1min"),
            pl.col(f"{col}_right").alias(f"{col}_5min"),
            (pl.col(col) - pl.col(f"{col}_right")).alias(f"{col}_diff")
        )
        .select([f"{col}_1min", f"{col}_5min", f"{col}_diff"])
        .rename(lambda x: x.replace("_", " "))
        .describe()
        [2:]
    )

    return data
