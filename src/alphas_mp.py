
# Setup ------------------------------------------------------------------------

import polars as pl
import numpy as np

from typing import NamedTuple, Final
from numpy.typing import NDArray

from concurrent.futures import ProcessPoolExecutor
from threadpoolctl import ThreadpoolController

from tqdm import tqdm
from pickle import load as _load

class pkl:
    load = staticmethod(_load)



# Parameters and data ----------------------------------------------------------

class AlphasParameters(NamedTuple):
    window_warmup: int # Initial burn-in window size
    scaler: float      # Bandwidth scaling factor
    trim_exp: float    # Trimming exponent zeta in (0, 1/4)
    k: int             # Number of principal components to retain


PARS_ALPHAS: Final[AlphasParameters] = AlphasParameters(
    window_warmup = 500,
    scaler = 8,
    trim_exp = 0.2,
    k = 3
)

class WindowArgs(NamedTuple):
    t: int
    t_1_Wm1: NDArray[np.uint32]      # (W,) t's of the window
    PC_1_Wm1: NDArray[np.float64]    # (W-1, k) historical PC predictors
    PC_t: NDArray[np.float64]        # (k,) current PC predictor at day t
    h_W: np.float64
    d_W: np.float64



# Worker setup -----------------------------------------------------------------

data_state: pl.DataFrame
data_adjusted: pl.LazyFrame
windows_args: list[WindowArgs]

def worker_init(
    data_state_path: str,
    data_adjusted_path: str,
    windows_args_path: str
) -> None:
    # Permanently limit thread pools on the worker's process:
    _thread_controller = ThreadpoolController()
    _thread_controller.limit(limits = 1).__enter__() # Consider user_api = "blas"

    # Set global variables:
    global data_state, data_adjusted, windows_args

    data_state = pl.read_parquet(data_state_path)
    data_adjusted = pl.scan_parquet(data_adjusted_path)
    with open(windows_args_path, "rb") as f:
        windows_args = pkl.load(f)



# Estimators -------------------------------------------------------------------

INV_SQRT_2PI_K: float = (
    (1.0 / np.sqrt(2.0 * np.pi))
    ** PARS_ALPHAS.k
)

TRIM_SCALER = 0.01 * (2.0 * np.pi) ** (- PARS_ALPHAS.k / 2.0)
# Try 0.01 to 0.05

def kernel_gauss(
    x: NDArray[np.float64]
) -> NDArray[np.float64]:
    return INV_SQRT_2PI_K * np.exp(-0.5 * np.sum(x**2, axis = 1)) # Across k

def cae_gauss_trim(
    PC_t: NDArray[np.float64],     # (k,) conditioning state variables at time t (PC_t)
    Z_2_W: NDArray[np.float64],    # (W-1,) target risk-adjusted returns (Z_{2:W})
    PC_1_Wm1: NDArray[np.float64], # (W-1, k) historical state variables (PC_{1:W-1})
    h_W: np.float64,               # Kernel bandwidth (h_W)
    d_W: np.float64                # Trimming density threshold (d_W)
) -> float: # TODO: np.float64?
    h_W = h_W * PARS_ALPHAS.scaler

    K_1_Wm1 = kernel_gauss((PC_1_Wm1 - PC_t) / h_W)  # (T-1,)
    deno = np.mean(K_1_Wm1) / (h_W ** PARS_ALPHAS.k) # Across t

    if np.isnan(deno):
        raise ValueError("Denominator is NaN in cae_gauss_trim.")
    if deno <= d_W * TRIM_SCALER:
        return 0.0

    #num = np.mean(K_1_Wm1 * Z_2_W) # Across t
    return float(np.average(Z_2_W, weights = K_1_Wm1)) #or num / deno
# We could use denominator cancellation to avoid some np.mean()


def cae_linear(
    PC_t: NDArray[np.float64], # (k,)
    Z_2_W: NDArray[np.float64],    # (T-1,)
    PC_1_Wm1: NDArray[np.float64]    # (T-1, k)
) -> np.float64:
    X = np.column_stack([np.ones(PC_1_Wm1.shape[0]), PC_1_Wm1])

    beta, _, _, _ = np.linalg.lstsq(X, Z_2_W)
    return beta[0] + PC_t @ beta[1:]



# Map function -----------------------------------------------------------------

def get_permno_alphas(
    permno: int
):
    data_permno = (
        data_adjusted
        .filter(pl.col("permno") == permno)
        .select(["ts_day_ny", "adjusted_return"])
        .with_columns(Z_tp1 = pl.col("adjusted_return").shift(-1))
        .collect()
        .head(-1) # Remove null created by
    )

    t_permno = data_permno["ts_day_ny"].to_numpy()
    Z_2_Tp1 = data_permno["Z_tp1"].to_numpy()
    n = len(windows_args)

    res_ts = np.full(n, 0, dtype = np.uint32)
    res_alpha = np.full(n, np.nan, dtype = np.float64)
    res_linear = np.full(n, np.nan, dtype = np.float64)

    for i, w_args in enumerate(windows_args):
        t, t_1_Wm1, PC_1_Wm1, PC_t, h_W, d_W = w_args

        mask_t_pc_in_permno = np.isin(t_1_Wm1, t_permno, assume_unique = True)

        if not mask_t_pc_in_permno.sum() > PARS_ALPHAS.k:
            res_ts[i] = t
            continue

        idx_t_permno_in_pc = np.searchsorted(t_permno, t_1_Wm1[mask_t_pc_in_permno])
        Z_2_W = Z_2_Tp1[idx_t_permno_in_pc]
        PC_1_Wm1 = PC_1_Wm1[mask_t_pc_in_permno]

        # Nonparametric Kernel Alpha Estimate
        alpha_kernel = cae_gauss_trim(
            PC_t = PC_t, Z_2_W = Z_2_W, PC_1_Wm1 = PC_1_Wm1,
            h_W = h_W, d_W = d_W
        )

        # Linear Alpha Benchmark Estimate
        alpha_linear = cae_linear(
            PC_t = PC_t, Z_2_W = Z_2_W, PC_1_Wm1 = PC_1_Wm1
        )

        res_ts[i] = t
        res_alpha[i] = alpha_kernel
        res_linear[i] = alpha_linear

    return (permno, res_ts, res_alpha, res_linear)



# Orchestrator -----------------------------------------------------------------

def alphas_mp(
    permnos: list[int],
    initargs: tuple[str, str, str], # states, adjusted, and window paths
    max_workers: int | None = None,
    chunksize: int = 1,
    quiet: bool = False
) -> pl.DataFrame:
    with ProcessPoolExecutor(
        initializer = worker_init,
        initargs = initargs,
        max_workers = max_workers
    ) as executor:

        map_iterator = executor.map(
            get_permno_alphas,
            permnos,
            chunksize = chunksize
        )

        results = {"permno": [], "ts_day_ny": [], "alpha": [], "linear_alpha": []}

        for permno, t_arr, a_arr, l_arr in tqdm(map_iterator, disable = quiet, total = len(permnos)):
            results["permno"].append(np.full(len(t_arr), permno, dtype = np.uint32))
            results["ts_day_ny"].append(t_arr)
            results["alpha"].append(a_arr)
            results["linear_alpha"].append(l_arr)

        # Single bulk C-level concatenation and Polars wrapping at the end
        result_data = pl.DataFrame({k: np.concatenate(v) for k, v in results.items()})

    return result_data


if __name__ == "__main__":
    permnos = (
        pl.read_csv("data/stocks_raw/counts_permno.csv")
        ['permno'].unique().to_list()
    )
    result = alphas_mp(
        permnos[:120],
        (
            "data/states_raw/states_clean.parquet",
            "data/stocks_1min_adjusted.parquet",
            "data/states_raw/windows_args.pkl"
        ),
        max_workers = 6, chunksize = 2
    )
    result.write_parquet("data/alphas_1min.parquet")
    print(result)
