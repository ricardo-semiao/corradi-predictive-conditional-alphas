import polars as pl
import numpy as np

from typing import NamedTuple, Final
from numpy.typing import NDArray

from concurrent.futures import ProcessPoolExecutor
from threadpoolctl import ThreadpoolController

from tqdm import tqdm


class AlphasParameters(NamedTuple):
    window_warmup: int # Initial burn-in window size
    scaler: float      # Bandwidth scaling factor
    trim_exp: float    # Trimming exponent zeta in (0, 1/4)
    k: int             # Number of principal components to retain

PARS_ALPHAS: Final[AlphasParameters] = AlphasParameters(
    window_warmup = 500,
    scaler = 3.5, # 2, 3.5 (also 1, 5)
    trim_exp = 0.2, # (also 0.01, 0.10)
    k = 3
)


#  ----------------------------------------------------------

data_state: pl.DataFrame
data_adjusted: pl.LazyFrame
state_idx: NDArray[np.uint32]
state_vals: NDArray[np.float64]

d: int
band_const: float
band_exp: float

def worker_init(
    data_state_path: str,
    data_adjusted_path: str
) -> None:
    # Permanently limit thread pools on the worker's process:
    _thread_controller = ThreadpoolController()
    _thread_controller.limit(limits = 1).__enter__() # Consider user_api = "blas"

    # Set global variables:
    global data_state, data_adjusted, state_idx, state_vals, d, band_const, band_exp

    data_state = pl.read_parquet(data_state_path)
    data_adjusted = pl.scan_parquet(data_adjusted_path)
    state_idx = data_state.get_column("ts_day_ny").to_numpy()
    state_vals = data_state.drop("ts_day_ny").to_numpy()

    d = PARS_ALPHAS.k
    band_const = (4.0 / (d + 2.0)) ** (1.0 / (d + 4.0))
    band_exp = (-1.0 / (d + 4.0))


#  ----------------------------------------------------------


INV_SQRT_2PI_K: float = (
    (1.0 / np.sqrt(2.0 * np.pi))
    ** PARS_ALPHAS.k
)

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
) -> float:
    K_1_Wm1 = kernel_gauss((PC_1_Wm1 - PC_t) / (h_W * PARS_ALPHAS.scaler))  # (T-1,)
    deno = np.mean(K_1_Wm1) / (h_W ** PARS_ALPHAS.k) # Across t

    if deno <= d_W: # np.isnan(deno)
        return 0.0

    num = np.mean(K_1_Wm1 * Z_2_W) # Across t
    return float(num / deno)
# We could use denominator cancellation to avoid some np.mean()


def cae_linear(
    PC_t: NDArray[np.float64], # (k,)
    Z_2_W: NDArray[np.float64],    # (T-1,)
    PC_1_Wm1: NDArray[np.float64]    # (T-1, k)
) -> np.float64:
    X = np.column_stack([np.ones(PC_1_Wm1.shape[0]), PC_1_Wm1])

    beta, _, _, _ = np.linalg.lstsq(X, Z_2_W)
    return beta[0] + PC_t @ beta[1:]



#  ----------------------------------------------------------

def pca_window(
    C_1_W: NDArray[np.float64], # (W, k) historical state variables
    k: int
) -> NDArray[np.float64]:
    # Standardization:
    mean = np.mean(C_1_W, axis = 0) # (k,) Across t
    std = np.std(C_1_W, axis = 0, ddof = 1) # (k,) Across t
    if np.any(std == 0.0):
        raise ValueError("Constant feature detected in PCA window.")

    C_1_W_norm = (C_1_W - mean) / std

    # Eigendecomposition:
    cov_mat = np.cov(C_1_W_norm, rowvar = False, ddof = 1) # (k, k)
    eigvals, eigvecs = np.linalg.eigh(cov_mat)

    # Principal components:
    idx_sort = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx_sort]
    scores = C_1_W_norm @ eigvecs[:, :k] # (T, k)

    return scores / 100.0


#  ----------------------------------------------------------

def get_permno_alphas(permno: int):
    data_permno = (
        data_adjusted
        .filter(pl.col("permno") == permno)
        .select(["ts_day_ny", "adjusted_return"])
        .with_columns(Z_tp1 = pl.col("adjusted_return").shift(-1))
        .collect()
        .head(-1) # Remove null created by shift
    )

    resid = data_permno.get_column("Z_tp1").to_numpy()
    ts_day_ny = data_permno.get_column("ts_day_ny").to_numpy()

    # Align stock dates with state dates
    common_dates, stock_ind, state_ind = np.intersect1d(
        ts_day_ny, state_idx, return_indices = True
    )

    y_aligned = resid[stock_ind]
    state_aligned = state_vals[state_ind]

    if len(y_aligned) < PARS_ALPHAS.window_warmup:
        return (None, None, None, None)

    n = len(y_aligned) - PARS_ALPHAS.window_warmup # Need -/+1 ?
    res_ts = np.full(n, 0, dtype = np.uint32)
    res_alpha = np.full(n, np.nan, dtype = np.float64)
    res_linear = np.full(n, np.nan, dtype = np.float64)

    # Window loop
    for w in range(PARS_ALPHAS.window_warmup, len(y_aligned)):
        wi = w - PARS_ALPHAS.window_warmup
        res_ts[wi] = common_dates[w]

        # Extract current expanding window for states and response
        state_window = state_aligned[:w+1]
        y_window = y_aligned[:w+1]

        # Deal with windows that have too few valid observations
        if len(y_window) <= PARS_ALPHAS.k: # Or other threshold?
            continue

        # 1. PCA Summarization
        #tic
        x = pca_window(state_window, PARS_ALPHAS.k)

        # Standardize PCA scores
        x = (x - np.mean(x, axis=0)) / np.std(x, axis=0, ddof=1)

        # Setup train/eval split (Regress tomorrow's return on today's state)
        x_hist = x[:-1]
        x_eval = x[-1]
        y_hist = y_window[1:]

        T = len(x_hist)

        # 2. Bandwidth and threshold computation
        bandrate = band_const * (T ** band_exp)
        smx = np.sqrt(np.mean(np.std(x, axis=0, ddof=1)**2))
        band_smx = bandrate * smx
        #toc and save

        h_W = PARS_ALPHAS.scaler * band_smx
        d_W = band_smx**PARS_ALPHAS.trim_exp

        # 3. Calculate alphas
        res_linear[wi] = cae_linear(x_eval, y_hist, x_hist)
        res_alpha[wi] = cae_gauss_trim(x_eval, y_hist, x_hist, h_W, d_W)

    return permno, res_ts, res_alpha, res_linear



#  ----------------------------------------------------------

def alphas_mp(
    permnos: list[int],
    initargs: tuple[str, str], # states, adjusted, and window paths
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
            if permno is None:
                continue
            results["permno"].append(np.full(len(t_arr), permno, dtype = np.uint32))
            results["ts_day_ny"].append(t_arr)
            results["alpha"].append(a_arr)
            results["linear_alpha"].append(l_arr)

        # Single bulk C-level concatenation and Polars wrapping at the end
        result_data = pl.DataFrame({k: np.concatenate(v) for k, v in results.items()})

    return result_data



if __name__ == "__main__":
    permnos = np.load("data/permnos_liquid.npy")
    res = alphas_mp(
        permnos,
        ("data/states_raw/states_clean.parquet", "data/stocks_1min_adjusted.parquet"),
        max_workers = 6, chunksize = 10
    )
    res.write_parquet("data/alphas_1min_2.parquet")
