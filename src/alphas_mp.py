
# Setup ------------------------------------------------------------------------

import polars as pl
import numpy as np

from typing import NamedTuple, Final
from numpy.typing import NDArray

from concurrent.futures import ProcessPoolExecutor
from threadpoolctl import ThreadpoolController

from tqdm import tqdm
from pickle import load as _load
from scipy.io import loadmat

class pkl:
    load = staticmethod(_load)



# Parameters and data ----------------------------------------------------------

class AlphasParameters(NamedTuple):
    window_warmup: int # Initial burn-in window size
    scaler: float      # Bandwidth scaling factor
    trim_exp: float    # Trimming exponent zeta in (0, 1/4)
    trim_scaler: float # Trimming density threshold scaling factor
    k: int             # Number of principal components to retain

PARS_ALPHAS: Final[AlphasParameters] = AlphasParameters(
    window_warmup = 500,
    scaler = 1,      # Try 1, 2, 3.5
    trim_exp = 0.2,  # Bounded by (0, 1/4)
    trim_scaler = 0, # 0.01, # Try 0.01 to 0.05
    k = 3
)

k = PARS_ALPHAS.k
band_const = (4 / (k + 2)) ** (1 / (k + 4))
band_exp = - 1 / (k + 4)
#trim_scaler = PARS_ALPHAS.trim_scaler * ((2.0 * np.pi) ** (- k / 2.0))

DATE_START, DATE_END = (
    pl.lit(x.astype("datetime64[D]").astype(np.uint32))
    for x in [np.datetime64("2001-01-01 00:00:00", "m"), np.datetime64("2017-12-29 23:59:59", "m")]
)



# Worker setup -----------------------------------------------------------------

# For every column, finds the date of the last non-NaN observation. For a
# window's W date, we want to use the column with such date being <= W. Also
# removes columns with duplicate last observation dates
def get_ads_data(ads: NDArray, ads_idx: NDArray):
    ads_last_obs = np.where(
        np.isnan(ads).any(axis = 0),
        np.argmax(np.isnan(ads), axis = 0) - 1,
        len(ads) - 1
    )

    _, idx_unique = np.unique(ads_last_obs, return_index = True)
    idx_unique = np.sort(idx_unique)

    ads = ads[:, idx_unique]
    ads_last_obs = ads_last_obs[idx_unique]

    #return {ads_idx[ads_last_obs[i]]: ads[:, i] for i in range(ads.shape[1])}
    return (ads_idx[ads_last_obs], ads)

data_factors: pl.DataFrame
data_ads: pl.DataFrame
data_permnos: pl.LazyFrame
f_idx: NDArray[np.uint32]
ads_idx: NDArray[np.uint32]
f: NDArray[np.float64]
ads: NDArray[np.float64]
ads_last_idx: NDArray[np.uint32]

def worker_init(freq: str) -> None:
    # Permanently limit thread pools on the worker's process:
    _thread_controller = ThreadpoolController()
    _thread_controller.limit(limits = 1).__enter__() # Consider user_api = "blas"

    # Set global variables:
    global data_factors, data_permnos, data_ads, f_idx, ads_idx, f, ads, ads_last_idx

    # Data loading:
    state_mat = loadmat("data/states_raw/FactorData_2018.mat")
    state_raws = []
    for key in ["factors", "ADSdata"]:
        mat = state_mat[key]
        data = (
            pl.DataFrame(mat[:, 1:])
            .with_columns(ts_day_ny = (mat[:, 0] - 719_529).astype(np.uint32))
            .select(pl.col("ts_day_ny"), pl.all().exclude("ts_day_ny"))
            .filter(
                (pl.col("ts_day_ny") >= DATE_START)
                & (pl.col("ts_day_ny") <= DATE_END)
            )
            .tail(-1) # To match with adjusted returns' dropped 1st observation
        )
        state_raws.append(data)

    data_factors = state_raws[0]
    f_idx = data_factors["ts_day_ny"].to_numpy()
    f = data_factors.drop("ts_day_ny").to_numpy()

    data_ads = state_raws[1]
    ads_idx = data_ads["ts_day_ny"].to_numpy()
    ads = data_ads.drop("ts_day_ny").to_numpy()

    # Align factors and ADS data:
    _, ads_inf, f_in_ads = np.intersect1d(
        ads_idx, f_idx, return_indices = True
    )

    f, f_idx = f[f_in_ads], f_idx[f_in_ads]
    ads, ads_idx = ads[ads_inf], ads_idx[ads_inf]

    ads_last_idx, ads = get_ads_data(ads, ads_idx)

    # Loading permnos dataset:
    data_permnos = pl.scan_parquet(f"data/stocks_{freq}_adjusted.parquet")



# Estimators -------------------------------------------------------------------

INV_SQRT_2PI_K: float = (
    (1.0 / np.sqrt(2.0 * np.pi))
    ** PARS_ALPHAS.k
)

def kernel_gauss(
    x: NDArray[np.float64]
) -> NDArray[np.float64]:
    return INV_SQRT_2PI_K * np.exp(-0.5 * np.sum(x**2, axis = 1)) # Across k

def cae_gauss_trim(
    PC_W: NDArray[np.float64],     # (k,) conditioning state variables at time t (PC_W)
    Z_2_W: NDArray[np.float64],    # (W-1,) target risk-adjusted returns (Z_{2:W})
    PC_1_Wm1: NDArray[np.float64], # (W-1, k) historical state variables (PC_{1:W-1})
    h_W: np.float64,               # Kernel bandwidth (h_W)
    d_W: np.float64                # Trimming density threshold (d_W)
) -> np.float64:
    K_1_Wm1 = kernel_gauss((PC_1_Wm1 - PC_W) / h_W)  # (T-1,)
    K_x = K_1_Wm1 / (h_W ** k)
    deno = np.mean(K_x) # Across t

    if np.isnan(deno):
        raise ValueError("Denominator is NaN in cae_gauss_trim.")
    if deno <= d_W:
        return np.float64(0.0)

    num = np.mean(K_x * Z_2_W) # Across t
    return num / deno
# NOTE: We could use denominator cancellation to avoid some np.mean()


def cae_linear(
    PC_W: NDArray[np.float64],    # (k,)
    Z_2_W: NDArray[np.float64],   # (T-1,)
    PC_1_Wm1: NDArray[np.float64] # (T-1, k)
) -> np.float64:
    X = np.column_stack([np.ones(PC_1_Wm1.shape[0]), PC_1_Wm1])

    beta, _, _, _ = np.linalg.lstsq(X, Z_2_W)
    return beta[0] + PC_W @ beta[1:]


def pca_window(
    C_1_W: NDArray[np.float64], # (W, k) historical state variables
    k: int = 3
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

    # Rescaling:
    return scores / 100.0


# Map function -----------------------------------------------------------------

def get_permno_alphas(
    permno: int
):
    data_permno = (
        data_permnos
        .filter(pl.col("permno") == permno)
        .select(["ts_day_ny", "adjusted_return"])
        .collect()
    )

    Z_idx = data_permno["ts_day_ny"].to_numpy()
    Z_1_T = data_permno["adjusted_return"].to_numpy()

    # Align stock and factor data (and thus also with ADS data):
    zc_idx, z_in_f, f_in_z = np.intersect1d(
        Z_idx, f_idx, return_indices = True
    )

    Z_1_T = Z_1_T[z_in_f]
    f_1_T = f[f_in_z]
    ads_1_T = ads[f_in_z]

    n = len(Z_1_T)
    if n <= PARS_ALPHAS.window_warmup:
        # Can happen as permnos are filtered before aligning with state data
        return (None, None, None, None)

    n_out = n - PARS_ALPHAS.window_warmup
    res_ts = np.full(n_out, 0, dtype = np.uint32)
    res_alpha = np.full(n_out, np.nan, dtype = np.float64)
    res_linear = np.full(n_out, np.nan, dtype = np.float64)

    for w in range(PARS_ALPHAS.window_warmup, n): # CHECK: add +1 to start?
        wi = w - PARS_ALPHAS.window_warmup
        res_ts[wi] = zc_idx[w - 1] # CHECK: w-1?

        # Window data:
        w_idx = zc_idx[:w]
        W = len(w_idx) - 1

        # if W < k: continue # Uneeded because of warmup window size

        f_1_W = f_1_T[:w, :]
        ads_current_col = np.searchsorted(ads_last_idx, zc_idx[w - 1])
        ads_1_W = ads_1_T[:w, ads_current_col:(ads_current_col + 1)]

        # PCA and its standardization:
        PC_1_W = pca_window(np.hstack([f_1_W, ads_1_W]), k)

        PC_std = np.std(PC_1_W, axis = 0)
        if np.any(PC_std == 0.0): # Surely never happens
            raise ValueError("Constant feature detected in PCA window.")
        PC_1_W = (PC_1_W - np.mean(PC_1_W, axis = 0)) / PC_std

        # Bandwidth and trimming:
        bandrate = band_const * (W ** band_exp)
        # smx = np.sqrt(np.mean(np.std(PC_1_W, axis = 0, ddof = 1)**2)) # Should be = 1 and uneeded
        smx = 1.0

        h_W = PARS_ALPHAS.scaler * (bandrate * smx)
        d_W = (bandrate * smx)**PARS_ALPHAS.trim_exp * INV_SQRT_2PI_K * PARS_ALPHAS.trim_scaler

        # Compute alphas:
        PC_W = PC_1_W[-1, :]
        PC_1_Wm1 = PC_1_W[:-1, :]
        Z_2_W = Z_1_T[1:w]

        res_linear[wi] = cae_linear(PC_W, Z_2_W, PC_1_Wm1)
        res_alpha[wi] = cae_gauss_trim(PC_W, Z_2_W, PC_1_Wm1, h_W, d_W)

    return (permno, res_ts, res_alpha, res_linear)



# Orchestrator -----------------------------------------------------------------

def alphas_mp(
    permnos: list[int],
    freq: str,
    max_workers: int | None = None,
    chunksize: int = 1,
    quiet: bool = False
) -> pl.DataFrame:
    with ProcessPoolExecutor(
        initializer = worker_init,
        initargs = (freq,),
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
    freq = "1min" # "5min" or "1min"

    result = alphas_mp(
        permnos, freq,
        max_workers = 6, chunksize = 10
    )
    result.write_parquet(f"data/alphas_{freq}.parquet")
