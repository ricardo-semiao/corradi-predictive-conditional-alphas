
# Setup ------------------------------------------------------------------------

import polars as pl
import numpy as np
from einops import reduce, rearrange

from typing import TypedDict
from numpy.typing import NDArray

from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager
from concurrent.futures import ProcessPoolExecutor
from threadpoolctl import ThreadpoolController

from tqdm import tqdm



# Helpers ----------------------------------------------------------------------

class SharedArrayMeta(TypedDict):
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype # Maybe better to pass str for pickling

def create_shared_array(
    arr: NDArray, smm: SharedMemoryManager
) -> tuple[SharedArrayMeta, shared_memory.SharedMemory]:
    arr_c = np.ascontiguousarray(arr)
    shm = smm.SharedMemory(size = arr_c.nbytes)
    shared_arr = np.ndarray(arr_c.shape, dtype = arr_c.dtype, buffer = shm.buf)
    shared_arr[:] = arr_c[:]

    meta: SharedArrayMeta = {"name": shm.name, "shape": arr_c.shape, "dtype": arr_c.dtype}
    return meta, shm

WORKER_OBJS: dict[str, NDArray]
WORKER_SHMS: dict[str, shared_memory.SharedMemory]

def worker_init(shared_metas: dict[str, SharedArrayMeta]) -> None:
    # Permanently limit thread pools on the worker's process:
    _thread_controller = ThreadpoolController()
    _thread_controller.limit(limits = 1).__enter__() # Consider user_api = "blas"

    # Attach shared memory objects:
    global WORKER_OBJS, WORKER_SHMS
    WORKER_OBJS = {}
    WORKER_SHMS = {}

    for name, meta in shared_metas.items():
        shm = shared_memory.SharedMemory(name = meta["name"])
        WORKER_SHMS[name] = shm
        WORKER_OBJS[name] = np.ndarray(
            meta["shape"], dtype = meta["dtype"], buffer = shm.buf
        )



# Inner iteration: frequency results -------------------------------------------

def freq_results(
    F: NDArray, P: NDArray, # (T_s * M, D), (T_s * M,)
    M: int,
    F_day_s: NDArray, P_day_s: NDArray, # (T_s - 1, D), (T_s - 1,)
    T_s: np.int64
) -> tuple[NDArray, NDArray]: # (T_s, D), (T_s - 1,)
    # Reshape into tensors:
    FF = rearrange(F, '(t m) d -> t m d', t = T_s, m = M) # (T_s, M, D)
    PP = rearrange(P, '(t m) -> t m 1', t = T_s, m = M)   # (T_s, M, 1)

    # Compute cov. and var. across each day and factor:
    cov = reduce(FF * PP, 't m d -> t d', 'sum') # (T_s, D)
    var = reduce(FF * FF, 't m d -> t d', 'sum') # (T_s, D)

    # Compute betas, with var = 0 -> beta = 0:
    betas = np.divide(
        cov, var,
        out = np.zeros_like(cov), where = var != 0
    ) # (T_S, D)
    betas_lag = betas[:-1, :] # (T_s - 1, D)

    # Get the adjusted daily returns:
    P_day_hat = reduce(F_day_s * betas_lag, 't d -> t', 'sum') # (T_s - 1,)
    P_day_adj = P_day_s - P_day_hat                            # (T_s - 1,)

    return betas, P_day_adj



# Outer iteration: stock results -----------------------------------------------

class ResultsStock(TypedDict):
    permno: int
    day_range: tuple[np.int64, np.int64]
    m1: tuple[NDArray, NDArray] # (T_s, D), (T_s - 1,)
    m5: tuple[NDArray, NDArray] # (T_s, D), (T_s - 1,)

def stock_results_mp(permno: int, P_filepath: str) -> ResultsStock:
    # Fetch global shared matrices directly from dictionary
    F_m1, F_m5, F_m1_idx = (WORKER_OBJS[x] for x in ["F_m1", "F_m5", "F_m1_idx"])

    # Query permno from P_df (if query empty, should err):
    P_query = (
        pl.scan_parquet(P_filepath)
        .filter(pl.col("permno") == permno)
        .select(["ts_min_ny", "log_return"])
        .collect()
    )

    P_m1 = P_query["log_return"].to_numpy()    # (T_s * M1,)
    P_m1_idx = P_query["ts_min_ny"].to_numpy() # (T_s * M1,)

    # Crop data to stock's window:
    idx_start: np.int64 = np.searchsorted(F_m1_idx, P_m1_idx[0])
    idx_end: np.int64 = np.searchsorted(F_m1_idx, P_m1_idx[-1], side = "right")
    day_start = idx_start // 390 # First minute of first day (left-closed)
    day_end = np.int64(np.ceil(idx_end / 390)) # First minute of last day + 1 (right-open)
    T_s = day_end - day_start

    F_m1_s = F_m1[day_start * 390 : day_end * 390, :] # (T_s * M1, D)
    F_m5_s = F_m5[day_start * 78 : day_end * 78, :]   # (T_s * M5, D)

    # Crop P_m1 to (day_start, day_end), set missing to 0:
    F_m1_s_idx = F_m1_idx[day_start * 390 : day_end * 390]

    fs_in_p = np.clip(np.searchsorted(F_m1_s_idx, P_m1_idx), 0, len(F_m1_s_idx) - 1)
    fs_in_p_exact = F_m1_s_idx[fs_in_p] == P_m1_idx
    # To solve out-of-bounds indexing (TODO: happens for permno 22859)

    P_m1_s = np.zeros(T_s * 390, dtype = np.float64) # (T_s * M1,)
    P_m1_s[fs_in_p[fs_in_p_exact]] = P_m1[fs_in_p_exact]

    # Aggregate to different frequencies, get t+1 day returns:
    P_m5_s  = reduce(P_m1_s, '(t m) -> t', 'sum', m = 5)             # (T_s * M5,)
    P_day_s = reduce(P_m5_s, '(t m) -> t', 'sum', m = 78)[1:]        # (T_s - 1,)
    F_day_s = reduce(F_m5_s, '(t m) d -> t d', 'sum', m = 78)[1:, :] # (T_s - 1, D)

    # Frequency loop:
    results_permno: ResultsStock = {
        "permno": permno,
        "day_range": (day_start, day_end),
        "m1": freq_results(F_m1_s, P_m1_s, 390, F_day_s, P_day_s, T_s),
        "m5": freq_results(F_m5_s, P_m5_s, 78, F_day_s, P_day_s, T_s)
    }

    return results_permno
# Note: it would be better to save 390, 78 as constants



# Orchestrator -----------------------------------------------------------------

class ResultsRealized(TypedDict):
    permno: list[int]
    day_range: list[tuple[int, int]]
    m1: list[tuple[NDArray, NDArray]] # (T_s, D), (T_s - 1,)
    m5: list[tuple[NDArray, NDArray]] # (T_s, D), (T_s - 1,)

def betas_returns_realized_mp(
    permnos_ordered: NDArray, P_filepath: str, # (I,)
    F_m1: NDArray, F_m5: NDArray, # (T * M1, D), (T * M5, D)
    F_m1_idx: NDArray, # (T * M1,)
    quiet: bool = False,
    max_workers: int | None = None, chunksize: int = 1
) -> ResultsRealized:
    I = len(permnos_ordered)
    results: ResultsRealized = {"permno": [], "day_range": [], "m1": [], "m5": []}

    with SharedMemoryManager() as smm:
        metas: dict[str, SharedArrayMeta] = {}
        shms: dict[str, shared_memory.SharedMemory] = {}
        for key, arr in [("F_m1", F_m1), ("F_m5", F_m5), ("F_m1_idx", F_m1_idx)]:
            meta, shm = create_shared_array(arr, smm)
            metas[key] = meta
            shms[key] = shm

        with ProcessPoolExecutor(
            initializer = worker_init, initargs = (metas,),
            max_workers = max_workers
        ) as executor:

            map_iterator = executor.map(
                stock_results_mp,
                permnos_ordered, [P_filepath] * I,
                chunksize = chunksize
            )

            for results_stock in tqdm(map_iterator, disable = quiet, total = I):
                results["permno"].append(results_stock["permno"])
                results["day_range"].append(results_stock["day_range"])
                results["m1"].append(results_stock["m1"])
                results["m5"].append(results_stock["m5"])

    return results
# Note: this could be defined elsewhere to not pass to workers, but it is a low cost



# Debug ------------------------------------------------------------------------

if __name__ == "__main__":
    data_pc_1 = pl.read_csv("data/pca_1min.csv")
    data_pc_5 = pl.read_csv("data/pca_5min.csv")

    F_m1, F_m5 = (
        x.select([f"pc_{i}" for i in range(1, 7)]).to_numpy()
        for x in [data_pc_1, data_pc_5]
    )
    F_m1_idx, F_m5_idx = (
        x["datetime"]
        .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M")
        .to_numpy().astype("datetime64[m]").astype(np.uint32)
        for x in [data_pc_1, data_pc_5]
    )

    permnos_ordered = pl.read_csv("data/stocks_raw/counts_permno.csv")["permno"].to_numpy()

    betas_returns_realized_mp(
        permnos_ordered = permnos_ordered,
        P_filepath = "data/stocks_1min_returns.parquet",
        F_m1 = F_m1, F_m5 = F_m5,
        F_m1_idx = F_m1_idx,
        max_workers = 7,
        chunksize = 20
    )
