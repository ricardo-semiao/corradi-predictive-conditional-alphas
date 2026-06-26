import numpy as np

def get_k(Delta_n, d, T):
    theta = 0.5
    # Formula: theta * Delta_n^(-1/2) * sqrt(log(d))
    k_target = int(np.round(theta * (Delta_n**-0.5) * np.sqrt(np.log(d))))
    
    # Find the closest divisor of T to ensure clean block splits
    divisors = [i for i in range(1, T + 1) if T % i == 0]
    return min(divisors, key=lambda x: abs(x - k_target))

def pca_hf(
    X: np.ndarray,
    k: int | None = None, u: float | None = None,
    quiet: bool = True
):
    """
    Extracts high-frequency principal components from a matrix of variables.
    
    Parameters:
    X (np.ndarray): T x d matrix (Rows = moments, Cols = variables). 
        Typically log-prices or cumulative levels.
    k (int): Block size (number of moments per block).
    u (float): Optional threshold for jump truncation.
    
    Returns:
    np.ndarray: T x d matrix of Principal Components.
    """
    # Setup --------------------------------------------------------------------
    T, d = X.shape
    Delta_n = 1.0 / T

    # Get default k
    if k is None:
        k = get_k(Delta_n, d, T)
        if not quiet: print(f"Auto-calculated block size k: {k}")
    
    if u is None:
        # BPV_i = (pi/2) * sum(|r_t| * |r_{t-1}|)
        bpv = (np.pi / 2) * np.sum(np.abs(X[1:]) * np.abs(X[:-1]), axis=0)
        
        # u_{i} = 3 * sqrt(BPV_i) * Delta_n^0.47
        u = 3 * np.sqrt(bpv) * (Delta_n ** 0.47)
        if not quiet: print(f"Auto-calculated jump truncation threshold u: {u}")
    

    # Main ---------------------------------------------------------------------
    # Initialize output matrix for PC returns
    PC_returns = np.zeros_like(X)
    num_blocks = T // k
    prev_eigenvectors = None
    
    for i in range(num_blocks + 1):
        # Define block indices
        start_idx = i * k
        end_idx = min((i + 1) * k, T)
        
        if start_idx >= end_idx:
            break
            
        block_returns = X[start_idx:end_idx]
        
        # Spot Covariance Estimation (with jump truncation)
        valid_returns = np.where(np.abs(block_returns) <= u, block_returns, 0)
        # Obs: could make jump truncation optional

            
        # Calculate covariance (handling edge cases where truncation removes all data)
        if len(valid_returns) > 0:
            # c_i = (1/k) * R^T * R
            C_i = (valid_returns.T @ valid_returns) / k
            
            # Eigen-decomposition (np.linalg.eigh is highly optimized for symmetric matrices)
            eigenvalues, eigenvectors = np.linalg.eigh(C_i)
            
            # eigh returns eigenvalues in ascending order; we need descending for PCA
            sort_indices = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, sort_indices]
        else:
            # Fallback to identity matrix if the block has no valid data
            eigenvectors = np.eye(d)
        
        # Construct Principal Component Returns
        # Project the current block's returns using the PREVIOUS block's eigenvectors
        # (For the very first block, we use its own eigenvectors to start the process)
        if prev_eigenvectors is None:
            projection_matrix = eigenvectors
        else:
            projection_matrix = prev_eigenvectors
            
        PC_returns[start_idx:end_idx] = block_returns @ projection_matrix
        
        # Store current eigenvectors to be used by the next block
        prev_eigenvectors = eigenvectors
        
    # Integration Step
    # Cumulatively sum the PC returns across time to get the PC levels
    PC_levels = np.cumsum(PC_returns, axis=0)
    
    return PC_levels


# Loop version -----------------------------------------------------------------

def pca_hf_step(X_j, k, u=None, prev_eigenvectors=None, initial_levels=None):
    """
    Processes a single day's return matrix X_j in a multi-day sequential PCA pipeline.
    
    Parameters:
    X_j (np.ndarray): T x d matrix of returns for day j.
    k (int): Fixed block size (number of moments per block).
    u (np.ndarray or float, optional): Jump threshold. If None, calculated via daily BPV.
    prev_eigenvectors (np.ndarray, optional): d x d matrix of eigenvectors from the end of day j-1.
    initial_levels (np.ndarray, optional): 1 x d or (d,) array of PC levels at the end of day j-1.
    
    Returns:
    PC_levels_j (np.ndarray): T x d matrix of cumulative Principal Components for day j.
    current_eigenvectors (np.ndarray): d x d matrix of eigenvectors from the final block of day j.
    """
    T, d = X_j.shape
    
    # 1. Daily BPV Truncation (if u is not explicitly passed)
    if u is None:
        Delta_n = 1.0 / T  # Frequency relative to 1 day
        bpv = (np.pi / 2) * np.sum(np.abs(X_j[1:]) * np.abs(X_j[:-1]), axis=0)
        u = 3 * np.sqrt(bpv) * (Delta_n ** 0.47)
        
    PC_returns = np.zeros_like(X_j)
    num_blocks = T // k
    
    # Keep track of the active eigenvectors across blocks
    active_vectors = prev_eigenvectors
    
    for i in range(num_blocks + 1):
        start_idx = i * k
        end_idx = min((i + 1) * k, T)
        
        if start_idx >= end_idx:
            break
            
        block_returns = X_j[start_idx:end_idx]
        
        # Apply element-wise jump truncation
        valid_returns = np.where(np.abs(block_returns) <= u, block_returns, 0)
        
        # Calculate spot covariance
        if len(valid_returns) > 0:
            C_i = (valid_returns.T @ valid_returns) / k
            eigenvalues, eigenvectors = np.linalg.eigh(C_i)
            sort_indices = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, sort_indices]
        else:
            eigenvectors = np.eye(d) if active_vectors is None else active_vectors
            
        # Project returns: Use prev_eigenvectors if it's the very first block of the loop,
        # otherwise use the active vectors from the block prior.
        if active_vectors is None:
            projection_matrix = eigenvectors
        else:
            projection_matrix = active_vectors
            
        PC_returns[start_idx:end_idx] = block_returns @ projection_matrix
        active_vectors = eigenvectors
        
    # 2. Integration with Inter-day Continuity
    PC_levels_j = np.cumsum(PC_returns, axis=0)
    
    if initial_levels is not None:
        # Carry over the ending levels of yesterday as the baseline for today
        PC_levels_j += initial_levels
        
    # Return today's PC levels, and the final eigenvectors to pass to day j+1
    return PC_levels_j, active_vectors
