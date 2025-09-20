"""
Geometric Brownian Motion (GBM) for stock price simulation.
"""

import numpy as np
from typing import Union, Tuple


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float = 1.0,
    n_steps: int = 252,
    n_sims: int = 1000,
    random_seed: Union[int, None] = None,
) -> np.ndarray:
    """
    Simulate stock price paths using Geometric Brownian Motion.

    Parameters
    ----------
    S0 : float
        Initial stock price
    mu : float
        Expected return (annualized)
    sigma : float
        Volatility (annualized)
    T : float, optional
        Time to maturity in years, by default 1.0
    n_steps : int, optional
        Number of time steps, by default 252 (trading days)
    n_sims : int, optional
        Number of simulations, by default 1000
    random_seed : int, optional
        Random seed for reproducibility, by default None

    Returns
    -------
    np.ndarray
        Array of simulated price paths with shape (n_sims, n_steps + 1)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    dt = T / n_steps
    # Generate random numbers for the simulation
    rand_nums = np.random.standard_normal((n_steps, n_sims))
    
    # Calculate the price paths
    price_paths = np.zeros((n_steps + 1, n_sims))
    price_paths[0] = S0
    
    for t in range(1, n_steps + 1):
        price_paths[t] = price_paths[t-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + 
            sigma * np.sqrt(dt) * rand_nums[t-1]
        )
    
    return price_paths


def generate_antithetic_variates(
    S0: float,
    mu: float,
    sigma: float,
    T: float = 1.0,
    n_steps: int = 252,
    n_sims: int = 1000,
    random_seed: Union[int, None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate antithetic variates for variance reduction.
    
    Returns a tuple of (regular_paths, antithetic_paths)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    dt = T / n_steps
    # Generate random numbers for the first set of paths
    rand_nums = np.random.standard_normal((n_steps, n_sims // 2))
    
    # Create antithetic paths by using -rand_nums
    antithetic_rand = -rand_nums
    
    # Combine both sets of random numbers
    all_rand = np.concatenate([rand_nums, antithetic_rand], axis=1)
    
    # Generate price paths
    price_paths = np.zeros((n_steps + 1, n_sims))
    price_paths[0] = S0
    
    for t in range(1, n_steps + 1):
        price_paths[t] = price_paths[t-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + 
            sigma * np.sqrt(dt) * all_rand[t-1]
        )
    
    # Split into regular and antithetic paths
    regular_paths = price_paths[:, :n_sims//2]
    antithetic_paths = price_paths[:, n_sims//2:]
    
    return regular_paths, antithetic_paths
