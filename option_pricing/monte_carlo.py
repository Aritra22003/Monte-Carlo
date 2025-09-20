"""
Monte Carlo simulation for option pricing.
"""

import numpy as np
from typing import Callable, Optional, Tuple, Union


def european_option_mc(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = 'call',
    n_steps: int = 252,
    n_sims: int = 10000,
    use_antithetic: bool = False,
    random_seed: Optional[int] = None,
) -> Tuple[float, float, np.ndarray]:
    """
    Price a European option using Monte Carlo simulation.

    Parameters
    ----------
    S0 : float
        Initial stock price
    K : float
        Strike price
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    T : float
        Time to maturity in years
    option_type : str, optional
        'call' or 'put', by default 'call'
    n_steps : int, optional
        Number of time steps, by default 252 (trading days)
    n_sims : int, optional
        Number of simulations, by default 10000
    use_antithetic : bool, optional
        Whether to use antithetic variates for variance reduction, by default False
    random_seed : int, optional
        Random seed for reproducibility, by default None

    Returns
    -------
    Tuple[float, float, np.ndarray]
        (option_price, standard_error, all_payoffs)
    """
    from .gbm import simulate_gbm, generate_antithetic_variates
    
    # Generate price paths
    if use_antithetic:
        # Adjust number of simulations to be even
        if n_sims % 2 != 0:
            n_sims += 1
        
        # Get both regular and antithetic paths
        paths, _ = generate_antithetic_variates(
            S0, r, sigma, T, n_steps, n_sims, random_seed
        )
    else:
        paths = simulate_gbm(S0, r, sigma, T, n_steps, n_sims, random_seed)
    
    # Get terminal prices
    terminal_prices = paths[-1]
    
    # Calculate payoffs
    if option_type.lower() == 'call':
        payoffs = np.maximum(terminal_prices - K, 0)
    elif option_type.lower() == 'put':
        payoffs = np.maximum(K - terminal_prices, 0)
    else:
        raise ValueError("option_type must be either 'call' or 'put'")
    
    # Discount payoffs to present value
    discounted_payoffs = np.exp(-r * T) * payoffs
    
    # Calculate option price and standard error
    option_price = np.mean(discounted_payoffs)
    standard_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_sims)
    
    return option_price, standard_error, discounted_payoffs


def asian_option_mc(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = 'call',
    avg_type: str = 'arithmetic',
    n_steps: int = 252,
    n_sims: int = 10000,
    random_seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Price an Asian option using Monte Carlo simulation.
    
    Parameters
    ----------
    avg_type : str
        'arithmetic' or 'geometric' averaging
    """
    from .gbm import simulate_gbm
    
    # Generate price paths
    paths = simulate_gbm(S0, r, sigma, T, n_steps, n_sims, random_seed)
    
    # Calculate average price along each path
    if avg_type.lower() == 'arithmetic':
        avg_prices = np.mean(paths, axis=0)
    elif avg_type.lower() == 'geometric':
        avg_prices = np.exp(np.mean(np.log(paths), axis=0))
    else:
        raise ValueError("avg_type must be 'arithmetic' or 'geometric'")
    
    # Calculate payoffs
    if option_type.lower() == 'call':
        payoffs = np.maximum(avg_prices - K, 0)
    elif option_type.lower() == 'put':
        payoffs = np.maximum(K - avg_prices, 0)
    else:
        raise ValueError("option_type must be either 'call' or 'put'")
    
    # Discount payoffs to present value
    discounted_payoffs = np.exp(-r * T) * payoffs
    
    # Calculate option price and standard error
    option_price = np.mean(discounted_payoffs)
    standard_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_sims)
    
    return option_price, standard_error


def convergence_analysis(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = 'call',
    min_sims: int = 100,
    max_sims: int = 10000,
    step: int = 100,
    use_antithetic: bool = False,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyze convergence of Monte Carlo estimates with increasing number of simulations.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (n_sims_array, prices, std_errors)
    """
    n_sims_array = np.arange(min_sims, max_sims + step, step, dtype=int)
    prices = np.zeros_like(n_sims_array, dtype=float)
    std_errors = np.zeros_like(n_sims_array, dtype=float)
    
    for i, n in enumerate(n_sims_array):
        price, std_err, _ = european_option_mc(
            S0, K, r, sigma, T, option_type, 
            n_sims=n, use_antithetic=use_antithetic,
            random_seed=random_seed
        )
        prices[i] = price
        std_errors[i] = std_err
    
    return n_sims_array, prices, std_errors
