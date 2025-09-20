"""
Black-Scholes model for European option pricing.
"""

import numpy as np
from scipy.stats import norm
from typing import Union, Optional, Tuple


def d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    Calculate d1 for the Black-Scholes formula.
    
    Parameters
    ----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    q : float, optional
        Dividend yield (annualized), by default 0.0
        
    Returns
    -------
    float
        d1 parameter
    """
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    Calculate d2 for the Black-Scholes formula.
    """
    return d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)


def black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = 'call',
    q: float = 0.0,
) -> Tuple[float, dict]:
    """
    Calculate the Black-Scholes price of a European option.
    
    Parameters
    ----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    option_type : str, optional
        'call' or 'put', by default 'call'
    q : float, optional
        Dividend yield (annualized), by default 0.0
        
    Returns
    -------
    Tuple[float, dict]
        (option_price, greeks)
        where greeks is a dictionary containing delta, gamma, vega, theta, rho
    """
    d1_val = d1(S, K, T, r, sigma, q)
    d2_val = d2(S, K, T, r, sigma, q)
    
    if option_type.lower() == 'call':
        price = S * np.exp(-q * T) * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2_val)
        delta = np.exp(-q * T) * norm.cdf(d1_val)
        theta = (-S * sigma * np.exp(-q * T) * norm.pdf(d1_val)) / (2 * np.sqrt(T)) - \
                r * K * np.exp(-r * T) * norm.cdf(d2_val) + \
                q * S * np.exp(-q * T) * norm.cdf(d1_val)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2_val)
    elif option_type.lower() == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2_val) - S * np.exp(-q * T) * norm.cdf(-d1_val)
        delta = -np.exp(-q * T) * norm.cdf(-d1_val)
        theta = (-S * sigma * np.exp(-q * T) * norm.pdf(d1_val)) / (2 * np.sqrt(T)) + \
                r * K * np.exp(-r * T) * norm.cdf(-d2_val) - \
                q * S * np.exp(-q * T) * norm.cdf(-d1_val)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2_val)
    else:
        raise ValueError("option_type must be either 'call' or 'put'")
    
    # Calculate Greeks
    gamma = (np.exp(-q * T) * norm.pdf(d1_val)) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * norm.pdf(d1_val) * np.sqrt(T)
    
    # Adjust theta to be per day (from per year)
    theta = theta / 365.0
    
    greeks = {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,  # per day
        'rho': rho
    }
    
    return price, greeks


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = 'call',
    q: float = 0.0,
    sigma_guess: float = 0.2,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Calculate the implied volatility using the Newton-Raphson method.
    
    Parameters
    ----------
    market_price : float
        Observed market price of the option
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free rate (annualized)
    option_type : str, optional
        'call' or 'put', by default 'call'
    q : float, optional
        Dividend yield (annualized), by default 0.0
    sigma_guess : float, optional
        Initial guess for volatility, by default 0.2
    tol : float, optional
        Tolerance for convergence, by default 1e-6
    max_iter : int, optional
        Maximum number of iterations, by default 100
        
    Returns
    -------
    float
        Implied volatility
    """
    sigma = sigma_guess
    
    for _ in range(max_iter):
        price, greeks = black_scholes(S, K, T, r, sigma, option_type, q)
        vega = greeks['vega']
        
        # Avoid division by zero
        if vega < 1e-20:
            break
            
        # Newton-Raphson update
        diff = market_price - price
        sigma += diff / vega
        
        # Check for convergence
        if abs(diff) < tol:
            break
    
    return sigma
