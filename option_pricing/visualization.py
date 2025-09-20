"""
Visualization tools for option pricing analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union
import seaborn as sns

# Set the style for the plots
sns.set_style("whitegrid")
sns.set_palette("deep")


def plot_price_paths(
    paths: np.ndarray,
    n_paths: int = 10,
    title: str = "Simulated Stock Price Paths",
    xlabel: str = "Time Steps",
    ylabel: str = "Stock Price",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot sample price paths from a Monte Carlo simulation.
    
    Parameters
    ----------
    paths : np.ndarray
        Array of price paths with shape (n_steps + 1, n_sims)
    n_paths : int, optional
        Number of paths to plot, by default 10
    title : str, optional
        Plot title, by default "Simulated Stock Price Paths"
    xlabel : str, optional
        X-axis label, by default "Time Steps"
    ylabel : str, optional
        Y-axis label, by default "Stock Price"
    figsize : tuple, optional
        Figure size, by default (10, 6)
    save_path : str, optional
        If provided, save the plot to this path
    """
    plt.figure(figsize=figsize)
    
    # Plot a subset of paths
    n_sims = paths.shape[1]
    step = max(1, n_sims // n_paths)
    
    for i in range(0, min(n_paths * step, n_sims), step):
        plt.plot(paths[:, i], lw=1, alpha=0.6)
    
    # Plot mean path
    mean_path = np.mean(paths, axis=1)
    plt.plot(mean_path, 'k-', lw=2, label='Mean Path')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_payoff_distribution(
    payoffs: np.ndarray,
    option_type: str = 'call',
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the distribution of option payoffs.
    
    Parameters
    ----------
    payoffs : np.ndarray
        Array of discounted payoffs
    option_type : str, optional
        'call' or 'put', by default 'call'
    title : str, optional
        Plot title, if None a default title will be used
    figsize : tuple, optional
        Figure size, by default (10, 6)
    save_path : str, optional
        If provided, save the plot to this path
    """
    if title is None:
        title = f"Distribution of {option_type.capitalize()} Option Payoffs"
    
    plt.figure(figsize=figsize)
    
    # Plot histogram of payoffs
    sns.histplot(payoffs, kde=True, bins=50)
    
    # Add vertical line at mean
    mean_payoff = np.mean(payoffs)
    plt.axvline(mean_payoff, color='r', linestyle='--', 
                label=f'Mean Payoff: ${mean_payoff:.2f}')
    
    plt.title(title)
    plt.xlabel('Discounted Payoff')
    plt.ylabel('Frequency')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_convergence(
    n_sims: np.ndarray,
    prices: np.ndarray,
    std_errors: Optional[np.ndarray] = None,
    true_value: Optional[float] = None,
    title: str = "Monte Carlo Convergence",
    xlabel: str = "Number of Simulations",
    ylabel: str = "Option Price",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the convergence of Monte Carlo estimates.
    
    Parameters
    ----------
    n_sims : np.ndarray
        Array of number of simulations
    prices : np.ndarray
        Array of option prices corresponding to n_sims
    std_errors : np.ndarray, optional
        Array of standard errors, by default None
    true_value : float, optional
        True value (e.g., from Black-Scholes), by default None
    title : str, optional
        Plot title, by default "Monte Carlo Convergence"
    xlabel : str, optional
        X-axis label, by default "Number of Simulations"
    ylabel : str, optional
        Y-axis label, by default "Option Price"
    figsize : tuple, optional
        Figure size, by default (12, 6)
    save_path : str, optional
        If provided, save the plot to this path
    """
    plt.figure(figsize=figsize)
    
    # Plot price convergence
    plt.plot(n_sims, prices, 'b-', lw=2, alpha=0.7, label='MC Estimate')
    
    # Plot standard error bands if provided
    if std_errors is not None:
        plt.fill_between(
            n_sims, 
            prices - 2 * std_errors, 
            prices + 2 * std_errors,
            color='b', 
            alpha=0.1,
            label='95% Confidence Interval'
        )
    
    # Plot true value if provided
    if true_value is not None:
        plt.axhline(
            y=true_value, 
            color='r', 
            linestyle='--', 
            label=f'True Value: {true_value:.4f}'
        )
    
    plt.xscale('log')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_volatility_surface(
    strikes: np.ndarray,
    maturities: np.ndarray,
    implied_vols: np.ndarray,
    title: str = "Implied Volatility Surface",
    xlabel: str = "Strike Price",
    ylabel: str = "Time to Maturity (years)",
    zlabel: str = "Implied Volatility",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the implied volatility surface.
    
    Parameters
    ----------
    strikes : np.ndarray
        Array of strike prices
    maturities : np.ndarray
        Array of time to maturities
    implied_vols : np.ndarray
        2D array of implied volatilities with shape (len(maturities), len(strikes))
    title : str, optional
        Plot title, by default "Implied Volatility Surface"
    xlabel : str, optional
        X-axis label, by default "Strike Price"
    ylabel : str, optional
        Y-axis label, by default "Time to Maturity (years)"
    zlabel : str, optional
        Z-axis label, by default "Implied Volatility"
    figsize : tuple, optional
        Figure size, by default (12, 8)
    save_path : str, optional
        If provided, save the plot to this path
    """
    # Create a grid of strike prices and maturities
    K, T = np.meshgrid(strikes, maturities)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(
        K, T, implied_vols.T, 
        cmap='viridis', 
        edgecolor='none',
        alpha=0.8
    )
    
    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.title(title)
    
    # Rotate the view for better visualization
    ax.view_init(30, 45)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
