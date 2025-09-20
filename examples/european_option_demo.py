"""
Demo of European option pricing using Monte Carlo simulation and comparison with Black-Scholes.
"""

import numpy as np
import matplotlib.pyplot as plt
from option_pricing.gbm import simulate_gbm, generate_antithetic_variates
from option_pricing.monte_carlo import european_option_mc, convergence_analysis
from option_pricing.black_scholes import black_scholes
from option_pricing.visualization import plot_price_paths, plot_payoff_distribution, plot_convergence

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
S0 = 100.0       # Initial stock price
K = 100.0        # Strike price
T = 1.0          # Time to maturity (1 year)
r = 0.05         # Risk-free rate (5%)
sigma = 0.2      # Volatility (20%)
option_type = 'call'  # 'call' or 'put'

# Monte Carlo parameters
n_steps = 252    # Number of time steps (daily for 1 year)
n_sims = 10_000  # Number of simulations

print(f"Pricing European {option_type} option with the following parameters:")
print(f"  S0 = {S0}, K = {K}, T = {T}, r = {r}, sigma = {sigma}")
print(f"  n_steps = {n_steps}, n_sims = {n_sims}")
print("-" * 50)

# 1. Simulate stock price paths using GBM
print("1. Simulating stock price paths...")
paths = simulate_gbm(S0, r, sigma, T, n_steps, n_sims)

# 2. Price the option using Monte Carlo
print("2. Pricing option using Monte Carlo...")
mc_price, mc_std_err, payoffs = european_option_mc(
    S0, K, r, sigma, T, option_type, n_steps, n_sims
)
print(f"   MC Price: ${mc_price:.4f} ± ${2 * mc_std_err:.4f} (95% CI)")

# 3. Calculate Black-Scholes price for comparison
print("3. Calculating Black-Scholes price...")
bs_price, greeks = black_scholes(S0, K, T, r, sigma, option_type)
print(f"   BS Price: ${bs_price:.4f}")
print(f"   Delta: {greeks['delta']:.4f}, Gamma: {greeks['gamma']:.6f}")
print(f"   Vega: {greeks['vega']:.4f}, Theta: {greeks['theta']:.6f}, Rho: {greeks['rho']:.4f}")

# 4. Analyze convergence
print("4. Analyzing convergence...")
n_sims_array, prices, std_errors = convergence_analysis(
    S0, K, r, sigma, T, option_type,
    min_sims=100, max_sims=10_000, step=100
)

# 5. Visualize results
print("5. Generating visualizations...")

# Plot sample price paths
plt.figure(figsize=(12, 6))
plot_price_paths(
    paths[:, :10],  # Plot first 10 paths
    title=f"Sample GBM Paths (First 10 of {n_sims} Simulations)",
    xlabel="Trading Days",
    ylabel="Stock Price"
)

# Plot payoff distribution
plt.figure(figsize=(12, 6))
plot_payoff_distribution(
    payoffs,
    option_type=option_type,
    title=f"Distribution of {option_type.capitalize()} Option Payoffs (n={n_sims})"
)

# Plot convergence
plt.figure(figsize=(12, 6))
plot_convergence(
    n_sims_array,
    prices,
    std_errors,
    true_value=bs_price,
    title=f"Monte Carlo Convergence for {option_type.capitalize()} Option"
)

plt.show()

print("\nDemo completed successfully!")
