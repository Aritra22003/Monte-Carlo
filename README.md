# Monte Carlo Option Pricing

A Python framework for pricing European and exotic options using Monte Carlo simulation, with validation against the Black-Scholes model.

## Features

- **Geometric Brownian Motion (GBM)** simulation for stock price paths
- **European option pricing** (call/put) using Monte Carlo simulation
- **Black-Scholes model** for validation and comparison
- **Variance reduction techniques** including antithetic variates
- **Asian option pricing** with arithmetic or geometric averaging
- **Convergence analysis** to study the behavior of Monte Carlo estimates
- **Visualization tools** for price paths, payoff distributions, and convergence

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/monte-carlo-option-pricing.git
   cd monte-carlo-option-pricing
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Example

```python
import numpy as np
from option_pricing.monte_carlo import european_option_mc
from option_pricing.black_scholes import black_scholes

# Parameters
S0 = 100.0       # Initial stock price
K = 100.0        # Strike price
T = 1.0          # Time to maturity (1 year)
r = 0.05         # Risk-free rate (5%)
sigma = 0.2      # Volatility (20%)
option_type = 'call'  # 'call' or 'put'

# Monte Carlo simulation
mc_price, mc_std_err, _ = european_option_mc(S0, K, r, sigma, T, option_type)
print(f"Monte Carlo Price: ${mc_price:.4f} ± ${2 * mc_std_err:.4f} (95% CI)")

# Black-Scholes price for comparison
bs_price, _ = black_scholes(S0, K, T, r, sigma, option_type)
print(f"Black-Scholes Price: ${bs_price:.4f}")
```

### Running the Demo

Run the example script to see Monte Carlo simulation in action:

```bash
python examples/european_option_demo.py
```

This will:
1. Simulate stock price paths using GBM
2. Price a European option using Monte Carlo
3. Compare with the Black-Scholes price
4. Generate visualizations of price paths, payoff distribution, and convergence

## Project Structure

```
option_pricing/
├── __init__.py          # Package initialization
├── gbm.py               # Geometric Brownian Motion simulation
├── monte_carlo.py       # Monte Carlo option pricing
├── black_scholes.py     # Black-Scholes model
└── visualization.py     # Plotting and visualization tools
examples/
└── european_option_demo.py  # Demo script
```

## Documentation

### Geometric Brownian Motion (GBM)

The `gbm` module provides functions for simulating stock price paths:

- `simulate_gbm()`: Simulate stock price paths using GBM
- `generate_antithetic_variates()`: Generate antithetic variates for variance reduction

### Monte Carlo Pricing

The `monte_carlo` module implements option pricing using Monte Carlo simulation:

- `european_option_mc()`: Price European call/put options
- `asian_option_mc()`: Price Asian options with arithmetic or geometric averaging
- `convergence_analysis()`: Analyze convergence of Monte Carlo estimates

### Black-Scholes Model

The `black_scholes` module provides analytical solutions:

- `black_scholes()`: Calculate option price and Greeks
- `implied_volatility()`: Calculate implied volatility from market price

### Visualization

The `visualization` module provides plotting functions:

- `plot_price_paths()`: Plot simulated stock price paths
- `plot_payoff_distribution()`: Plot distribution of option payoffs
- `plot_convergence()`: Plot convergence of Monte Carlo estimates
- `plot_volatility_surface()`: Plot implied volatility surface

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- Hull, J. C. (2017). Options, Futures and Other Derivatives (10th ed.). Pearson.
- Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering. Springer.
- Wilmott, P. (2006). Paul Wilmott on Quantitative Finance (2nd ed.). Wiley.
