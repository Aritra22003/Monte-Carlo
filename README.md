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


## Project Structure

```bash
Monte-Carlo/
│── option_pricing/
│   ├── gbm.py               # Simulate stock price paths
│   ├── monte_carlo.py       # Monte Carlo pricers
│   ├── black_scholes.py     # Closed-form BS model
│   ├── visualization.py     # Plotting utilities
│
│── examples/
│   ├── european_option_demo.py
│   ├── asian_option_demo.py
│
│── notebooks/
│   ├── MonteCarlo_European_Call.ipynb   # Interactive demo
│
│── requirements.txt
│── README.md
│── LICENSE

```
## Installation
Clone the repo and install dependencies

```bash
git clone https://github.com/Aritra22003/Monte-Carlo.git
cd Monte-Carlo

# create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

#install dependencies
pip install -r requirements.txt
```

## Usage Example

Run the European option demo:
```bash
python examples/european_option_demo.py

```

Expected output(sample run with 100,000 simulations):

```yaml
Monte Carlo European Call Price: 10.45 ± 0.05
Black–Scholes Analytical Price: 10.49

```
## Results and Visualizations

- Simulated GBM paths
  ![GBM Paths](https://github.com/Aritra22003/Monte-Carlo/blob/439ef94c3048efef3ffb9ec4c81f86c15195c5e4/gbm.png)
- Distribution of Call Option Payoffs
  ![Call Option Distribution](https://github.com/Aritra22003/Monte-Carlo/blob/8c64749f6cd9e8ebde6e9b7ece338e799cc5ff99/call%20option%20payoff.png)
- Monte Carlo Convergence for Call Option Pricing
  ![MC Convergence](https://github.com/Aritra22003/Monte-Carlo/blob/53d9c4a55ac0ac46b97e4db0c4228dc4706dcbb8/call%20option.png)

## 🔬 Methodology

1. **Model stock prices with GBM**

$$
S_T = S_0 \exp\Big((r - \tfrac{1}{2}\sigma^2)T + \sigma \sqrt{T} Z\Big), \quad Z \sim N(0,1)
$$

---

2. **Monte Carlo pricing**

$$
C_{MC} = e^{-rT} \cdot \frac{1}{M} \sum_{i=1}^M \max\!\big(S_T^{(i)} - K, 0\big)
$$

---

3. **Black–Scholes benchmark**

Closed-form formula for a European call:

$$
C_{BS} = S_0 N(d_1) - K e^{-rT} N(d_2)
$$

where

$$
d_1 = \frac{\ln \tfrac{S_0}{K} + (r + \tfrac{1}{2}\sigma^2)T}{\sigma \sqrt{T}}, 
\quad 
d_2 = d_1 - \sigma \sqrt{T}
$$

---

4. **Variance reduction techniques**

- **Antithetic variates:** use pairs \( Z \) and \( -Z \) to reduce variance.  
- **Control variates:** compare Monte Carlo results with Black–Scholes analytical solution.

## Sample Plots(plots generated in MonteCarlo_European_call.ipynb)

- GBM sample paths
- Convergence of Monte Carlo price vs simulations
- Efficient variance reduction comparison

## What I Learned

- How Monte Carlo methods approximate option pricing by simulating asset paths.
- Importance of variance reduction to improve efficiency.
- When Monte Carlo is essential (path-dependent options like Asian options).
- How to validate results against analytical Black–Scholes solutions.
- Practical skills: Python modularization, NumPy vectorization, and Matplotlib visualization.

## Next Steps

- mplement Greeks (Δ, Γ, Θ, ρ, Vega) estimation via Monte Carlo.
- Explore stochastic volatility models (e.g., Heston).
- Parallelize simulations using Numba or joblib for performance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- Hull, J. C. (2017). Options, Futures and Other Derivatives (10th ed.). Pearson.
- Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering. Springer.
- Wilmott, P. (2006). Paul Wilmott on Quantitative Finance (2nd ed.). Wiley.
