# Databricks notebook source
# MAGIC %md
# MAGIC <img src=https://d1r5llqwmkrl74.cloudfront.net/notebooks/fs-lakehouse-logo.png width="600px">
# MAGIC 
# MAGIC [![DBR](https://img.shields.io/badge/DBR-10.4ML-red?logo=databricks&style=for-the-badge)](https://docs.databricks.com/release-notes/runtime/10.4ml.html)
# MAGIC [![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://databricks.com/try-databricks)
# MAGIC [![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)
# MAGIC 
# MAGIC *Traditional banks relying on on-premises infrastructure can no longer effectively manage risk. Banks must abandon the computational inefficiencies of legacy technologies and build an agile Modern Risk Management practice capable of rapidly responding to market and economic volatility. Using value-at-risk use case, you will learn how Databricks is helping FSIs modernize their risk management practices, leverage Delta Lake, Apache Spark and MLFlow to adopt a more agile approach to risk management.* 
# MAGIC 
# MAGIC ___
# MAGIC <antoine.amend@databricks.com>

# COMMAND ----------

# MAGIC %md
# MAGIC <img src='https://raw.githubusercontent.com/databricks-industry-solutions/value-at-risk/master/images/reference_architecture.png' width=800>

# COMMAND ----------

# MAGIC %md
# MAGIC ## VAR 101
# MAGIC 
# MAGIC VaR is measure of potential loss at a specific confidence interval. A VAR statistic has three components: a time period, a confidence level and a loss amount (or loss percentage). What is the most I can - with a 95% or 99% level of confidence - expect to lose in dollars over the next month? There are 3 ways to compute Value at risk
# MAGIC # 
# MAGIC 
# MAGIC + **Historical Method**: The historical method simply re-organizes actual historical returns, putting them in order from worst to best.
# MAGIC + **The Variance-Covariance Method**: This method assumes that stock returns are normally distributed and use pdf instead of actual returns.
# MAGIC + **Monte Carlo Simulation**: This method involves developing a model for future stock price returns and running multiple hypothetical trials.
# MAGIC 
# MAGIC ### More intuition on each method
# MAGIC 
# MAGIC **Historical Method**
# MAGIC 
# MAGIC - Uses actual past returns as if they were possible future outcomes.
# MAGIC - Collect a history of returns (e.g., 1‑day returns over the last 250 trading days), sort them from worst (most negative) to best, and read off the chosen percentile:
# MAGIC   - 95% VaR → 5th percentile (bottom 5% of returns)
# MAGIC   - 99% VaR → 1st percentile
# MAGIC - That percentile return, multiplied by the current portfolio value, gives you the VaR.
# MAGIC - Pros: simple, uses real data, naturally captures fat tails and skew.  
# MAGIC - Cons: assumes the future looks like the historical sample, and is sensitive to the chosen lookback window.
# MAGIC 
# MAGIC **Variance–Covariance (Parametric) Method**
# MAGIC 
# MAGIC - Assumes returns follow a parametric distribution, most commonly a normal distribution.
# MAGIC - Estimate daily mean return (μ) and standard deviation (σ), then use a z‑score for your confidence level (≈1.65 for 95%, ≈2.33 for 99%).
# MAGIC - 1‑day VaR at confidence level α is approximated as:
# MAGIC   - VaRₐ ≈ (zₐ · σ − μ) × portfolio value
# MAGIC - For portfolios, you compute a portfolio standard deviation from the covariance matrix of asset returns.
# MAGIC - Pros: fast, analytic, and scales well to large portfolios.  
# MAGIC - Cons: relies heavily on distributional assumptions (e.g., normality) and can underestimate extreme losses or nonlinear payoffs.
# MAGIC 
# MAGIC **Monte Carlo Simulation**
# MAGIC 
# MAGIC - Builds a model for future returns (e.g., geometric Brownian motion, GARCH, factor models, jump‑diffusion) and simulates many scenarios.
# MAGIC - For each scenario, you re‑value the portfolio to get a distribution of simulated P&L.
# MAGIC - VaR is then read from the appropriate percentile of this simulated P&L distribution, just like in the historical method.
# MAGIC - Pros: very flexible; can handle complex/nonlinear instruments and richer dynamics (fat tails, stochastic volatility, jumps).  
# MAGIC - Cons: computationally more expensive and results depend on the quality of the chosen model and calibration.
# MAGIC 
# MAGIC We report in below example a simple Value at risk calculation for a synthetic instrument, given a volatility (i.e. standard deviation of instrument returns) and a time horizon (300 days). **What is the most I could lose in 300 days with a 95% confidence?**

# COMMAND ----------

# time horizon
days = 300

# volatility
sigma = 0.04 

# drift (average growth rate)
mu = 0.05  

# initial starting price
start_price = 10

# COMMAND ----------

# Visualize many simulated price paths for the synthetic instrument so the reader
# can see the range of potential future price trajectories that underlie the VaR
# calculation introduced in the previous cell.

import matplotlib.pyplot as plt
from utils.var_utils import generate_prices

plt.figure(figsize=(16,6))
for i in range(1, 500):
    plt.plot(generate_prices(start_price, mu, sigma, days))

plt.title('Simulated price')
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()

# COMMAND ----------

# Run many simulated price paths to obtain a distribution of end-of-horizon prices
# and visualize the 99% Value at Risk based on that simulated terminal price distribution.

from utils.var_viz import plot_var
simulations = [generate_prices(start_price, mu, sigma, days)[-1] for i in range(10000)]
plot_var(simulations, 99)

# COMMAND ----------

# MAGIC %md
# MAGIC Expected shortfall is measure that produces better incentives for traders than VAR. This is also sometimes referred to as conditional VAR, or tail loss. Where VAR asks the question 'how bad can things get?', expected shortfall asks 'if things do get bad, what is our expected loss?'. 

# COMMAND ----------

from utils.var_utils import get_var
from utils.var_utils import get_shortfall

print('Var99: {}'.format(round(get_var(simulations, 99), 2)))
print('Shortfall: {}'.format(round(get_shortfall(simulations, 99), 2)))

# COMMAND ----------

var99 = get_var(simulations, 99)
shortfall99 = get_shortfall(simulations, 99)

loss_var99 = start_price - var99
loss_shortfall99 = start_price - shortfall99

print(
    f"With an initial price of {start_price}, the 99% VaR is {var99:.2f}, "
    f"meaning there is only about a 1% chance the final price falls below this level "
    f"and the loss exceeds {loss_var99:.2f}."
)
print('-'*100)
print(
    f"At the same 99% level, the expected shortfall is {shortfall99:.2f}, "
    f"which is the average final price in the worst 1% of scenarios "
    f"(an average loss of {loss_shortfall99:.2f} when things go that badly)."
)

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | Yfinance                               | Yahoo finance           | Apache2    | https://github.com/ranaroussi/yfinance              |
# MAGIC | tempo                                  | Timeseries library      | Databricks | https://github.com/databrickslabs/tempo             |
# MAGIC | PyYAML                                 | Reading Yaml files      | MIT        | https://github.com/yaml/pyyaml                      |
