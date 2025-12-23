def download_market_data(tick, min_date, max_date):
    import pandas as pd
    import yfinance as yf
    msft = yf.Ticker(tick)
    raw = msft.history(start=min_date, end=max_date)[['Open', 'High', 'Low', 'Close', 'Volume']]
    # fill in missing business days
    idx = pd.date_range(min_date, max_date, freq='B')
    # use last observation carried forward for missing value
    output_df = raw.reindex(idx, method='pad')
    # Pandas does not keep index (date) when converted into spark dataframe
    output_df['date'] = output_df.index
    output_df['ticker'] = tick
    output_df = output_df.rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Volume": "volume", "Close": "close"})
    return output_df

def generate_prices(start_price, mu, sigma, days):
    """
    Simulate a single price path over a fixed horizon using a simple
    multiplicative random walk with normally distributed shocks.

    In this context, a "random walk" means that each new price is based on
    the previous price plus a random change in return. The random component,
    referred to here as a "shock", represents an unpredictable return
    realization drawn from a normal distribution. Specifically, at each
    time step we draw a small return shock whose mean and standard deviation
    are scaled by the sampling rate (1 / days), and then update the price
    by multiplying the previous price by (1 + shock). This approximates a
    geometric Brownian motion-style process and ensures that relative
    changes, not absolute changes, drive the path. The price is floored at
    zero to avoid negative prices.

    Parameters
    ----------
    start_price : float
        Initial price at time step 0; this becomes the first element of the
        returned price path and the base level from which all shocks are
        applied.
    mu : float
        Annualized (or total-horizon) drift parameter for the process. On
        each step we use `mu * (1 / days)` as the mean of the normal shock,
        so larger `mu` tilts the simulated path upward over the full horizon.
    sigma : float
        Annualized (or total-horizon) volatility of returns. On each step we
        use `sigma * sqrt(1 / days)` as the standard deviation of the normal
        shock, so larger `sigma` produces more volatile, jagged paths.
    days : int
        Number of discrete time steps in the simulated price path; it
        controls both the length of the returned array and the time
        resolution over which `mu` and `sigma` are scaled.

    Returns
    -------
    numpy.ndarray
        Array of length `days` containing the simulated price path.
    """
    import numpy as np
    shock = np.zeros(days)
    price = np.zeros(days)
    sample_rate = 1 / float(days)
    price[0] = start_price
    for i in range(1, days):
        shock[i] = np.random.normal(loc=mu * sample_rate, scale=sigma * np.sqrt(sample_rate))
        price[i] = max(0, price[i - 1] + shock[i] * price[i - 1])
    return price


def create_seed_df(runs):
    import pandas as pd
    import numpy as np
    return pd.DataFrame(list(np.arange(0, runs)), columns=['trial_id'])


def get_shortfall(simulations, var):
    """
    Compute the expected shortfall (conditional VaR) at a given confidence level.

    Expected shortfall measures the average loss in the tail beyond the VaR
    cutoff. For example, at 99% this returns the mean of all simulated outcomes
    that are worse than or equal to the 99% VaR.

    Parameters
    ----------
    simulations : array-like of float
        Simulated portfolio outcomes (typically returns or end-of-horizon values).
    var : float or int
        VaR confidence level (e.g., 95 or 99), passed to `get_var` to determine
        the cutoff point in the left tail.

    Returns
    -------
    float
        The average of all simulated values less than or equal to the VaR
        threshold, representing the expected loss when losses exceed VaR.
    """
    import numpy as np
    var = get_var(simulations, var)
    return float(np.mean([s for s in simulations if s <= var]))


def get_var(simulations, var):
    """
    Compute the Value at Risk (VaR) at a given confidence level from simulations.

    VaR answers the question: "How much can I lose (or how low can this metric
    go) with a given confidence level over the horizon of the simulations?" For
    example, at 99% this function returns the 1st percentile of the simulated
    distribution.

    Parameters
    ----------
    simulations : array-like of float
        Simulated portfolio outcomes (typically returns or end-of-horizon values).
    var : float or int
        VaR confidence level (e.g., 95 or 99). Internally converted to the
        corresponding lower-tail percentile `100 - var`.

    Returns
    -------
    float
        The empirical VaR value (lower-tail percentile) computed from the
        simulations.
    """
    import numpy as np
    return float(np.percentile(simulations, 100 - var))


def non_linear_features(xs):
    import numpy as np
    fs = []
    for x in xs:
        fs.append(x)
        fs.append(np.sign(x) * x ** 2)
        fs.append(x ** 3)
        fs.append(np.sign(x) * np.sqrt(abs(x)))
    return fs


def predict_non_linears(ps, fs):
    s = ps[0]
    for i, f in enumerate(fs):
        s = s + ps[i + 1] * f
    return float(s)
