from __future__ import annotations

import math

from scipy.stats import norm


_EPS = 1e-12


def black_scholes_call_price(spot: float, strike: float, maturity: float, r: float, sigma: float) -> float:
    if maturity <= 0.0:
        return max(spot - strike, 0.0)

    vol_term = sigma * math.sqrt(maturity)
    d1 = (math.log((spot + _EPS) / strike) + (r + 0.5 * sigma**2) * maturity) / (vol_term + _EPS)
    d2 = d1 - vol_term
    return spot * norm.cdf(d1) - strike * math.exp(-r * maturity) * norm.cdf(d2)


def black_scholes_call_delta(spot: float, strike: float, maturity: float, r: float, sigma: float) -> float:
    if maturity <= 0.0:
        return 1.0 if spot > strike else 0.0

    vol_term = sigma * math.sqrt(maturity)
    d1 = (math.log((spot + _EPS) / strike) + (r + 0.5 * sigma**2) * maturity) / (vol_term + _EPS)
    return norm.cdf(d1)
