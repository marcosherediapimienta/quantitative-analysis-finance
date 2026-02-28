import numpy as np
import scipy.stats as stats


class BlackScholesPricer:
    @staticmethod
    def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @classmethod
    def _d2(cls, S: float, K: float, T: float, r: float, sigma: float) -> float:
        return cls._d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    @classmethod
    def call_price(cls, S: float, K: float, T: float, r: float, sigma: float) -> float:
        d1 = cls._d1(S, K, T, r, sigma)
        d2 = cls._d2(S, K, T, r, sigma)
        return float(S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2))

    @classmethod
    def put_price(cls, S: float, K: float, T: float, r: float, sigma: float) -> float:
        d1 = cls._d1(S, K, T, r, sigma)
        d2 = cls._d2(S, K, T, r, sigma)
        return float(K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1))

    @classmethod
    def vega(cls, S: float, K: float, T: float, r: float, sigma: float) -> float:
        d1 = cls._d1(S, K, T, r, sigma)
        return float(S * stats.norm.pdf(d1) * np.sqrt(T))
