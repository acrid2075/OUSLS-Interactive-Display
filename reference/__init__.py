"""NumPy reference implementation for the signed-K SLS simulator."""

from .simulation import expected_gain, generate_price_paths, k_star, moments, simulate_sls

__all__ = [
    "expected_gain",
    "generate_price_paths",
    "k_star",
    "moments",
    "simulate_sls",
]
