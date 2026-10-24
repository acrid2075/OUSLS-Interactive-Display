"""Reference calculations for trending OU prices and signed-K SLS.

The price engine consumes fixed standard-normal innovations so results can be
reproduced and compared with another implementation. Price returns supplied to
the controller are the forward-Euler Ito increments, ``Delta X + sigma^2 dt/2``.
They are not sampled simple returns ``exp(Delta X) - 1``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
_SMALL_ARGUMENT = 1.0e-4
_SMALL_GAIN_ARGUMENT = 1.0e-3


def _scalar(value: Any, name: str) -> float:
    """Convert a real scalar parameter and reject arrays, booleans, and NaNs."""
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _parameters(
    *, nu0: Any, nu1: Any, kappa: Any, sigma: Any
) -> tuple[float, float, float, float]:
    values = (
        _scalar(nu0, "nu0"),
        _scalar(nu1, "nu1"),
        _scalar(kappa, "kappa"),
        _scalar(sigma, "sigma"),
    )
    if values[3] < 0.0:
        raise ValueError("sigma must be nonnegative")
    return values


def _array(values: ArrayLike, name: str, *, ndim: int) -> FloatArray:
    try:
        raw = np.asarray(values)
        if np.iscomplexobj(raw):
            raise ValueError(f"{name} must contain real values")
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(name):
            raise
        raise ValueError(f"{name} must be a numeric array") from exc
    if result.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-dimensional array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _positive_time_grid(times: ArrayLike) -> FloatArray:
    grid = _array(times, "times", ndim=1)
    if np.any(grid < 0.0):
        raise ValueError("times must be nonnegative")
    return grid


def _check_finite(name: str, values: ArrayLike) -> None:
    if not np.all(np.isfinite(values)):
        raise FloatingPointError(
            f"nonfinite {name} produced; reduce parameter magnitudes or use a more stable configuration"
        )


def _ou_integrals(times: FloatArray, kappa: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return A, B, C where m=nu0*A+nu1*B and v=sigma^2*C.

    A = integral exp(-k(t-s)) ds, B = integral s exp(-k(t-s)) ds,
    and C = integral exp(-2k(t-s)) ds. Series avoid cancellation when k*t is
    close to zero, including the exact kappa=0 limits.
    """
    q = kappa * times
    small = np.abs(q) < _SMALL_ARGUMENT
    A = np.empty_like(times)
    B = np.empty_like(times)
    C = np.empty_like(times)

    if np.any(small):
        qs = q[small]
        ts = times[small]
        A[small] = ts * (
            1.0
            - qs / 2.0
            + qs**2 / 6.0
            - qs**3 / 24.0
            + qs**4 / 120.0
            - qs**5 / 720.0
            + qs**6 / 5040.0
        )
        B[small] = ts**2 * (
            0.5
            - qs / 6.0
            + qs**2 / 24.0
            - qs**3 / 120.0
            + qs**4 / 720.0
            - qs**5 / 5040.0
            + qs**6 / 40320.0
        )
        C[small] = ts * (
            1.0
            - qs
            + (2.0 / 3.0) * qs**2
            - qs**3 / 3.0
            + (2.0 / 15.0) * qs**4
            - (2.0 / 45.0) * qs**5
            + (4.0 / 315.0) * qs**6
        )

    large = ~small
    if np.any(large):
        ql = q[large]
        tl = times[large]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            al = -np.expm1(-ql) / kappa
            A[large] = al
            B[large] = (tl - al) / kappa
            C[large] = -np.expm1(-2.0 * ql) / (2.0 * kappa)
    _check_finite("OU moment integrals", (A, B, C))
    return A, B, C


def _mean_variance(
    times: FloatArray, nu0: float, nu1: float, kappa: float, sigma: float
) -> tuple[FloatArray, FloatArray]:
    A, B, C = _ou_integrals(times, kappa)
    with np.errstate(over="ignore", invalid="ignore"):
        mean = nu0 * A + nu1 * B
        variance = sigma**2 * C
    _check_finite("OU mean", mean)
    _check_finite("OU variance", variance)
    return mean, variance


def _zeta(times: FloatArray, kappa: float, sigma: float) -> FloatArray:
    """Compute zeta = (v - sigma^2 t)/2 without near-zero cancellation."""
    q = kappa * times
    small = np.abs(q) < _SMALL_ARGUMENT
    result = np.empty_like(times)
    if np.any(small):
        qs = q[small]
        ts = times[small]
        difference = ts * (
            -qs
            + (2.0 / 3.0) * qs**2
            - qs**3 / 3.0
            + (2.0 / 15.0) * qs**4
            - (2.0 / 45.0) * qs**5
            + (4.0 / 315.0) * qs**6
        )
        result[small] = 0.5 * sigma**2 * difference
    large = ~small
    if np.any(large):
        _, _, C = _ou_integrals(times[large], kappa)
        with np.errstate(over="ignore", invalid="ignore"):
            result[large] = 0.5 * sigma**2 * (C - times[large])
    _check_finite("zeta", result)
    return result


def _log_cosh(values: FloatArray) -> FloatArray:
    absolute = np.abs(values)
    return absolute + np.log1p(np.exp(-2.0 * absolute)) - np.log(2.0)


def generate_price_paths(
    normals: ArrayLike,
    *,
    nu0: float,
    nu1: float,
    kappa: float,
    sigma: float,
    T: float = 1.0,
) -> Mapping[str, FloatArray]:
    """Generate log prices, prices, and Ito return increments using Euler steps.

    ``normals`` has shape [paths, steps]. Its entries are the actual fixed
    standard-normal innovations, not a seed. Production uses shape [100, 50],
    yielding 100 paths with 51 stored points; smaller shapes support unit tests.
    """
    z = _array(normals, "normals", ndim=2)
    if not z.shape[0] or not z.shape[1]:
        raise ValueError("normals must have at least one path and one step")
    nu0, nu1, kappa, sigma = _parameters(
        nu0=nu0, nu1=nu1, kappa=kappa, sigma=sigma
    )
    T = _scalar(T, "T")
    if T <= 0.0:
        raise ValueError("T must be positive")

    path_count, step_count = z.shape
    dt = T / step_count
    if kappa > 0.0 and kappa * dt >= 2.0:
        raise ValueError(
            "Euler log-price recursion is not mean-square stable for this kappa and step size"
        )
    times = np.linspace(0.0, T, step_count + 1, dtype=np.float64)
    log_prices = np.zeros((path_count, step_count + 1), dtype=np.float64)
    return_increments = np.empty((path_count, step_count), dtype=np.float64)
    noise_scale = sigma * np.sqrt(dt)

    for n in range(step_count):
        old_x = log_prices[:, n]
        drift = nu0 + nu1 * times[n] - kappa * old_x
        with np.errstate(over="ignore", invalid="ignore"):
            dx = drift * dt + noise_scale * z[:, n]
            next_x = old_x + dx
            dr = dx + 0.5 * sigma**2 * dt
        _check_finite(f"log-price step {n + 1}", next_x)
        _check_finite(f"return increment at step {n}", dr)
        log_prices[:, n + 1] = next_x
        return_increments[:, n] = dr

    with np.errstate(over="ignore", invalid="ignore"):
        prices = np.exp(log_prices)
    _check_finite("price paths", prices)
    return {
        "times": times,
        "log_prices": log_prices,
        "prices": prices,
        "return_increments": return_increments,
    }


def simulate_sls(
    return_increments: ArrayLike, *, I0: float, K: float
) -> Mapping[str, FloatArray]:
    """Apply signed-K long and short controller recurrences to return increments."""
    drs = _array(return_increments, "return_increments", ndim=2)
    if not drs.shape[0] or not drs.shape[1]:
        raise ValueError("return_increments must have at least one path and one step")
    I0 = _scalar(I0, "I0")
    K = _scalar(K, "K")
    if I0 <= 0.0:
        raise ValueError("I0 must be positive")

    path_count, step_count = drs.shape
    long_gain = np.zeros((path_count, step_count + 1), dtype=np.float64)
    short_gain = np.zeros_like(long_gain)
    for n in range(step_count):
        old_long = long_gain[:, n]
        old_short = short_gain[:, n]
        dr = drs[:, n]
        with np.errstate(over="ignore", invalid="ignore"):
            next_long = old_long + (I0 + K * old_long) * dr
            next_short = old_short + (-I0 - K * old_short) * dr
        _check_finite(f"long gain at step {n + 1}", next_long)
        _check_finite(f"short gain at step {n + 1}", next_short)
        long_gain[:, n + 1] = next_long
        short_gain[:, n + 1] = next_short
    with np.errstate(over="ignore", invalid="ignore"):
        total_gain = long_gain + short_gain
    _check_finite("total gain", total_gain)
    return {
        "long_gain": long_gain,
        "short_gain": short_gain,
        "total_gain": total_gain,
    }


def moments(
    times: ArrayLike, *, nu0: float, nu1: float, kappa: float, sigma: float
) -> Mapping[str, FloatArray]:
    """Return analytical mean and variance of the trending OU log price."""
    t = _positive_time_grid(times)
    nu0, nu1, kappa, sigma = _parameters(
        nu0=nu0, nu1=nu1, kappa=kappa, sigma=sigma
    )
    mean, variance = _mean_variance(t, nu0, nu1, kappa, sigma)
    return {"mean": mean, "variance": variance}


def expected_gain(
    times: ArrayLike,
    *,
    nu0: float,
    nu1: float,
    kappa: float,
    sigma: float,
    I0: float,
    K: float,
) -> FloatArray:
    """Continuous-time expected total dollar gain from the paper's formula.

    Small K uses the analytic odd-power expansion. Otherwise a log-cosh
    formulation and expm1 avoid avoidable overflow and cancellation. Any result
    that is still nonfinite is raised as a diagnostic instead of being clipped.
    """
    t = _positive_time_grid(times)
    nu0, nu1, kappa, sigma = _parameters(
        nu0=nu0, nu1=nu1, kappa=kappa, sigma=sigma
    )
    I0 = _scalar(I0, "I0")
    K = _scalar(K, "K")
    if I0 <= 0.0:
        raise ValueError("I0 must be positive")
    if K == 0.0:
        return np.zeros_like(t)

    mean, _ = _mean_variance(t, nu0, nu1, kappa, sigma)
    with np.errstate(over="ignore", invalid="ignore"):
        phi = mean + 0.5 * sigma**2 * t
        zeta = _zeta(t, kappa, sigma)
        scaled_phi = K * phi
        scaled_zeta = K * (K * zeta)
    _check_finite("scaled expected-gain argument", (scaled_phi, scaled_zeta))

    small = np.maximum(np.abs(scaled_phi), np.sqrt(np.abs(scaled_zeta))) < _SMALL_GAIN_ARGUMENT
    result = np.zeros_like(t)
    if np.any(small):
        a = scaled_zeta[small]
        b = scaled_phi[small]
        b2 = b**2
        with np.errstate(over="ignore", invalid="ignore"):
            result[small] = (2.0 * (I0 / K)) * (
                a
                + b2 / 2.0
                + a**2 / 2.0
                + a * b2 / 2.0
                + b2**2 / 24.0
                + a**3 / 6.0
                + a**2 * b2 / 4.0
                + a * b2**2 / 24.0
                + b2**3 / 720.0
            )

    regular = ~small
    if np.any(regular):
        log_product = scaled_zeta[regular] + _log_cosh(scaled_phi[regular])
        with np.errstate(over="ignore", invalid="ignore"):
            result[regular] = (2.0 * I0 / K) * np.expm1(log_product)
    _check_finite("analytical expected gain", result)
    result[t == 0.0] = 0.0
    return result


def k_star(
    *, nu0: float, nu1: float, kappa: float, sigma: float, T: float
) -> float | None:
    """Return the sufficient finite-horizon bound, or None outside its domain."""
    nu0, nu1, kappa, sigma = _parameters(
        nu0=nu0, nu1=nu1, kappa=kappa, sigma=sigma
    )
    T = _scalar(T, "T")
    if T <= 0.0:
        raise ValueError("T must be positive")
    if kappa <= 0.0 or sigma <= 0.0:
        return None
    t = np.array([T], dtype=np.float64)
    mean, _ = _mean_variance(t, nu0, nu1, kappa, sigma)
    zeta = _zeta(t, kappa, sigma)[0]
    phi = mean[0] + 0.5 * sigma**2 * T
    if not np.isfinite(zeta) or zeta >= 0.0:
        raise FloatingPointError("K-star denominator is numerically invalid")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        bound = abs(phi) / zeta
    if not np.isfinite(bound):
        raise FloatingPointError("K-star is nonfinite for these parameters")
    return float(bound)
