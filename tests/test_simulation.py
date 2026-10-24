"""Numerical contract tests for the documented signed-K SLS reference."""

import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss

from reference.simulation import (
    expected_gain,
    generate_price_paths,
    k_star,
    moments,
    simulate_sls,
)


def test_forward_euler_uses_old_log_price_and_ito_return_increment():
    normals = np.zeros((1, 2))
    result = generate_price_paths(
        normals, nu0=0.3, nu1=0.4, kappa=0.5, sigma=0.2, T=1.0
    )

    dt = 0.5
    # Step 0: b = nu0; step 1: b = nu0 + nu1*dt - kappa*X_1.
    x1 = (0.3 * dt)
    x2 = x1 + (0.3 + 0.4 * dt - 0.5 * x1) * dt
    np.testing.assert_allclose(result["times"], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(result["log_prices"], [[0.0, x1, x2]])
    np.testing.assert_allclose(result["prices"], np.exp([[0.0, x1, x2]]))
    np.testing.assert_allclose(
        result["return_increments"],
        [[x1 + 0.5 * 0.2**2 * dt, (x2 - x1) + 0.5 * 0.2**2 * dt]],
    )


def test_forward_euler_uses_same_fixed_noise_increments_for_price_and_returns():
    normals = np.array([[1.0, -0.5]])
    sigma, dt = 0.3, 0.5
    result = generate_price_paths(
        normals, nu0=0.1, nu1=0.2, kappa=0.4, sigma=sigma, T=1.0
    )
    dlog = np.diff(result["log_prices"], axis=1)
    np.testing.assert_allclose(
        result["return_increments"], dlog + 0.5 * sigma**2 * dt
    )
    # The same Brownian increment appears in both expressions, not a second draw.
    np.testing.assert_allclose(
        dlog,
        np.array([[0.1 * dt + sigma * np.sqrt(dt),  # X_0 = 0
                   (0.1 + 0.2 * dt - 0.4 * result["log_prices"][0, 1]) * dt
                   - 0.5 * sigma * np.sqrt(dt)]]),
    )


@pytest.mark.parametrize("K", [2.0, -2.0])
def test_signed_k_controller_updates_both_legs_from_old_gain_state(K):
    increments = np.array([[0.1, -0.04, 0.02]])
    I0 = 5000.0
    result = simulate_sls(increments, I0=I0, K=K)
    long_gain = np.zeros(4)
    short_gain = np.zeros(4)
    for n, dr in enumerate(increments[0]):
        old_long, old_short = long_gain[n], short_gain[n]
        long_gain[n + 1] = old_long + (I0 + K * old_long) * dr
        short_gain[n + 1] = old_short + (-I0 - K * old_short) * dr

    np.testing.assert_allclose(result["long_gain"][0], long_gain)
    np.testing.assert_allclose(result["short_gain"][0], short_gain)
    np.testing.assert_allclose(
        result["total_gain"], result["long_gain"] + result["short_gain"]
    )


def test_zero_k_cancels_total_gain_pathwise_for_arbitrary_return_increments():
    increments = np.array([[0.1, -0.03, 0.2], [-0.2, 0.04, 0.01]])
    result = simulate_sls(increments, I0=5000.0, K=0.0)
    np.testing.assert_allclose(result["total_gain"], 0.0, atol=0.0)
    np.testing.assert_allclose(
        result["long_gain"], -result["short_gain"], atol=1e-12
    )
    np.testing.assert_allclose(result["total_gain"][:, 0], 0.0, atol=0.0)


def test_moments_have_continuous_kappa_zero_limits():
    times = np.array([0.0, 0.25, 1.0])
    result = moments(times, nu0=0.3, nu1=-0.2, kappa=0.0, sigma=0.4)
    np.testing.assert_allclose(result["mean"], 0.3 * times - 0.1 * times**2)
    np.testing.assert_allclose(result["variance"], 0.4**2 * times)


def test_moments_remain_accurate_for_near_zero_kappa():
    times = np.array([0.0, 0.25, 1.0])
    parameters = dict(nu0=0.3, nu1=-0.2, sigma=0.4)
    exact_limit = moments(times, **parameters, kappa=0.0)
    near_limit = moments(times, **parameters, kappa=1e-12)
    np.testing.assert_allclose(near_limit["mean"], exact_limit["mean"], rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(near_limit["variance"], exact_limit["variance"], rtol=1e-11, atol=1e-13)


def test_moments_match_closed_forms_for_nonzero_kappa():
    times = np.array([0.0, 0.4, 1.0])
    nu0, nu1, kappa, sigma = 0.2, -0.1, 0.7, 0.3
    result = moments(
        times, nu0=nu0, nu1=nu1, kappa=kappa, sigma=sigma
    )
    mean = (nu0 / kappa - nu1 / kappa**2) * (1 - np.exp(-kappa * times))
    mean += nu1 * times / kappa
    variance = sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * times))
    np.testing.assert_allclose(result["mean"], mean, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(result["variance"], variance, rtol=1e-12, atol=1e-14)


def test_expected_gain_zero_time_zero_k_and_odd_signed_k():
    times = np.array([0.0, 0.3, 1.0])
    common = dict(nu0=0.2, nu1=0.05, kappa=1.5, sigma=0.1, I0=5000.0)
    zero_k = expected_gain(times, **common, K=0.0)
    positive_k = expected_gain(times, **common, K=2.0)
    negative_k = expected_gain(times, **common, K=-2.0)

    np.testing.assert_array_equal(zero_k, np.zeros_like(times))
    assert positive_k[0] == 0.0
    np.testing.assert_allclose(negative_k, -positive_k, rtol=1e-12, atol=1e-12)


def test_expected_gain_matches_documented_continuous_time_formula():
    times = np.array([0.2, 0.7, 1.0])
    nu0, nu1, kappa, sigma, I0, K = 0.15, -0.04, 1.2, 0.25, 5000.0, -1.7
    model_moments = moments(
        times, nu0=nu0, nu1=nu1, kappa=kappa, sigma=sigma
    )
    phi = model_moments["mean"] + 0.5 * sigma**2 * times
    zeta = 0.5 * model_moments["variance"] - 0.5 * sigma**2 * times
    expected = 2 * I0 / K * (np.exp(K**2 * zeta) * np.cosh(K * phi) - 1)
    actual = expected_gain(
        times,
        nu0=nu0,
        nu1=nu1,
        kappa=kappa,
        sigma=sigma,
        I0=I0,
        K=K,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)


def test_expected_gain_matches_independent_gaussian_quadrature():
    """Integrate the paper's two pathwise legs, without calling moments()."""
    nodes, weights = hermgauss(100)
    nu0, nu1, kappa, sigma, I0, K = 0.15, -0.04, 1.2, 0.25, 5000.0, -1.7
    times = np.array([0.2, 0.7, 1.0])

    def independent_moments(t):
        # Direct closed forms, evaluated independently of reference.moments.
        decay = np.exp(-kappa * t)
        mean = (nu0 / kappa - nu1 / kappa**2) * (1.0 - decay) + nu1 * t / kappa
        variance = sigma**2 * (-np.expm1(-2.0 * kappa * t)) / (2.0 * kappa)
        return mean, variance

    quadrature = []
    for t in times:
        mean, variance = independent_moments(t)
        gaussian_samples = mean + np.sqrt(2.0 * variance) * nodes
        long_leg = I0 / K * (
            np.exp(K * gaussian_samples + 0.5 * K * sigma**2 * t - 0.5 * K**2 * sigma**2 * t)
            - 1.0
        )
        short_leg = I0 / K * (
            np.exp(-K * gaussian_samples - 0.5 * K * sigma**2 * t - 0.5 * K**2 * sigma**2 * t)
            - 1.0
        )
        quadrature.append(np.dot(weights, long_leg + short_leg) / np.sqrt(np.pi))

    actual = expected_gain(
        times, nu0=nu0, nu1=nu1, kappa=kappa, sigma=sigma, I0=I0, K=K
    )
    np.testing.assert_allclose(actual, quadrature, rtol=2e-12, atol=2e-10)


@pytest.mark.parametrize("K", [1.7, -1.7, 1e-9])
def test_continuous_time_total_path_formula_is_sum_of_paper_legs(K):
    sigma, I0, t = 0.25, 5000.0, 0.7
    # Deterministically check the two Lemma 1 expressions against their compact sum.
    for x in (-0.3, 0.0, 0.2):
        long_leg = I0 / K * np.expm1(
            K * x + 0.5 * K * sigma**2 * t - 0.5 * K**2 * sigma**2 * t
        )
        short_leg = I0 / K * np.expm1(
            -K * x - 0.5 * K * sigma**2 * t - 0.5 * K**2 * sigma**2 * t
        )
        total_from_legs = long_leg + short_leg
        c = -0.5 * K**2 * sigma**2 * t
        z = K * (x + 0.5 * sigma**2 * t)
        # Stable form of exp(c) cosh(z) - 1, preserving the small-K limit.
        compact_inner = np.expm1(c) * np.cosh(z) + 2.0 * np.sinh(0.5 * z) ** 2
        total_compact = 2.0 * I0 / K * compact_inner
        np.testing.assert_allclose(total_from_legs, total_compact, rtol=2e-12, atol=2e-9)


def test_expected_gain_uses_kappa_zero_continuous_limits():
    times = np.array([0.0, 0.5, 1.0])
    nu0, nu1, sigma, I0, K = 0.2, -0.1, 0.3, 5000.0, 1.5
    phi = (nu0 + 0.5 * sigma**2) * times + 0.5 * nu1 * times**2
    # At kappa=0, variance=sigma^2*t, hence zeta=0.
    expected = 2 * I0 / K * (np.cosh(K * phi) - 1)
    actual = expected_gain(
        times,
        nu0=nu0,
        nu1=nu1,
        kappa=0.0,
        sigma=sigma,
        I0=I0,
        K=K,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)


def test_expected_gain_has_stable_near_zero_k_limit_and_deterministic_case():
    times = np.array([0.0, 0.3, 1.0])
    common = dict(nu0=0.2, nu1=-0.1, kappa=1.5, sigma=0.1, I0=5000.0)
    model_moments = moments(times, **{k: common[k] for k in ("nu0", "nu1", "kappa", "sigma")})
    phi = model_moments["mean"] + 0.5 * common["sigma"] ** 2 * times
    zeta = 0.5 * model_moments["variance"] - 0.5 * common["sigma"] ** 2 * times
    K = 1e-12
    near_zero = expected_gain(times, **common, K=K)
    first_order_limit = common["I0"] * K * (2.0 * zeta + phi**2)
    np.testing.assert_allclose(near_zero, first_order_limit, rtol=1e-12, atol=1e-24)

    sigma_zero = expected_gain(
        times, nu0=0.2, nu1=-0.1, kappa=1.5, sigma=0.0, I0=5000.0, K=0.5
    )
    deterministic_mean = moments(
        times, nu0=0.2, nu1=-0.1, kappa=1.5, sigma=0.0
    )["mean"]
    np.testing.assert_allclose(
        sigma_zero,
        2.0 * 5000.0 / 0.5 * (np.cosh(0.5 * deterministic_mean) - 1.0),
        rtol=1e-12,
        atol=1e-12,
    )
    zero_drift = expected_gain(
        times, nu0=0.0, nu1=0.0, kappa=0.0, sigma=0.0, I0=5000.0, K=1e200
    )
    np.testing.assert_array_equal(zero_drift, np.zeros_like(times))


def test_finite_horizon_bound_is_only_returned_in_theorem_domain():
    nu0, nu1, kappa, sigma, T = 0.2, -0.1, 1.5, 0.3, 1.0
    mean = (nu0 / kappa - nu1 / kappa**2) * (1 - np.exp(-kappa * T))
    mean += nu1 * T / kappa
    phi = mean + 0.5 * sigma**2 * T
    variance = sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * T))
    zeta = 0.5 * variance - 0.5 * sigma**2 * T
    np.testing.assert_allclose(
        k_star(nu0=nu0, nu1=nu1, kappa=kappa, sigma=sigma, T=T),
        abs(phi) / zeta,
        rtol=1e-12,
    )
    assert k_star(nu0=nu0, nu1=nu1, kappa=0.0, sigma=sigma, T=T) is None
    assert k_star(nu0=nu0, nu1=nu1, kappa=kappa, sigma=0.0, T=T) is None
    assert k_star(nu0=nu0, nu1=nu1, kappa=-0.1, sigma=sigma, T=T) is None


def test_sufficient_bound_implies_positive_analytical_terminal_expectation():
    nu0, nu1, kappa, sigma, T, I0 = 0.2, -0.1, 1.5, 0.3, 1.0, 5000.0
    bound = k_star(nu0=nu0, nu1=nu1, kappa=kappa, sigma=sigma, T=T)
    assert bound <= 0.0
    value = expected_gain(
        np.array([T]),
        nu0=nu0,
        nu1=nu1,
        kappa=kappa,
        sigma=sigma,
        I0=I0,
        K=bound - 0.5,
    )
    assert value[0] > 0.0


def test_fixed_noise_end_to_end_shapes_identity_and_shared_path_sweep():
    # Compose the documented primitives as a minimal reproducible experiment.
    normals = np.random.default_rng(7).standard_normal((100, 50))
    prices = generate_price_paths(
        normals, nu0=0.05, nu1=0.01, kappa=1.5, sigma=0.1, T=1.0
    )
    repeated_prices = generate_price_paths(
        normals, nu0=0.05, nu1=0.01, kappa=1.5, sigma=0.1, T=1.0
    )
    selected = simulate_sls(prices["return_increments"], I0=5000.0, K=2.0)
    sweep = np.stack(
        [
            simulate_sls(prices["return_increments"], I0=5000.0, K=K)[
                "total_gain"
            ][:, -1]
            for K in (-2.0, 0.0, 2.0)
        ]
    )
    theory = expected_gain(
        prices["times"],
        nu0=0.05,
        nu1=0.01,
        kappa=1.5,
        sigma=0.1,
        I0=5000.0,
        K=2.0,
    )

    assert prices["times"].shape == (51,)
    assert prices["prices"].shape == (100, 51)
    assert selected["total_gain"].shape == (100, 51)
    assert sweep.shape == (3, 100)
    assert theory.shape == (51,)
    assert np.all(prices["prices"] > 0)
    np.testing.assert_allclose(prices["times"], np.linspace(0.0, 1.0, 51))
    np.testing.assert_allclose(np.diff(prices["times"]), 0.02)
    # Identical innovations and parameters reproduce prices exactly; the same
    # return increments are then reused for all K values in the sweep.
    np.testing.assert_array_equal(prices["prices"], repeated_prices["prices"])
    np.testing.assert_array_equal(
        prices["return_increments"], repeated_prices["return_increments"]
    )
    np.testing.assert_allclose(selected["total_gain"][:, 0], 0.0, atol=0.0)
    np.testing.assert_allclose(sweep[1], 0.0, atol=0.0)
    selected_again = simulate_sls(prices["return_increments"], I0=5000.0, K=-2.0)
    assert selected_again["total_gain"].shape == selected["total_gain"].shape


def test_invalid_inputs_are_rejected_with_clear_errors():
    with pytest.raises(ValueError, match="2-dimensional"):
        generate_price_paths(np.zeros(3), nu0=0.0, nu1=0.0, kappa=0.0, sigma=0.1)
    with pytest.raises(ValueError, match="finite"):
        generate_price_paths(np.array([[np.nan]]), nu0=0.0, nu1=0.0, kappa=0.0, sigma=0.1)
    with pytest.raises(ValueError, match="sigma must be nonnegative"):
        generate_price_paths(np.zeros((1, 1)), nu0=0.0, nu1=0.0, kappa=0.0, sigma=-0.1)
    with pytest.raises(ValueError, match="T must be positive"):
        generate_price_paths(np.zeros((1, 1)), nu0=0.0, nu1=0.0, kappa=0.0, sigma=0.1, T=0.0)
    with pytest.raises(ValueError, match="I0 must be positive"):
        simulate_sls(np.zeros((1, 1)), I0=0.0, K=1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        moments([-0.1], nu0=0.0, nu1=0.0, kappa=1.0, sigma=0.1)
    with pytest.raises(ValueError, match="finite"):
        expected_gain([1.0], nu0=0.0, nu1=0.0, kappa=1.0, sigma=0.1, I0=5000.0, K=np.inf)


def test_numerical_overflow_is_reported_instead_of_clipped():
    with pytest.raises(FloatingPointError, match="nonfinite price paths"):
        generate_price_paths(
            np.zeros((1, 1)), nu0=1000.0, nu1=0.0, kappa=0.0, sigma=0.0
        )
    with pytest.raises(FloatingPointError, match="nonfinite long gain"):
        simulate_sls(np.array([[1e308]]), I0=5000.0, K=0.0)
    with pytest.raises(FloatingPointError, match="scaled expected-gain argument"):
        expected_gain(
            [1.0], nu0=0.2, nu1=0.0, kappa=1.0, sigma=0.1, I0=5000.0, K=1e200
        )


def test_positive_kappa_with_unstable_euler_step_is_rejected():
    with pytest.raises(ValueError, match="not mean-square stable"):
        generate_price_paths(
            np.zeros((1, 1)), nu0=0.0, nu1=0.0, kappa=2.1, sigma=0.1, T=1.0
        )
