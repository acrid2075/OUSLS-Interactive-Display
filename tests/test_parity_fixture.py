"""Keep the shared TypeScript parity fixture anchored to the Python reference."""

import json
from pathlib import Path

import numpy as np

from reference.simulation import expected_gain, generate_price_paths, k_star, simulate_sls


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "numerical_parity.json"


def test_shared_fixed_innovation_fixture_matches_python_reference():
    fixture = json.loads(FIXTURE_PATH.read_text())
    normals = np.asarray(fixture["normals"], dtype=np.float64)
    snapshot_steps = fixture["snapshot_steps"]
    return_steps = fixture["return_steps"]

    for case in fixture["cases"].values():
        parameters = case["parameters"]
        prices = generate_price_paths(normals, **parameters)
        np.testing.assert_allclose(
            prices["log_prices"][:, snapshot_steps],
            case["price_snapshots"]["log_prices"],
            rtol=2e-12,
            atol=2e-12,
        )
        np.testing.assert_allclose(
            prices["prices"][:, snapshot_steps],
            case["price_snapshots"]["prices"],
            rtol=2e-12,
            atol=2e-12,
        )
        np.testing.assert_allclose(
            prices["return_increments"][:, return_steps],
            case["price_snapshots"]["return_increments"],
            rtol=2e-12,
            atol=2e-12,
        )

        for K in (2.0, -2.0, 0.0, 1e-12):
            expected = case["controller"][format(K, "g")]
            gains = simulate_sls(prices["return_increments"], I0=5000.0, K=K)
            for field, output in (
                ("long_gain_snapshots", "long_gain"),
                ("short_gain_snapshots", "short_gain"),
                ("total_gain_snapshots", "total_gain"),
            ):
                np.testing.assert_allclose(
                    gains[output][:, snapshot_steps],
                    expected[field],
                    rtol=2e-12,
                    atol=2e-10,
                )
            np.testing.assert_allclose(
                gains["total_gain"][:, -1],
                expected["terminal_total_gain"],
                rtol=2e-12,
                atol=2e-10,
            )
            np.testing.assert_allclose(
                np.mean(gains["total_gain"][:, -1]),
                expected["terminal_mean_gain"],
                rtol=2e-12,
                atol=2e-10,
            )

            theory_times = prices["times"][case["theory_steps"]]
            theory = expected_gain(
                theory_times,
                **{
                    key: parameters[key]
                    for key in ("nu0", "nu1", "kappa", "sigma")
                },
                I0=5000.0,
                K=K,
            )
            np.testing.assert_allclose(
                theory,
                case["theory"][format(K, "g")],
                rtol=2e-12,
                atol=2e-10,
            )

        actual_bound = k_star(
            nu0=parameters["nu0"],
            nu1=parameters["nu1"],
            kappa=parameters["kappa"],
            sigma=parameters["sigma"],
            T=parameters["T"],
        )
        if case["k_star"] is None:
            assert actual_bound is None
        else:
            np.testing.assert_allclose(actual_bound, case["k_star"], rtol=2e-12, atol=2e-12)
