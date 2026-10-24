"""Regenerate the shared Python/TypeScript fixed-innovation parity fixture."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from reference.simulation import expected_gain, generate_price_paths, k_star, simulate_sls


OUTPUT = ROOT / "fixtures" / "numerical_parity.json"
SNAPSHOT_STEPS = [0, 1, 2, 25, 50]
RETURN_STEPS = [0, 1, 24, 49]
THEORY_STEPS = [0, 1, 25, 50]
K_VALUES = [2.0, -2.0, 0.0, 1e-12]


def columns(values: np.ndarray, indices: list[int]) -> list[list[float]]:
    return values[:, indices].tolist()


def main() -> None:
    rng = np.random.Generator(np.random.PCG64(314159))
    normals = rng.standard_normal((100, 50))
    cases = {
        "stable": {"nu0": 0.05, "nu1": 0.01, "kappa": 1.5, "sigma": 0.1, "T": 1.0},
        "zero_kappa": {"nu0": 0.2, "nu1": -0.1, "kappa": 0.0, "sigma": 0.3, "T": 1.0},
        "near_zero_kappa": {"nu0": 0.2, "nu1": -0.1, "kappa": 1e-12, "sigma": 0.3, "T": 1.0},
        "negative_kappa": {"nu0": 0.05, "nu1": -0.02, "kappa": -0.3, "sigma": 0.15, "T": 1.0},
        "zero_sigma": {"nu0": -0.1, "nu1": 0.05, "kappa": 1.2, "sigma": 0.0, "T": 1.0},
    }
    fixture_cases: dict[str, object] = {}

    for case_name, parameters in cases.items():
        prices = generate_price_paths(normals, **parameters)
        controller_cases: dict[str, object] = {}
        theory_cases: dict[str, object] = {}
        for K in K_VALUES:
            k_key = format(K, "g")
            gains = simulate_sls(prices["return_increments"], I0=5000.0, K=K)
            controller_cases[k_key] = {
                "long_gain_snapshots": columns(gains["long_gain"], SNAPSHOT_STEPS),
                "short_gain_snapshots": columns(gains["short_gain"], SNAPSHOT_STEPS),
                "total_gain_snapshots": columns(gains["total_gain"], SNAPSHOT_STEPS),
                "terminal_total_gain": gains["total_gain"][:, -1].tolist(),
                "terminal_mean_gain": float(np.mean(gains["total_gain"][:, -1])),
            }
            theory_cases[k_key] = expected_gain(
                prices["times"][THEORY_STEPS],
                **{key: parameters[key] for key in ("nu0", "nu1", "kappa", "sigma")},
                I0=5000.0,
                K=K,
            ).tolist()

        fixture_cases[case_name] = {
            "parameters": parameters,
            "price_snapshots": {
                "log_prices": columns(prices["log_prices"], SNAPSHOT_STEPS),
                "prices": columns(prices["prices"], SNAPSHOT_STEPS),
                "return_increments": columns(prices["return_increments"], RETURN_STEPS),
            },
            "controller": controller_cases,
            "theory_steps": THEORY_STEPS,
            "theory": theory_cases,
            "k_star": k_star(
                nu0=parameters["nu0"],
                nu1=parameters["nu1"],
                kappa=parameters["kappa"],
                sigma=parameters["sigma"],
                T=parameters["T"],
            ),
        }

    fixture = {
        "description": "Fixed Python-generated innovations and expected reference outputs for TS parity.",
        "innovation_source": {
            "generator": "NumPy Generator(PCG64)",
            "seed": 314159,
            "numpy_version": np.__version__,
            "shape": [100, 50],
        },
        "normals": normals.tolist(),
        "snapshot_steps": SNAPSHOT_STEPS,
        "return_steps": RETURN_STEPS,
        "cases": fixture_cases,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(fixture, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
