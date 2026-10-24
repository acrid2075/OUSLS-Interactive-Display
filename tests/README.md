# Python reference tests

These tests define the test-facing contract for the NumPy reference described
in `SLS_OUSLS_Codex_Handoff.md`. The repository currently has no implementation,
website, or Pages workflow, so collection/execution depends on those artifacts
being added. The tests intentionally import `reference.simulation` and check
the documented equations; they do not add a second simulator or assume a
separate OUSLS formula.

Expected reference API:

- `generate_price_paths(normals, *, nu0, nu1, kappa, sigma, T=1.0)` returns a
  mapping with `times` `[N+1]`, `log_prices` and `prices` `[M,N+1]`, and
  `return_increments` `[M,N]`. `normals` is `[M,N]` and contains the fixed
  standard-normal innovations used for reproducibility.
- `simulate_sls(return_increments, *, I0, K)` returns a mapping with
  `long_gain`, `short_gain`, and `total_gain`, each `[M,N+1]`.
- `moments(times, *, nu0, nu1, kappa, sigma)` returns a mapping with `mean`
  and `variance`, each shaped like `times`.
- `expected_gain(times, *, nu0, nu1, kappa, sigma, I0, K)` returns the
  continuous-time analytical expected total dollar gain, shaped like `times`.
- `k_star(*, nu0, nu1, kappa, sigma, T)` returns the finite-horizon bound for
  the theorem's domain and `None` outside it.

The Pages artifact test checks the proposed Vite/GitHub Pages delivery contract
from repository files. It is an integration test for deployment wiring, not a
network check against a published URL. Its failures include the missing file
or contract in the assertion message because no Pages app exists yet.

## Python and TypeScript numerical parity

`fixtures/numerical_parity.json` contains the actual fixed `[100, 50]` normal
innovations and Python reference outputs. It covers stable, zero, near-zero,
and negative `kappa`, plus zero `sigma`; signed `K = 2`, `K = -2`, `K = 0`,
and near-zero `K = 1e-12`. The stored checkpoints
include intermediate log prices, prices, Ito return increments, both gain legs,
total gains, analytical expectations, and terminal results. The fixture stores
the innovations themselves; the RNG seed is only provenance and is not used to
claim parity.

Run `python -m pytest -q tests/test_simulation.py tests/test_parity_fixture.py`
to check the Python reference and fixture. Run
`node --experimental-strip-types tests/typescript_parity.mjs` to execute the
TypeScript engine against those same innovations and Python-generated values.
Regenerate the fixture only when the Python reference contract changes, using
`python tests/generate_parity_fixture.py`, then verify both commands.
