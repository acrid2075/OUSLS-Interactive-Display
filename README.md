# SLS / OUSLS research simulator

This repository is the source for a reproducible browser simulator for signed-K
SLS on trending Ornstein–Uhlenbeck log prices. Access the site at https://acrid2075.github.io/OUSLS-Interactive-Display/.

## Project layout

- `reference/` — NumPy reference model and scientific diagnostics.
- `src/model/` — typed simulation contracts, seeded randomness, price process,
  controller, theory, and experiment assembly.
- `src/workers/` — browser worker entry points for simulation requests.
- `src/ui/` — controls, application state, and methodology text.
- `src/charts/` — linked price and gain visualizations.
- `tests/` — Python, TypeScript, and browser verification.
- `.github/workflows/` — CI and GitHub Pages workflows.

The frontend uses TypeScript, Vite, Plotly.js, and a dedicated Web Worker.
Python/NumPy is the reference implementation. `package-lock.json` pins the
browser dependencies; `requirements-dev.txt` pins the Python test dependencies.

## Local development and checks

```sh
npm ci
npm run dev
```

Before submitting changes, run the same checks as CI:

```sh
npm run typecheck
npm run build
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
npm run test:ts-parity
```

The Pages workflow is manually triggered from GitHub Actions. It performs the
static TypeScript check and production build before uploading `dist/`.

## Model conventions

The baseline uses `T = 1`, 50 Euler–Maruyama intervals, 100 paths, and 51
stored timestamps. The initial single-`nu` interface convention is
`nu_0 = nu`, `nu_1 = 0`; both drift coefficients remain available in the model.
Until an authoritative OUSLS definition is supplied, the controller should be
described as signed-K SLS rather than as a distinct OUSLS rule.
