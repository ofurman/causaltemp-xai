# CausalTemp-XAI

**Benchmarking Counterfactual Explanations for Temporal Causal Data**

[![CI](https://github.com/Healthpy/causaltemp-xai/actions/workflows/ci.yml/badge.svg)](https://github.com/Healthpy/causaltemp-xai/actions/workflows/ci.yml)

## Description


`causaltemp-xai` provides:

- **LinearSCM-T** - a VAR(L) benchmark generator with an explicit causal graph
  and non-Gaussian noise, so the true counterfactual distribution is known.
- **NlinearSCM-T** - the nonlinear sibling: same graph, additive-noise per-node
  MLP transition mechanisms (spectral-norm-capped for stability). Tests
  generalization beyond linear VAR identifiability. See below.
- **CFfaith** - a hard/soft metric that checks whether a proposed CF respects
  the causal mechanisms of the data-generating process.
- **Axis-C metrics** - four complementary quality axes (validity, proximity,
  sparsity, OOD plausibility) that together characterise a CF explanation.
- **CF method stubs** - a uniform `generate(x, model)` interface for Wachter,
  DiCE, and CARLA-style recourse, ready for implementation or wrapping.

## Installation

All Python tooling goes through [`uv`](https://docs.astral.sh/uv/) (never pip):

```bash
git clone https://github.com/Healthpy/causaltemp-xai.git
cd causaltemp-xai
uv sync --extra dev        # creates .venv + uv.lock, installs deps + dev extras
```

## Reproduce v0.1

The full benchmark runs end-to-end from a clean checkout. Datasets and the TCN
checkpoint are **regenerated deterministically** (seeded), never committed.

```bash
# 0. environment
uv sync --extra dev

# 1. generate the locked paper dataset (k=10, L=1, T=100, N=10_000, Laplace noise)
uv run python -m causaltemp_xai.data_io --config full

# 2. train + freeze the LSTM classifier -> data/scm_t/full/lstm.pt
#    (this is the checkpoint the harness loads; --config full trains at paper scale)
uv run python -m causaltemp_xai.classifiers.lstm --config full --train --patience 20

# 3. run the harness: 3 CF methods x Axis-C + both CF-faith metrics
#    + IG attribution foil + Shift-VR-lite -> experiments/results.json (+ per_instance.csv)
uv run python experiments/run_all.py --config full --n-cf 100

# 4. render the 3 publication figures -> experiments/figures/
uv run python experiments/figures.py --results experiments/results.json
```

Swap `--config full` for `--config smoke` (k=5, T=30, N=500) for a fast pass; the
smoke pipeline is what CI runs. The harness defaults to the official `dice-ml`
gradient backend; pass `--no-dice-ml` to use the fast from-scratch DPP fallback
(much quicker on the large `full` config). See
[`docs/hypotheses_assessment.md`](docs/hypotheses_assessment.md) for the H1/H3/H4
verdicts and the go/no-go decision.

## NlinearSCM-T (nonlinear benchmark)

`NlinearSCM-T` is the nonlinear sibling of LinearSCM-T: it keeps the same lagged
causal graph but replaces the VAR coefficients with **additive-noise per-node
MLP transitions**

```
x_t^i = decay_i · x_{t-1}^i  +  gain · tanh( MLP_i(masked lagged parents) )  +  eps_t^i
```

with spectral-norm-capped weights so trajectories stay bounded over long
horizons. It tests generalization **beyond linear VAR identifiability**. Because
the noise is **additive**, Pearl abduction is an exact subtraction
(`eps = x − f(parents)`), so `CFfaith` (both the rollout and the
abduction-based `pearl_delta` semantics) and the **oracle structural-CF** carry
over unchanged to the nonlinear mechanisms.

```bash
# 1. generate the nonlinear dataset (smoke: k=5, T=30, N=500 / full_nl mirrors `full`)
uv run python -m causaltemp_xai.data_io --config smoke_nl

# 2a. oracle-CF path (default): classifier-free positive control
uv run python experiments/run_all.py --config smoke_nl               # or full_nl

# 2b. real-CF path: train an LSTM, then run Wachter/DiCE/CARLA + oracle controls
uv run python -m causaltemp_xai.classifiers.lstm --config full_nl --train --patience 20
uv run python experiments/run_all.py --config full_nl --nl-mode real --n-cf 100 --dice-method random
```

By default (`--nl-mode oracle`) `run_all.py` routes nonlinear configs to a
**classifier-free oracle-CF path**: it needs no checkpoint and runs no real CF
methods, scoring CF-faith on the Stage-4 oracle structural counterfactual as a
built-in positive control. Two mutually-exclusive oracle variants are emitted —
the Pearl oracle scores `pearl_hard=1` and the noiseless (skeleton) oracle scores
`rollout_hard=1` by construction — demonstrating the rollout-vs-pearl contrast on
nonlinear data.

With **`--nl-mode real`** the harness instead trains/loads an LSTM on the
nonlinear data and runs the full real-CF pipeline (Wachter/DiCE/CARLA + the full
Axis-C, CF-faith, IG-attribution and Shift-VR metric suite), keeping the two
oracle variants as ground-truth positive controls. CARLA's recourse is already
mechanism-generic — it rolls the differentiable `MLPMechanism.forward_torch`
forward — so it is faithful by construction (`rollout_hard=1`) on MLP mechanisms
just as on the linear VAR.

**Scope:** this ships nonlinear *transitions* only. Nonlinear **mixing**
`x = g(z)` (an invertible observation map over latents) and **non-additive**
noise are a separate identifiability axis (iVAE/CITRIS) and are documented as a
**future extension** — see the [`docs/plans/nlinearscm-t/`](docs/plans/nlinearscm-t/index.md)
Backlog.

## Quickstart (library API)

```python
import numpy as np
from causaltemp_xai.benchmark.generator import LinearSCMT
from causaltemp_xai.classifiers import TCNClassifier
from causaltemp_xai.metrics import CFfaith, validity, proximity, sparsity

# 1. Generate synthetic causal time-series
gen = LinearSCMT(k=5, L=1, sparsity=0.2, noise_type="laplace", T=30, N=500, seed=0)
data = gen.generate()
# data["X"]          shape (500, 30, 5)  -- multivariate time-series
# data["Y"]          shape (500,)        -- binary labels
# data["graph"]      shape (5, 5, 1)     -- adjacency per lag
# data["mechanism"]  Mechanism object    -- SCM transition (LinearMechanism here)

X, Y = data["X"], data["Y"]

# 2. Train a TCN classifier (sklearn-style wrapper; (N, T, k) in/out)
clf = TCNClassifier(n_inputs=5, max_epochs=30)
clf.fit(X[:400], Y[:400], X[400:], Y[400:])

# 3. Evaluate causal faithfulness of a counterfactual
scorer = CFfaith(tol=1e-3, scale=1.0)              # default noiseless-rollout semantics
x_orig = X[0]                  # shape (T, k)
x_cf   = X[0].copy()           # build your CF here

result = scorer.score(
    x_orig, x_cf,
    intervention_t=10,
    graph=data["graph"],
    mechanism=data["mechanism"],
)
print(result)   # {"hard": ..., "soft": ...}

# 4. Axis-C metrics
print("validity :", validity(x_cf[np.newaxis], clf, target_class=1))
print("proximity:", proximity(x_orig, x_cf, norm="l1"))
print("sparsity :", sparsity(x_orig, x_cf))
```

## Evaluation Axes

| Axis | Function / Class | Description |
|------|-----------------|-------------|
| **Validity** | `validity(x_cf, model, target_class)` | Flip-rate: fraction of CFs the model assigns to `target_class`. |
| **Proximity** | `proximity(x_original, x_cf, norm="l1")` | Mean L1 (or L2) distance between CF and original. Lower is better. |
| **Sparsity** | `sparsity(x_original, x_cf)` | Fraction of unchanged features in [0, 1]. Higher is sparser. |
| **OOD Plausibility** | `ood_plausibility(x_train, x_cf)` | IsolationForest decision score; higher means more in-distribution. |
| **CF-faith (hard)** | `CFfaith.score(...)["hard"]` | 1.0 iff the CF exactly follows SCM mechanisms from the intervention time. |
| **CF-faith (soft)** | `CFfaith.score(...)["soft"]` | exp(-L1 residual / scale); continuous relaxation of hard faithfulness. |

Two CF-faith *semantics* are reported side by side (`CFfaith(semantics=...)`):
`"noiseless_rollout"` (default — what CARLA-causal is built to satisfy) and
`"pearl_delta"` (Pearl delta-recursion). A single CF cannot be `hard=1` under
both; the contrast is itself a benchmark result.

## Running Tests

```bash
uv run pytest tests/ -q
```

## Project Structure

```
causaltemp-xai/
├── causaltemp_xai/
│   ├── config.py                # BenchmarkConfig + SMOKE/FULL presets, shifted_config
│   ├── data_io.py               # stratified 60/20/20 split, generate/load datasets (+CLI)
│   ├── eval.py                  # evaluate_method (Axis-C + both CF-faith) + shift_vr
│   ├── benchmark/
│   │   ├── generator.py         # LinearSCM-T VAR(L) + NlinearSCM-T MLP generators
│   │   ├── mechanisms.py        # Mechanism / LinearMechanism / MLPMechanism (+ serialize)
│   │   └── structural_cf.py     # oracle abduction-action-prediction counterfactual
│   ├── metrics/
│   │   ├── cf_faith.py          # CFfaith (noiseless_rollout | pearl_delta semantics)
│   │   └── axis_c.py            # validity, proximity, sparsity, ood_plausibility
│   ├── classifiers/tcn.py       # TCN + TCNClassifier wrapper (+ train CLI)
│   ├── attribution/
│   │   ├── integrated_gradients.py   # hand-rolled IG attribution foil (WP3)
│   │   └── perturbation_curves.py    # deletion / insertion curves
│   └── methods/
│       ├── intervention.py      # derive_intervention_t (uniform rule)
│       ├── wachter.py           # WachterCF (gradient CF)
│       ├── dice.py              # DiCECF (dice-ml gradient + DPP fallback)
│       └── carla.py             # CARLARecourse (causal noiseless-rollout recourse)
├── experiments/
│   ├── run_all.py               # end-to-end harness -> results.json + per_instance.csv
│   ├── figures.py               # 3 publication figures -> experiments/figures/
│   └── phenomenon_check.py      # fail-fast Wachter-vs-CARLA CF-faith guard
├── docs/hypotheses_assessment.md  # H1/H3/H4 verdicts + go/no-go
├── tests/
├── notebooks/01_data_exploration.ipynb
└── data/scm_t/                 # generated datasets + checkpoints (gitignored)
```

## License

MIT
