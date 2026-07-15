# CausalTemp-XAI — Scientific Overview

*A benchmark for counterfactual explanations on temporal causal data, for ML
researchers. This document explains **what** each component does, **why** it
exists, and **where** it lives in the tree.*

---

## 1. The research question

Counterfactual (CF) explanations answer *"what minimal change to the input would
flip the model's prediction?"*. On i.i.d. tabular data they are judged by
**validity** (does the CF actually flip the label?), **proximity** (is it close
to the original?), and **sparsity** (does it change few features?). None of these
ask whether the CF is **causally coherent** — whether the proposed change is
something the data-generating process could actually produce.

For **time series** this gap is sharp: editing `x[t]` should, under the true
dynamics, propagate to `x[t+1], x[t+2], …`. A CF that rewrites the whole
trajectory arbitrarily may be valid and proximal yet describe a world that cannot
exist. **CausalTemp-XAI** makes this measurable by generating data from a
**known** structural causal model (SCM), so the *true* counterfactual is
computable and any proposed CF can be scored against it.

**Central thesis (what the benchmark is built to expose):** traditional CF
quality metrics and *causal faithfulness* can **disagree — even rank methods in
opposite order**. The repo provides the generators, metrics, classifiers, CF
methods, and harness to demonstrate and quantify that disagreement.

---

## 2. Conceptual framework at a glance

```
        SCM generator                     black-box model              CF method
   (known graph + mechanism)            (LSTM / TCN classifier)      (Wachter/DiCE/CARLA)
             │                                    │                        │
             ▼                                    ▼                        ▼
        X (N,T,k), Y (N,)  ───────────────▶  ŷ = f(X)   ◀──────────  x_cf for flip candidates
             │                                                            │
             │   graph + mechanism (ground truth)                         │
             └──────────────────────────┬─────────────────────────────────┘
                                         ▼
              ┌───────────────────────────────────────────────────────┐
              │  Evaluation                                            │
              │   • Axis-C:  validity · proximity · sparsity · OOD     │  ← traditional
              │   • CF-faith: noiseless_rollout  |  pearl_delta        │  ← causal (the point)
              │   • Foils:   Integrated-Gradients deletion/insertion   │
              │   • Robustness: Shift-VR (validity under noise shift)  │
              └───────────────────────────────────────────────────────┘
```

The **hinge** is that CF methods emit a fully-perturbed trajectory with *no
declared intervention point*, but causal scoring needs one. The benchmark
**derives** the intervention timestep uniformly for every method
(`derive_intervention_t`, §7) so all methods are scored on equal footing.

---

## 3. Repository map (what is where)

```
causaltemp_xai/                     installable package (the library)
├── config.py            §8   BenchmarkConfig dataclass + named presets (SMOKE/FULL/…)
├── data_io.py           §8   generate/persist/load datasets; stratified 60/20/20 split; CLI
├── eval.py              §6   evaluate_method (Axis-C + both CF-faith) + shift_vr
├── benchmark/
│   ├── generator.py     §4   LinearSCMT (VAR) and NlinearSCMT (MLP) data generators
│   ├── mechanisms.py    §4   Mechanism / LinearMechanism / MLPMechanism (the SCM transition)
│   └── structural_cf.py §5   oracle abduction-action-prediction counterfactual (ground truth)
├── metrics/
│   ├── cf_faith.py      §5   CFfaith — the causal metric (two semantics, hard+soft)
│   └── axis_c.py        §6   validity, proximity, sparsity, ood_plausibility
├── classifiers/
│   ├── lstm.py          §9   LSTMClassifier (last-timestep readout) — the harness default
│   └── tcn.py           §9   TCNClassifier (dilated causal convs, avg-pool)
├── methods/
│   ├── intervention.py  §7   derive_intervention_t — the uniform intervention rule
│   ├── wachter.py       §7   WachterCF — gradient CF
│   ├── dice.py          §7   DiCECF — dice-ml backend (method configurable) + DPP fallback
│   └── carla.py         §7   CARLARecourse — causal noiseless-rollout recourse
└── attribution/
    ├── integrated_gradients.py §10  IG saliency (the attribution "foil")
    └── perturbation_curves.py  §10  deletion / insertion AUC curves

experiments/                        the runnable harness (not imported by the library)
├── run_all.py           §8   end-to-end per-config run → results.json + per_instance.csv
├── run_suite.py         §8   multi-config driver; archives each config (no clobber)
├── report.py            §8   aggregates archives → docs/final_report.md
├── figures.py           §8   3 publication figures from a results.json
└── phenomenon_check.py  §8   fail-fast Wachter-vs-CARLA CF-faith guard

docs/
├── overview.md               this file
├── final_report.md      §11  current full-scale results (LSTM harness, supersedes below)
├── hypotheses_assessment.md  frozen v0.1 record (TCN-era) — superseded
├── general_plan.md / mvp_plan.md
└── plans/{mvp-v0.1-completion,nlinearscm-t}/  staged design docs + decisions log

tests/                              147 tests incl. golden numerical pins
data/scm_t/<config>/                generated datasets + checkpoints (gitignored)
```

---

## 4. The data generators (`benchmark/`)

Two SCM families share a **lagged causal graph** and a **binary label rule**, and
differ only in the transition mechanism. Both fix lag `L=1` in every shipped
preset (a single lag makes the temporal graph trivially acyclic and lets the
per-matrix spectral cap guarantee stationarity).

**Graph & label** (`generator.py`): the graph is a sparse random adjacency over
`k` variables per lag (`_sample_graph`, Bernoulli mask, no lag-1 self-loops). The
label is a **median threshold on the final-timestep value of variable 0**
(`Y = (x[T-1, 0] > median).astype(int)`), balanced by construction. *This
endpoint-based label is why classifier choice matters* (§9).

### LinearSCM-T — `LinearSCMT` (VAR)
A vector-autoregressive process with non-Gaussian innovations:

```
x_t = Σ_{l=1..L} A_l · x_{t-l} + eps_t ,     eps ~ Laplace (or Uniform)
```

Each lag matrix `A_l` is spectral-capped to radius ≤ 0.9 (`_stabilise`) for
stationarity. Because the true dynamics are linear and known, the exact
counterfactual is closed-form.

### NlinearSCM-T — `NlinearSCMT` (additive-noise MLP)
Same graph and label rule; linear `A·x` replaced by a **per-node 2-layer MLP**
with a leaky decay term and bounded activation:

```
x_t^i = decay_i · x_{t-1}^i + gain · tanh( MLP_i( masked lagged parents ) ) + eps_t^i
```

Weights are spectral-norm-capped (Lipschitz < 1) so trajectories stay bounded
over long horizons; a deterministic divergence-resample + clip fallback
guarantees finite output. **Additive noise is the key design choice**: it makes
Pearl abduction an *exact subtraction* (`eps = x − f(parents)`), so the causal
metric and the oracle CF carry over to the nonlinear case unchanged.

Tests generalisation **beyond linear VAR identifiability**. *Scope:* nonlinear
*transitions* only — nonlinear *mixing* `x = g(z)` (iVAE/CITRIS territory) is a
documented future extension.

### The `Mechanism` abstraction — `benchmark/mechanisms.py`
Linearity used to be hard-coded at four `A @ x` sites (generation, CF-faith,
CARLA rollout, persistence). All four now route through a polymorphic
`Mechanism` with:

| Method | Purpose |
|---|---|
| `forward_numpy(history)` | next-step mean for generation & CF-faith (numpy) |
| `forward_torch(history)` | **differentiable** next-step mean for CARLA (torch) |
| `state_dict` / `from_state_dict` | npz serialization (round-trips `LinearMechanism` and `MLPMechanism`) |

numpy and torch paths agree to <1e-5. This is what lets one SCM be *generated*,
*scored*, *optimised against*, and *saved* consistently.

---

## 5. The causal metric: CF-faith (`metrics/cf_faith.py`, `benchmark/structural_cf.py`)

`CFfaith.score(x_orig, x_cf, intervention_t, graph, mechanism) → {"hard", "soft"}`
is the benchmark's core contribution. A CF is faithful iff (i) it makes **no
pre-intervention change** (no retroactive causality) and (ii) post-intervention
values follow the **known mechanism**.

- **hard** ∈ {0,1}: 1 iff the pre-intervention prefix is untouched **and** the
  post-intervention L1 residual < `tol`. Any retroactive edit forces 0.
- **soft** ∈ (0,1]: `exp(−residual / scale)` — a continuous relaxation.

Two **semantics** are reported side by side (a single CF cannot be `hard=1` under
both — the contrast is itself a result):

| Semantics | Definition | Who satisfies it |
|---|---|---|
| `noiseless_rollout` (default) | roll the **deterministic** skeleton forward from the intervention: `x_pred[t] = mechanism.forward(window)` | CARLA (built to emit this) |
| `pearl_delta` | **abduction-action-prediction**: abduct `eps` from the factual, hold the intervened prefix, re-roll reusing `eps` | on-manifold / noise-preserving CFs |

For a **linear** mechanism `pearl_delta` reduces exactly to the homogeneous
recursion `delta[t] = Σ_l A_l·delta[t-l]`, so v0.1 numbers are unchanged
(pinned by golden tests); for nonlinear mechanisms it is the correct
generalisation.

**Oracle structural CF** (`structural_cf.py`) is the **positive control**: given
`do(x[t0, node] = value)` on a known additive-noise SCM it returns the
ground-truth CF (`abduct_noise` then re-roll). Two variants are mutually
exclusive by construction — the noisy/Pearl oracle scores `pearl_hard=1`, the
noiseless/skeleton oracle scores `rollout_hard=1` — which is how the nonlinear
path validates the whole metric stack **without any CF method or classifier**.

---

## 6. Traditional metrics & batch evaluation (`metrics/axis_c.py`, `eval.py`)

**Axis-C** (`axis_c.py`) — the four conventional axes:

| Function | Meaning | Better |
|---|---|---|
| `validity(x_cf, model, target_class)` | flip-rate: fraction predicted as target | higher |
| `proximity(x_orig, x_cf, norm="l1"\|"l2")` | mean distance to original | lower |
| `sparsity(x_orig, x_cf)` | fraction of unchanged cells ∈ [0,1] | higher |
| `ood_plausibility(x_train, x_cf)` | IsolationForest decision score | higher (in-dist) |

`eval.evaluate_method(...)` runs the full Axis-C suite **plus both CF-faith
semantics**, batch-averaged, returning one record per method. `eval.shift_vr(...)`
computes the **Shift-VR** robustness ratio (§10).

---

## 7. Counterfactual methods (`methods/`)

All expose a uniform `generate(x, model)` / `generate_batch(X, model)` interface;
causal methods additionally accept `(graph, mechanism)` and the harness routes
them by signature introspection.

| Method | File | Idea |
|---|---|---|
| **Wachter** | `wachter.py` | gradient descent on `CE(f(x_cf), target) + λ‖x_cf−x‖` — edits the whole trajectory |
| **DiCE** | `dice.py` | diverse CFs; official **dice-ml** backend (`method` configurable: `gradient`/`random`/…) with a from-scratch **DPP-diversity** fallback |
| **CARLA** | `carla.py` | *causal recourse*: intervenes at `t0` then rolls the **noiseless SCM** forward via `mechanism.forward_torch` — faithful to `noiseless_rollout` by construction |

**The uniform intervention rule** (`intervention.py`): `derive_intervention_t(x,
x_cf)` = first timestep where `|x_cf − x| > tol` (else `T-1`). Applied identically
to *every* method's output so CF-faith is comparable — this is the hinge that
makes cross-method causal scoring fair.

> Real CF methods on the **nonlinear** SCM are out of scope in this repo (a
> collaborator track); the nonlinear path uses the oracle CF (§5) instead.

---

## 8. Configuration, data I/O & the experiment harness

**Configs** (`config.py`) are code-defined `BenchmarkConfig` presets selected by
`--config`:

| Name | Mechanism | k | T | N | sparsity | purpose |
|---|---|---:|---:|---:|---:|---|
| `smoke` / `smoke_nl` | linear / mlp | 5 | 30 | 500 | 0.2 | CI / fast iteration |
| `full` / `full_nl` | linear / mlp | 10 | 100 | 10 000 | 0.2 | paper-scale results |
| `full_sparse` | linear | 10 | 100 | 10 000 | 0.1 | H5 sparsity hint |

`shifted_config(base)` derives an **Axis-D** environment: identical seed/graph/
mechanism, only the innovation distribution changes (Laplace→Uniform) — isolating
a pure distribution shift for Shift-VR.

**Data I/O** (`data_io.py`): `generate_and_save` / `load_dataset` write
`data/scm_t/<config>/` (X/Y `.npy`, `graph.npy`, `mechanism.npz`, `meta.json`);
stratified 60/20/20 split. Datasets are deterministic (seeded) and **never
committed** — regenerated on demand.

**The harness** (`experiments/`):

| Script | Role | Output |
|---|---|---|
| `run_all.py` | one config end-to-end: load data + LSTM ckpt, run 3 CF methods × (Axis-C + both CF-faith) + IG foil + Shift-VR | `experiments/results.json`, `per_instance.csv` (fixed paths) |
| `run_suite.py` | drives many configs; **snapshots** each into `experiments/results_archive/<config>/` so runs don't clobber | per-config archive + figures |
| `report.py` | aggregates the archive into a cross-config markdown report (Spearman ρ, H1/H3/H4/H5 verdicts, oracle control) | `docs/final_report.md` |
| `figures.py` | 3 figures (validity-vs-faith scatter, faith distributions, rank correlation) | PNGs |
| `phenomenon_check.py` | fail-fast guard that CARLA beats Wachter on rollout-faith | exit code |

Nonlinear configs auto-route inside `run_all.py` to the classifier-free
oracle-CF path (no checkpoint, no CF methods).

**Reproduce (full scale):**
```bash
uv sync --extra dev
uv run --offline python experiments/run_suite.py            # full, full_sparse, full_nl
uv run --offline python experiments/report.py --configs full full_sparse full_nl
```
(Use `--offline` locally to avoid a private-index re-resolve; `smoke`/`smoke_nl`
for a fast pass. CI runs the smoke pipeline in `.github/workflows/ci.yml`.)

---

## 9. Classifiers (`classifiers/`)

Both are sklearn-style wrappers over PyTorch that accept `(N, T, k)` arrays and
expose `predict`, `predict_proba`, `score`, and a differentiable `torch_logits`
(the gradient hook CF methods optimise through), plus `save`/`load`.

- **LSTMClassifier** (`lstm.py`) — stacked (optionally bidirectional) LSTM,
  **last-timestep hidden state → linear head**. This is the harness default
  (loads `lstm.pt`).
- **TCNClassifier** (`tcn.py`) — dilated *causal* convolutions (Bai et al. 2018),
  **global average pooling** → head.

**Why the choice matters:** the label is an **endpoint** threshold (§4). The
LSTM's last-timestep readout matches that target almost exactly (test acc ~0.99),
whereas the TCN's average pooling dilutes the endpoint signal (capped ~0.79). CF
*faithfulness* is classifier-agnostic, but *validity* is not — a stronger
classifier changes which methods look "valid" (see §11).

---

## 10. Foils & robustness

- **Attribution foil** (`attribution/`): Integrated Gradients saliency plus
  **deletion/insertion** perturbation curves. A saliency map can look good on
  these curves while saying nothing about causal faithfulness — the contrast that
  motivates a *causal* metric in the first place.
- **Shift-VR** (`eval.shift_vr`): with the **frozen** classifier, generate fresh
  CFs for a shifted-noise environment's inputs and report
  `validity(E_shift) / validity(E_base)` per method (nan sentinel when base
  validity is 0). Measures robustness of recourse to an innovation-distribution
  shift while holding the causal structure fixed.

---

## 11. What the current results say (`docs/final_report.md`)

Full-scale run (LSTM harness, test acc ~0.99, DiCE `method="random"`):

- **CARLA** is the only **causally faithful** method throughout
  (`rollout_hard = 1.0`); Wachter ≈ 0.0, DiCE ≈ 0.2.
- **The rank inversion is conditional on graph sparsity.** At `full`
  (sparsity 0.2) the strong classifier makes *every* method valid (1.0), so
  validity no longer discriminates and there is no inversion. At `full_sparse`
  (sparsity 0.1) **CARLA validity collapses to 0** and the validity↔faith
  Spearman ρ = **−0.87** returns — the traditional and causal metrics rank the
  methods in opposite order.
- On the denser graph the tension shows up in **proximity/sparsity vs faith**
  rather than validity vs faith.
- **Nonlinear (`full_nl`)**: oracle positive controls pass
  (`OracleCF-Pearl → pearl_hard=1`, `OracleCF-Rollout → rollout_hard=1`),
  confirming the metric stack is correct on MLP mechanisms.

This refines the frozen v0.1 story (`docs/hypotheses_assessment.md`), where a
weaker avg-pool TCN made the CARLA collapse and the inversion look universal.

---

## 12. Where to plug in your own work

| You want to… | Do this |
|---|---|
| **Add a CF method** | implement `generate(x, model)` (+ optional `(graph, mechanism)`) in `methods/`, add it to `build_methods` in `run_all.py` |
| **Add a metric** | add a function in `metrics/axis_c.py` and thread it through `eval.evaluate_method` |
| **Add an SCM family** | subclass `Mechanism` (numpy+torch+serialize) and add a generator in `benchmark/generator.py`; register a preset in `config.py` |
| **Add a classifier** | mirror the `LSTMClassifier` API (esp. `torch_logits`) so gradient CF methods work unchanged |
| **New benchmark instance** | add a `BenchmarkConfig` to `config.py`; everything downstream is config-driven |

**Correctness guarantees:** the linear path is pinned **bit-for-bit** by
`tests/test_golden_linear.py`; `MLPMechanism` numpy/torch parity, trajectory
boundedness, oracle positive controls, and abduction exactness are all under test
(147 tests). Run `uv run --offline pytest tests/ -q`.

---

## 13. Key references (methods this repo instantiates)

- Wachter, Mittelstadt & Russell (2017) — counterfactual explanations.
- Mothilal, Sharma & Tan (2020) — DiCE (diverse CFs).
- Pawelczyk et al. (2021) — CARLA (algorithmic recourse benchmark).
- Sundararajan, Taly & Yan (2017) — Integrated Gradients.
- Hoyer et al. (2008); Nasr-Esfahany et al. (2023) — additive-noise / bijective
  SCM identifiability (why abduction is exact here).
- Bai, Kolter & Koltun (2018) — TCN architecture.
- Pearl — abduction-action-prediction counterfactual recipe.
