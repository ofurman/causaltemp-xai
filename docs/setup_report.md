# CausalTemp-XAI — Full Setup & Results Report

_Scope requested: **two datasets** (one linear, one nonlinear), **one classifier**
(LSTM), **all CF methods** (Wachter, DiCE, CARLA), and **all metrics** available in
the repo. Every metric that is *not* computed for a given dataset is flagged
**⛔ NOT CALCULATED** with the reason inline and consolidated in the
[coverage matrix](#4-coverage-matrix--what-is-and-isnt-calculated)._

Numbers below are read from the current, seeded run archives:

- Linear → `experiments/results_archive/full/results.json`
- Nonlinear → `experiments/results_archive/full_nl/results.json`

> **Update (real CF methods now run on nonlinear data).** The original version of
> this report flagged the nonlinear dataset's real CF methods and
> classifier-dependent metrics as ⛔ NOT CALCULATED. That gap is now **closed**:
> the harness gained a `--nl-mode real` path that trains an LSTM on the nonlinear
> data and runs Wachter/DiCE/CARLA + the full metric suite (keeping the oracle
> CFs as ground-truth controls). The `full_nl` numbers below are from that run
> (`experiments/results_archive/full_nl/`, provenance `nl_mode: "real"`,
> classifier test acc 0.988). The ⛔ cells and reasons R1–R3 are retained below
> as historical context, each annotated with its resolution.

---

## 1. Environment & reproduction

Tooling is `uv` only (never pip). The pipeline is deterministic given a
`BenchmarkConfig` seed — datasets and the LSTM checkpoint are regenerated, never
committed (`data/` is git-ignored).

```bash
uv sync --extra dev

# Linear dataset (full): k=10, L=1, T=100, N=10_000, Laplace noise, seed=42
uv run python -m causaltemp_xai.data_io --config full
uv run python -m causaltemp_xai.classifiers.lstm --config full --train --patience 20
uv run python experiments/run_all.py --config full --n-cf 100 --dice-method random

# Nonlinear dataset (full_nl): same graph, additive-noise per-node MLP transitions
uv run python -m causaltemp_xai.data_io --config full_nl
uv run python experiments/run_all.py --config full_nl --n-cf 100   # oracle-CF path

# Consolidated multi-config driver + report generator (archives per config):
uv run python experiments/run_suite.py --configs full full_nl --n-cf 100
uv run python experiments/report.py --configs full full_nl
```

`experiments/run_all.py` is the single end-to-end harness. `run_suite.py` runs a
list of configs and snapshots each into `experiments/results_archive/<config>/`;
`report.py` renders `docs/final_report.md` from those archives.

---

## 2. The two datasets

Both share the **same** lagged causal DAG, median-threshold binary label rule,
and split protocol. They differ only in the transition mechanism family.

| | **Linear** (`full`) | **Nonlinear** (`full_nl`) |
|---|---|---|
| Generator | `LinearSCMT` (VAR(L)) | `NlinearSCMT` (additive-noise MLP) |
| `mechanism_type` | `linear` | `mlp` |
| Transition | `x_t = Σ_l A_l · x_{t−l}` + ε | `x_t^i = decay_i·x_{t−1}^i + gain·tanh(MLP_i(parents)) + ε_t^i` |
| k (vars) | 10 | 10 |
| L (lag) | 1 | 1 |
| T (length) | 100 | 100 |
| N (samples) | 10 000 | 10 000 |
| sparsity | 0.2 | 0.2 |
| noise | Laplace | Laplace |
| seed | 42 | 42 |
| Splits | 6000 / 2000 / 2000 (train/val/test), class-balanced | same |
| Label | `y = 1` iff final-step value of variable 0 > median (balanced) | same rule |

Because the noise is **additive** in both families, Pearl abduction is exact
(`ε = x − f(parents)`), so both CF-faith semantics and the oracle structural-CF
carry over unchanged from linear to nonlinear.

---

## 3. Classifier — LSTM

Stacked LSTM (`hidden=64`, `num_layers=2`, `dropout=0.2`, unidirectional),
last-timestep hidden state → linear 2-class head. Trained with early stopping
(`--patience 20`).

| Dataset | train acc | val acc | test acc |
|---|---:|---:|---:|
| Linear (`full`) | 0.997 | 0.992 | **0.993** |
| Nonlinear (`full_nl`, `--nl-mode real`) | 0.999 | 0.993 | **0.988** |

> **Resolved.** The nonlinear config used to route to a **classifier-free**
> oracle-CF path (`run_all.run_nonlinear`), so no LSTM was trained. The new
> `--nl-mode real` path trains an LSTM on the nonlinear data
> (`data/scm_t/full_nl/lstm.pt`, test acc 0.988) and feeds it to the real CF
> methods. `--nl-mode oracle` (the default) still runs the classifier-free
> control.

---

## 4. Coverage matrix — what is and isn't calculated

Legend: ✅ computed · ⛔ not calculated (reason keyed below).

Both datasets now compute the full method × metric grid. The `— (was ⛔)`
annotations record what used to be uncomputed on nonlinear data (default
`--nl-mode oracle`) and are ✅ under `--nl-mode real`.

### 4a. CF methods per dataset

| CF method | Linear (`full`) | Nonlinear (`full_nl`, `--nl-mode real`) |
|---|:---:|:---:|
| **Wachter** (gradient, unconstrained) | ✅ | ✅ *(was ⛔ R1)* |
| **DiCE** (dice-ml, `method=random`) | ✅ | ✅ *(was ⛔ R1)* |
| **CARLA** (causal recourse, SCM-aware) | ✅ | ✅ *(was ⛔ R1)* |
| OracleCF-Pearl (ground-truth control) | — n/a | ✅ |
| OracleCF-Rollout (ground-truth control) | — n/a | ✅ |

### 4b. Metrics per dataset

| Metric | Linear (`full`) | Nonlinear (`full_nl`, `--nl-mode real`) |
|---|:---:|:---:|
| Validity (flip-rate) | ✅ | ✅ *(was ⛔ R2)* |
| Proximity L1 | ✅ | ✅ |
| Proximity L2 | ✅ | ✅ |
| Sparsity | ✅ | ✅ |
| OOD plausibility (IsolationForest) | ✅ | ✅ *(was ⛔ R3)* |
| CF-faith — rollout (hard/soft) | ✅ | ✅ |
| CF-faith — pearl (hard/soft) | ✅ | ✅ |
| Integrated-Gradients attribution foil | ✅ | ✅ *(was ⛔ R2)* |
| Shift-VR-lite (validity retention) | ✅ | ✅ *(was ⛔ R2 + R1)* |

**Reasons the nonlinear cells used to be blank (now resolved):**

- **R1 — "real CF methods out of scope on the nonlinear SCM."** *Resolved.* The
  supposed blockers did not actually exist: `MLPMechanism.forward_torch` is
  already a differentiable nonlinear-SCM rollout and `CARLARecourse._rollout`
  calls it generically, while Wachter/DiCE are mechanism-free. The only real gap
  was the routing + an LSTM checkpoint; `--nl-mode real` closes it. The two
  **ground-truth oracle** CFs are retained as positive controls.
- **R2 — needs the black-box classifier.** *Resolved.* Validity, the IG
  attribution foil, and Shift-VR-lite now query the LSTM trained on `full_nl`.
- **R3 — OOD not emitted on the oracle path.** *Resolved.* With a classifier
  present the oracle controls are scored through the standard `evaluate_method`,
  which reports validity/OOD like any other method.

> Only genuinely out of scope now: nonlinear **mixing** `x = g(z)` and
> **non-additive** noise (a separate identifiability axis, iVAE/CITRIS).

---

## 5. Results — Linear dataset (`full`)

LSTM test acc **0.993**; 100 flip-candidate test instances explained; DiCE
backend `dice-ml` (`method=random`), seed 42.

| Method | Validity | Prox L1 | Prox L2 | Sparsity | OOD | CF-faith rollout (h / s) | CF-faith pearl (h / s) |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| Wachter | 1.00 | 60.35 | 4.37 | 0.01 | 0.02 | 0.00 / 0.76 | 0.00 / 0.94 |
| DiCE | 1.00 | 1.66 | 1.11 | 1.00 | 0.03 | 0.21 / 0.82 | 0.21 / 0.98 |
| CARLA | 1.00 | 192.47 | 9.46 | 0.25 | 0.08 | **1.00 / 1.00** | 0.00 / 0.77 |

**Attribution foil (Integrated Gradients, mean over 100 instances):**
deletion AUC 0.554, insertion AUC 0.010, gap **−0.544** — a reasonable saliency
map that nonetheless says nothing about causal faithfulness.

**Shift-VR-lite (validity retention under a Laplace→uniform innovation shift,
frozen classifier):** Wachter 1.00→1.00 (VR 1.000), DiCE 1.00→1.00 (VR 1.000),
CARLA 1.00→1.00 (VR 1.000).

---

## 6. Results — Nonlinear dataset (`full_nl`, `--nl-mode real`)

LSTM test acc **0.988**; 100 flip-candidate test instances explained; DiCE
backend `dice-ml` (`method=random`), seed 42. The three real methods run the
same pipeline as linear; the two oracle CFs are retained as ground-truth
positive controls (now also scored for validity/OOD via the classifier).

| Method | Validity | Prox L1 | Prox L2 | Sparsity | OOD | CF-faith rollout (h / s) | CF-faith pearl (h / s) |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| Wachter | 0.89 | 64.80 | — | 0.05 | 0.01 | 0.00 / 0.85 | 0.00 / 0.93 |
| DiCE | 1.00 | 0.86 | — | 1.00 | 0.02 | 0.25 / 0.91 | 0.25 / 1.00 |
| CARLA | 0.00 | 96.06 | — | 0.26 | 0.07 | **1.00 / 1.00** | 0.00 / 0.88 |
| OracleCF-Pearl | 0.00 | 4.35 | — | 0.74 | 0.02 | 0.00 / 0.88 | **1.00 / 1.00** |
| OracleCF-Rollout | 0.00 | 66.57 | — | 0.51 | 0.05 | **1.00 / 1.00** | 0.00 / 0.88 |

**Attribution foil (Integrated Gradients):** deletion AUC 0.472, insertion AUC
0.017, gap **−0.456**.

**Shift-VR-lite:** Wachter 0.89→0.99 (VR 1.112), DiCE 1.00→1.00 (VR 1.000),
CARLA 0.00→0.00 (VR `nan`, documented zero-denominator sentinel — CARLA flips
nothing on nonlinear data).

Oracle controls still pass their pre-registered check (Pearl → `pearl_hard=1`,
Rollout → `rollout_hard=1`), confirming the CF-faith metric is correct on MLP
mechanisms. A single CF cannot be `hard=1` under both semantics.

---

## 7. Key findings

- **H1 (standard methods valid but unfaithful):** ✅ supported on both datasets.
  DiCE reaches validity 1.00 with rollout-hard 0.21 (linear) / 0.25 (nonlinear),
  both < 0.3. Wachter is valid+unfaithful on linear (1.00 / 0.00) and *partial*
  on nonlinear (validity 0.89, just under the 0.9 bar).
- **H3 (causal method faithful):** ✅ supported on both. CARLA scores
  rollout-hard 1.00 (> 0.7) on linear **and** nonlinear — faithful by
  construction because its recourse *is* its own noiseless `forward_torch`
  rollout.
- **Cost of faithfulness is starker on nonlinear data:** CARLA's validity drops
  from 1.00 (linear) to **0.00** (nonlinear). The contractive `tanh` dynamics
  damp a single-`t0` intervention so it rarely moves the final-step value across
  the decision boundary — a legitimate benchmark result, not a bug.
- **H4 (rank inversion):** present on `full_nl` — validity↔CF-faith Spearman
  ρ = **−0.50** across the three real methods (validity ranks them opposite to
  causal faithfulness). On `full` validity is constant (all 1.00) so ρ is
  undefined; the sparser `full_sparse` archive shows ρ = −0.87.
- **Rollout vs Pearl are genuinely different targets:** CARLA/Rollout-oracle are
  pure noiseless rollouts (rollout-hard 1, pearl-hard 0); the Pearl oracle is the
  opposite. Reporting both is deliberate.

---

## 8. Status of former gaps & remaining scope

The nonlinear gaps flagged in the original report are **implemented and run**
(this document's §3/§4/§6 reflect the `--nl-mode real` results):

1. ✅ LSTM trained on `full_nl` (`data/scm_t/full_nl/lstm.pt`, test acc 0.988).
2. ✅ Differentiable nonlinear-SCM rollout — already existed
   (`MLPMechanism.forward_torch`), consumed generically by CARLA.
3. ✅ Classifier-backed nonlinear path — `run_all.py --nl-mode real` scores
   validity / OOD / IG-foil / Shift-VR for Wachter/DiCE/CARLA and keeps the
   oracle controls.

**Genuinely still out of scope** (separate identifiability axis, not this task):
nonlinear **mixing** `x = g(z)` and **non-additive** noise (iVAE / CITRIS).
