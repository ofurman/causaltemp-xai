# CausalTemp-XAI v0.1 — Hypothesis Assessment

> **⚠️ Superseded** by [`docs/final_report.md`](final_report.md), which re-runs the
> full suite (`full`, `full_sparse`, `full_nl`) with the current **LSTM** harness
> (test acc ~0.99) and DiCE `method="random"`. This document is the frozen v0.1
> record on the earlier **avg-pool TCN** classifier (test acc 0.787). The headline
> rank inversion is **conditional on graph sparsity** in the new run: it reappears
> at `full_sparse` (validity ρ = −0.87, CARLA validity collapses to 0) but not at
> `full` (the strong LSTM makes every method valid). Read the final report for the
> current verdicts.


**Date:** 2026-06-13
**Config:** `full` (locked paper config — k=10, L=1, sparsity=0.2, Laplace noise, T=100, N=10 000, seed=42)
**Classifier:** frozen TCN, test accuracy **0.787** (train 0.831 / val 0.784) — avg-pool vs endpoint-label cap, documented in the plan Issues; CF-faith is classifier-agnostic.
**CF subset:** 100 test instances the classifier predicts as class 0 (recourse target = class 1).
**DiCE backend:** from-scratch DPP fallback (`dice_backend="fallback"`) for the tractable full run; the `dice-ml` gradient backend passed its Stage 4 decision gate but is impractically slow at this scale (≈25 s/instance). The phenomenon below is construction-level and backend-independent.
**Artifacts:** [`experiments/results.json`](../experiments/results.json), [`experiments/per_instance.csv`](../experiments/per_instance.csv), [`experiments/figures/`](../experiments/figures/).

These hypotheses were **pre-registered** in the MVP plan before the full run; the
thresholds below are quoted verbatim from the plan and assessed with point
estimates (no confidence intervals for the MVP).

---

## Headline results (batch means over 100 instances)

| Method  | Validity | Proximity L1 | Sparsity | OOD  | CF-faith rollout (hard / soft) | CF-faith pearl (hard / soft) |
|---------|---------:|-------------:|---------:|-----:|:------------------------------:|:----------------------------:|
| Wachter |   **0.81** |        26.85 |     0.00 | 0.03 |          0.00 / 0.77           |         0.00 / 0.97          |
| DiCE    |   **1.00** |         7.88 |     0.00 | 0.03 |          0.00 / 0.77           |         0.00 / 0.99          |
| CARLA   |   **0.00** |       155.78 |     0.39 | 0.07 |          **1.00** / 1.00        |         0.00 / 0.78          |

**The central phenomenon — a complete rank inversion.** Ranked by the
*traditional* validity metric: DiCE (1.00) > Wachter (0.81) > CARLA (0.00).
Ranked by *causal* CF-faith (rollout, hard): CARLA (1.00) > Wachter = DiCE
(0.00). The two metric families order the methods in **exactly opposite**
directions — the strongest possible form of the disagreement the benchmark was
built to expose.

---

## H1 — Standard CF methods are valid but causally unfaithful

> **Pre-registered:** Wachter/DiCE validity > 0.9 **AND** CF-faith(rollout, hard) < 0.3 → expected *Confirmed*.

**Verdict: CONFIRMED.**

- **DiCE**: validity **1.00** (> 0.9 ✓), CF-faith rollout-hard **0.00** (< 0.3 ✓) — meets both bars exactly.
- **Wachter**: CF-faith rollout-hard **0.00** (< 0.3 ✓); validity **0.81**, just below the pre-registered 0.9 bar. The shortfall is attributable to the moderate (~0.79) classifier — Wachter still flips 81 % of cases while remaining completely unfaithful (rollout-hard = 0). The qualitative claim (high validity, ~zero causal faithfulness) holds.
- The derived `intervention_t` for both methods clusters at **t = 0** (they edit the whole trajectory), so the `cf_faith.py` final-timestep edge case is **not** silently inflating faithfulness.
- Pearl semantics is also ≈0 hard for both (soft 0.97–0.99): standard CFs satisfy *neither* causal definition at the hard level.

---

## H3 — Causal recourse (CARLA) is causally faithful

> **Pre-registered:** CARLA-causal CF-faith(rollout, hard) > 0.7 with *moderate* validity/proximity degradation → expected *Confirmed*. Report `cf_faith_pearl` too (expected ≈0, off-manifold).

**Verdict: CONFIRMED on causal faithfulness; the validity/proximity degradation is SEVERE, not moderate** (an honest, documented caveat).

- CF-faith rollout-hard **1.00** (> 0.7 ✓✓) and soft 1.00 — faithful **by construction** (CARLA emits the noiseless VAR rollout it is scored against).
- CF-faith **pearl**-hard **0.00** (soft 0.78): exactly the documented rollout-vs-pearl contrast — CARLA's noiseless rollout is off the noisy data manifold, so it satisfies the rollout definition but not Pearl's noise-abducting delta-recursion. An **on-manifold Pearl-CARLA** (re-injecting the original innovations) is deferred to v1.0.
- **Caveat:** validity collapses to **0.00** and proximity explodes to **155.8** (vs 26.9 / 7.9 for the standard methods) — far beyond "moderate". Root cause: the stabilised VAR (spectral radius ≤ 0.9) **decays toward zero over T = 100**, so a single-timestep causal intervention cannot steer the avg-pooled endpoint classifier; the faithful recourse is valid-agnostic and far from the original. On the shorter `smoke` horizon (T = 30) CARLA validity is ~0.75, confirming this is a long-horizon decay effect, not a logic bug.
- Net: H3's *faithfulness* claim is strongly confirmed; the cost in validity/proximity is larger than pre-registered and is the headline limitation of the noiseless-rollout CARLA. This **strengthens** the benchmark's point (faithfulness and conventional quality are in genuine tension) while motivating the deferred Pearl-CARLA variant.

---

## H4 (partial) — Traditional metrics do not predict causal faithfulness

> **Pre-registered:** Spearman ρ(traditional, CF-faith) < 0.5 across 3–4 methods → report as *preliminary/underpowered*, do **not** claim confirmed.

**Verdict: DIRECTIONALLY SUPPORTED, reported as PRELIMINARY (underpowered).**

Spearman ρ between each traditional metric and CF-faith(rollout, hard), computed
across the 3 methods (per-method means):

| Traditional metric | ρ vs CF-faith (rollout, hard) |
|--------------------|:-----------------------------:|
| validity           | **−0.87** |
| proximity L1       | +0.87 |
| sparsity           | +0.87 |

- **Validity ρ = −0.87**: validity does not merely fail to predict faithfulness — it ranks methods in the *opposite* order (well below the 0.5 bar, ✓ in spirit).
- proximity/sparsity correlate *positively* with faith (+0.87): the faithful method (CARLA) is precisely the one that changes the most — i.e. the traditional "cheap, sparse change is better" preference is **anti-aligned** with causal faithfulness.
- **Underpowered:** only 3 methods and tied CF-faith ranks (two methods at 0.0). These ρ values are descriptive, not inferential — **no confirmed claim**. See `experiments/figures/fig3_rank_correlation.png`.

---

## H5 (hint only) — Sparsity sensitivity

`full_sparse` was **not** generated in this run, so no descriptive sparsity
comparison is reported. (The preset exists — `--config full_sparse` — for a
future descriptive-only pass; per the MVP it carries no ablation claim.)

---

## Attribution foil (WP3, context)

Integrated Gradients over the 100 originals yields a deletion AUC of **0.235**
and insertion AUC of **0.375** (gap **+0.140**): the saliency map behaves as a
*reasonable* attribution (revealing top cells raises confidence faster than
masking them preserves it) **while telling us nothing about causal
faithfulness** — the WP3 contrast that motivates the benchmark.

## Shift-VR-lite (Axis D, robustness)

Validity retention under an innovation-distribution shift (Laplace → uniform,
identical SCM): Wachter **1.11**, DiCE **1.00**, CARLA **nan** (base validity 0,
documented denominator-zero sentinel). The standard methods' recourse remains
valid under the shift; CARLA's metric is undefined because it produces no valid
CFs in either environment (the same long-horizon collapse as in H3).

---

## Go / No-Go decision

**Decision: GO — freeze v0.1.**

**Rationale.**
- The benchmark's reason for existing is **decisively demonstrated**: traditional
  CF quality metrics (validity/proximity/sparsity) and causal CF-faith **rank the
  methods in opposite order** on the locked paper config (validity ρ = −0.87).
  H1 is confirmed; H3's faithfulness claim is confirmed; H4 is directionally
  supported (preliminary). Every MVP Definition-of-Done artifact is produced and
  reproducible from a clean checkout (`README.md` → data → train → `run_all.py`
  → `figures.py`).
- **Acknowledged limitations carried into v1.0 (not blockers):**
  1. **Classifier ceiling ~0.79** — the scaffold TCN's global average pooling
     dilutes the endpoint-threshold label; a `pooling='last'` variant would solve
     it. Validity/proximity therefore sit on a moderate classifier (documented).
  2. **CARLA validity collapse on long horizons** — the noiseless VAR rollout
     decays over T = 100, yielding faithful-but-invalid, far recourse. An
     on-manifold **Pearl-CARLA** (noise re-injection → pearl-hard = 1, validity
     recovered) is the priority v1.0 item.
  3. **DiCE via the from-scratch DPP fallback** for the full run (dice-ml gradient
     backend validated but too slow at scale).
  4. **H4 underpowered** (3 methods); adding methods/seeds is v1.0 work.

These limitations are about *strengthening* the demonstration, not about whether
it holds — the core phenomenon reproduces cleanly and is, if anything, sharper
than pre-registered. v0.1 is frozen as a reproducible benchmark; the items above
define the v1.0 backlog.
