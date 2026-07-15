"""End-to-end CausalTemp-XAI benchmark harness.

Loads a persisted LinearSCM-T dataset and its frozen TCN checkpoint (Stages 2-3),
runs the three counterfactual methods (Wachter, DiCE, CARLA-causal) over a subset
of the test set, scores each with the full Axis-C + **both** CF-faith metric
suites (Stage 5), adds the Integrated-Gradients attribution foil (Stage 6) and the
Shift-VR-lite robustness metric (Stage 7), then writes ``results.json`` plus a
per-instance dump for the figures.

Usage
-----
    uv run python experiments/run_all.py --config smoke           # CI smoke
    uv run python experiments/run_all.py --config full --n-cf 100  # paper results
    uv run python experiments/run_all.py --config smoke_nl        # NlinearSCM-T

For nonlinear configs (``smoke_nl`` / ``full_nl``, ``mechanism_type='mlp'``) the
run routes to a dedicated oracle-structural-CF path: no LSTM checkpoint and no
real CF methods are needed (CF methods on the nonlinear SCM are the collaborator's
track — see plan Backlog #2), and CF-faith is exercised via the Stage-4 oracle
positive control. See ``run_nonlinear``.

Outputs (under ``experiments/``):
    results.json            one summary record per method + attribution + shift-vr
                            + provenance (config, classifier accuracy, seed, n_cf)
    per_instance.csv        one row per (method, instance) for the scatter / violins
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaltemp_xai.attribution import (  # noqa: E402
    deletion_curve,
    insertion_curve,
    integrated_gradients,
)
from causaltemp_xai.benchmark.structural_cf import (  # noqa: E402
    structural_counterfactual,
)
from causaltemp_xai.classifiers import LSTMClassifier  # noqa: E402
from causaltemp_xai.config import get_config, shifted_config  # noqa: E402
from causaltemp_xai.data_io import (  # noqa: E402
    DEFAULT_OUT_DIR,
    generate_and_save,
    load_dataset,
    stratified_split,
)
from causaltemp_xai.eval import evaluate_method, shift_vr  # noqa: E402
from causaltemp_xai.methods import CARLARecourse, DiCECF, WachterCF  # noqa: E402
from causaltemp_xai.methods.intervention import derive_intervention_t  # noqa: E402
from causaltemp_xai.metrics.axis_c import proximity, sparsity  # noqa: E402
from causaltemp_xai.metrics.cf_faith import CFfaith  # noqa: E402

EXP_DIR = ROOT / "experiments"
TARGET_CLASS = 1


# ---------------------------------------------------------------------------
# CF-method plumbing
# ---------------------------------------------------------------------------


def _squeeze_cfs(cfs: np.ndarray) -> np.ndarray:
    """Collapse a ``(N, n_cfs, T, k)`` DiCE batch to ``(N, T, k)`` (first CF)."""
    cfs = np.asarray(cfs, dtype=np.float32)
    if cfs.ndim == 4:
        return cfs[:, 0]
    return cfs


def _generate(method, X, clf, graph, mech) -> np.ndarray:
    """Generate one CF per instance, routing graph/mechanisms to causal methods."""
    import inspect

    params = inspect.signature(method.generate_batch).parameters
    if "graph" in params or "mechanism" in params:
        cfs = method.generate_batch(X, clf, graph, mech)
    else:
        cfs = method.generate_batch(X, clf)
    return _squeeze_cfs(cfs)


class _DiCESingle:
    """Adapter exposing ``generate_batch(X, model) -> (N, T, k)`` for shift_vr.

    ``shift_vr``/``validity`` expect a 3-D CF batch, but ``DiCECF`` returns a
    ``(N, n_cfs, T, k)`` diverse set. This wrapper keeps the uniform
    ``(X, model)`` signature (so ``eval._generate_batch`` routes it as a
    non-causal method) and returns the first CF per instance.
    """

    def __init__(self, dice: DiCECF) -> None:
        self._dice = dice

    def generate_batch(self, X, model):
        return _squeeze_cfs(self._dice.generate_batch(X, model))


def build_methods(X_train: np.ndarray, use_dice_ml: bool, dice_method: str = "gradient"):
    """Construct the three CF methods. DiCE gets a (capped) background set."""
    background = X_train[: min(200, len(X_train))]
    wachter = WachterCF(target_class=TARGET_CLASS, n_steps=300, lr=0.1)
    dice = DiCECF(
        target_class=TARGET_CLASS,
        n_cfs=1,
        n_steps=300,
        background_data=background,
        use_dice_ml=use_dice_ml,
        method=dice_method,
    )
    carla = CARLARecourse(
        target_class=TARGET_CLASS, n_steps=300, t0_fractions=(0.25, 0.5)
    )
    return {"Wachter": wachter, "DiCE": dice, "CARLA": carla}


def select_flip_candidates(clf, X_test, n_cf, target_class=TARGET_CLASS):
    """Indices of test instances the classifier predicts as NOT ``target_class``.

    Recourse for these is meaningful (there is a class to flip *to*). Falls back
    to the first ``n_cf`` instances if too few candidates exist.
    """
    preds = clf.predict(X_test)
    src = [i for i in range(len(X_test)) if preds[i] != target_class]
    if len(src) < n_cf:
        src = list(range(len(X_test)))
    return np.asarray(src[:n_cf], dtype=int)


# ---------------------------------------------------------------------------
# Per-instance metric dump (for the scatter / violins)
# ---------------------------------------------------------------------------


def per_instance_records(clf, X_sel, CFs, graph, mech, method_name):
    """One record per instance with per-cf metrics (for figures.py)."""
    rollout = CFfaith(semantics="noiseless_rollout")
    pearl = CFfaith(semantics="pearl_delta")
    preds = np.asarray(clf.predict(CFs)).reshape(-1)
    rows = []
    for i, (x, x_cf) in enumerate(zip(X_sel, CFs)):
        t = derive_intervention_t(x, x_cf)
        r = rollout.score(x, x_cf, t, graph, mech)
        p = pearl.score(x, x_cf, t, graph, mech)
        rows.append(
            {
                "method": method_name,
                "instance": int(i),
                "validity": int(preds[i] == TARGET_CLASS),
                "proximity_l1": proximity(x, x_cf, norm="l1"),
                "proximity_l2": proximity(x, x_cf, norm="l2"),
                "sparsity": sparsity(x, x_cf),
                "intervention_t": int(t),
                "cf_faith_rollout_hard": r["hard"],
                "cf_faith_rollout_soft": r["soft"],
                "cf_faith_pearl_hard": p["hard"],
                "cf_faith_pearl_soft": p["soft"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Attribution foil (Integrated Gradients + deletion/insertion curves)
# ---------------------------------------------------------------------------


def attribution_block(clf, X_sel, ig_steps=64, curve_steps=50):
    """Mean deletion/insertion AUC of IG saliency over the selected instances.

    The foil: a faithful saliency method can score well here (sharp deletion
    drop / insertion rise) while telling us nothing about causal faithfulness.
    """
    del_aucs, ins_aucs = [], []
    for x in X_sel:
        ig = integrated_gradients(clf, x, TARGET_CLASS, steps=ig_steps)
        _, del_auc = deletion_curve(clf, x, ig, TARGET_CLASS, n_steps=curve_steps)
        _, ins_auc = insertion_curve(clf, x, ig, TARGET_CLASS, n_steps=curve_steps)
        del_aucs.append(del_auc)
        ins_aucs.append(ins_auc)
    return {
        "method": "IntegratedGradients",
        "deletion_auc": float(np.mean(del_aucs)),
        "insertion_auc": float(np.mean(ins_aucs)),
        # A good map: low deletion AUC, high insertion AUC → large positive gap.
        "insertion_minus_deletion": float(np.mean(ins_aucs) - np.mean(del_aucs)),
        "n": int(len(X_sel)),
    }


# ---------------------------------------------------------------------------
# Shift-VR environment
# ---------------------------------------------------------------------------


def load_or_make_shift_test(cfg, out_dir):
    """Return the shifted-environment test split (regenerated deterministically).

    Same SCM (graph/mechanisms) as the base config, ``noise_type='uniform'``.
    Persisted under ``<cfg>_shift`` so re-runs are cheap.
    """
    shift_cfg = shifted_config(cfg, noise_type="uniform")
    dest = Path(out_dir) / shift_cfg.name
    if not (dest / "meta.json").exists():
        generate_and_save(shift_cfg, out_dir=out_dir)
    data = load_dataset(shift_cfg.name, out_dir=out_dir)
    return data["X_test"]


# ---------------------------------------------------------------------------
# NlinearSCM-T smoke path — oracle structural-CF as the in-house "method"
# ---------------------------------------------------------------------------
#
# Real CF methods (Wachter/DiCE/CARLA) on the nonlinear SCM are **out of scope**
# (the collaborator's CF-methods track — Backlog #2; they need a nonlinear torch
# rollout + an LSTM retrained on nonlinear data). To still exercise the CF-faith
# metric path end-to-end on a nonlinear dataset, we use the Stage-4 *oracle
# structural counterfactual* as the in-house "method": a ground-truth CF that is
# faithful by construction. Its two variants are mutually exclusive positive
# controls — the noisy (Pearl) oracle scores ``pearl_hard=1`` and the noiseless
# (skeleton) oracle scores ``rollout_hard=1`` — which doubles as a demonstration
# of the rollout-vs-pearl contrast on nonlinear data, classifier-free.

#: Deterministic intervention magnitude for the oracle CF (> derive tol).
ORACLE_SHIFT = 1.5


def build_oracle_cfs(X_sel, mechanism, noiseless, shift=ORACLE_SHIFT):
    """Build a ``(N, T, k)`` batch of oracle structural counterfactuals.

    For each instance ``i`` we apply the deterministic atomic intervention
    ``do(x[t0, node] = x[t0, node] + shift)`` with ``t0 = T // 2`` and
    ``node = i % k`` (cycled so the batch spans nodes), then roll the known SCM
    forward via :func:`structural_counterfactual`. ``noiseless=False`` reuses the
    abducted factual noise (Pearl-faithful); ``noiseless=True`` rolls the
    deterministic skeleton (rollout-faithful).
    """
    X_sel = np.asarray(X_sel, dtype=float)
    T = X_sel.shape[1]
    k = mechanism.k
    t0 = T // 2
    cfs = []
    for i, x in enumerate(X_sel):
        node = i % k
        value = float(x[t0, node]) + shift
        cfs.append(
            structural_counterfactual(
                x, mechanism, t0, node, value, noiseless=noiseless
            )
        )
    return np.asarray(cfs, dtype=np.float32)


def _score_oracle_batch(X_sel, CFs, graph, mechanism, method_name):
    """Classifier-free batch metrics for an oracle-CF method.

    Mirrors :func:`causaltemp_xai.eval.evaluate_method` minus ``validity``/``ood``
    (no classifier needed — CF-faith is classifier-agnostic, the whole point of
    the oracle positive control). Returns the batch-mean record plus per-instance
    rows for the CSV dump.
    """
    rollout = CFfaith(semantics="noiseless_rollout")
    pearl = CFfaith(semantics="pearl_delta")
    prox_l1, prox_l2, spars = [], [], []
    r_h, r_s, p_h, p_s = [], [], [], []
    rows = []
    for i, (x, x_cf) in enumerate(zip(X_sel, CFs)):
        t = derive_intervention_t(x, x_cf)
        r = rollout.score(x, x_cf, t, graph, mechanism)
        p = pearl.score(x, x_cf, t, graph, mechanism)
        px1 = proximity(x, x_cf, norm="l1")
        px2 = proximity(x, x_cf, norm="l2")
        sp = sparsity(x, x_cf)
        prox_l1.append(px1)
        prox_l2.append(px2)
        spars.append(sp)
        r_h.append(r["hard"])
        r_s.append(r["soft"])
        p_h.append(p["hard"])
        p_s.append(p["soft"])
        rows.append(
            {
                "method": method_name,
                "instance": int(i),
                "validity": None,  # classifier-agnostic oracle run
                "proximity_l1": px1,
                "proximity_l2": px2,
                "sparsity": sp,
                "intervention_t": int(t),
                "cf_faith_rollout_hard": r["hard"],
                "cf_faith_rollout_soft": r["soft"],
                "cf_faith_pearl_hard": p["hard"],
                "cf_faith_pearl_soft": p["soft"],
            }
        )
    sparsity_mean = float(np.mean(spars))
    rec = {
        "method": method_name,
        "n": int(len(CFs)),
        "validity": None,
        "proximity_l1": float(np.mean(prox_l1)),
        "proximity_l2": float(np.mean(prox_l2)),
        "sparsity": sparsity_mean,
        "frac_altered": float(1.0 - sparsity_mean),
        "ood": None,
        "cf_faith_rollout_hard": float(np.mean(r_h)),
        "cf_faith_rollout_soft": float(np.mean(r_s)),
        "cf_faith_pearl_hard": float(np.mean(p_h)),
        "cf_faith_pearl_soft": float(np.mean(p_s)),
    }
    return rec, rows


def _ensure_dataset(config_name, out_dir):
    """Load the named dataset, generating + persisting it first if absent."""
    cfg = get_config(config_name)
    try:
        return load_dataset(cfg.name, out_dir=out_dir)
    except FileNotFoundError:
        print(f"[run_all] dataset for '{cfg.name}' missing — generating …")
        generate_and_save(cfg, out_dir=out_dir)
        return load_dataset(cfg.name, out_dir=out_dir)


def run_nonlinear(config_name, n_cf, out_dir):
    """NlinearSCM-T smoke run: score CF-faith on oracle structural-CFs.

    No LSTM checkpoint and no real CF methods are required (both out of scope —
    see the section banner). Writes the same ``results.json`` / ``per_instance.csv``
    artifacts as :func:`run`, with the ``methods`` block populated by the two
    oracle-CF positive controls.
    """
    cfg = get_config(config_name)
    data = _ensure_dataset(config_name, out_dir)
    X_test = data["X_test"]
    graph, mech = data["graph"], data["mechanism"]

    n = min(n_cf, len(X_test))
    X_sel = X_test[:n]
    print(
        f"[run_all] config={cfg.name} (nonlinear MLP) k={cfg.k} T={cfg.T} | "
        f"scoring CF-faith on {len(X_sel)} oracle structural-CFs"
    )

    oracle_variants = [
        ("OracleCF-Pearl", False),
        ("OracleCF-Rollout", True),
    ]
    summary, all_rows = [], []
    for name, noiseless in oracle_variants:
        print(f"[run_all] building oracle CFs: {name} …")
        cfs = build_oracle_cfs(X_sel, mech, noiseless=noiseless)
        rec, rows = _score_oracle_batch(X_sel, cfs, graph, mech, name)
        summary.append(rec)
        all_rows.extend(rows)

    results = {
        "provenance": {
            "config": cfg.as_dict(),
            "seed": cfg.seed,
            "n_cf": int(len(X_sel)),
            "mechanism_type": cfg.mechanism_type,
            "target_class": TARGET_CLASS,
            "note": (
                "NlinearSCM-T smoke run. Real CF methods (Wachter/DiCE/CARLA) on "
                "the nonlinear SCM are out of scope (collaborator's CF-methods "
                "track — see plan Backlog #2). CF-faith here is exercised via the "
                "Stage-4 oracle structural-CF positive control: the Pearl variant "
                "is pearl_hard=1 and the rollout variant is rollout_hard=1 by "
                "construction. validity/ood are omitted (classifier-agnostic)."
            ),
        },
        "methods": summary,
    }

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    results_path = EXP_DIR / "results.json"
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)

    per_instance_path = EXP_DIR / "per_instance.csv"
    _write_csv(per_instance_path, all_rows)

    print("\n=== NlinearSCM-T oracle CF-faith (batch means) ===")
    hdr = (
        f"{'Method':<18}{'prox_l1':>9}{'spars':>7}"
        f"{'roll_h':>8}{'roll_s':>8}{'pearl_h':>8}{'pearl_s':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in summary:
        print(
            f"{r['method']:<18}{r['proximity_l1']:>9.3f}{r['sparsity']:>7.2f}"
            f"{r['cf_faith_rollout_hard']:>8.2f}{r['cf_faith_rollout_soft']:>8.2f}"
            f"{r['cf_faith_pearl_hard']:>8.2f}{r['cf_faith_pearl_soft']:>8.2f}"
        )
    print(f"\n[run_all] wrote {results_path}")
    print(f"[run_all] wrote {per_instance_path}")
    return results


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


def run(config_name, n_cf, out_dir, use_dice_ml, dice_method="gradient", nl_mode="oracle"):
    cfg = get_config(config_name)
    is_nonlinear = cfg.mechanism_type != "linear"
    if is_nonlinear and nl_mode == "oracle":
        # Default nonlinear path: classifier-free oracle-CF positive control (no
        # checkpoint / no real CF methods — see run_nonlinear's banner).
        # Pass ``--nl-mode real`` to run the real CF pipeline below instead
        # (needs an lstm.pt trained on the nonlinear config).
        return run_nonlinear(config_name, n_cf, out_dir)
    data = load_dataset(cfg.name, out_dir=out_dir)
    X_train = data["X_train"]
    X_test, Y_test = data["X_test"], data["Y_test"]
    graph, mech = data["graph"], data["mechanism"]

    ckpt = Path(out_dir) / cfg.name / "lstm.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"no checkpoint at {ckpt}; train first: "
            f"uv run python -m causaltemp_xai.classifiers.lstm --config {cfg.name} --train"
        )
    clf = LSTMClassifier.load(ckpt)

    accuracies = {
        "train": clf.score(X_train, data["Y_train"]),
        "val": clf.score(data["X_val"], data["Y_val"]),
        "test": clf.score(X_test, Y_test),
    }

    sel = select_flip_candidates(clf, X_test, n_cf)
    X_sel = X_test[sel]
    print(
        f"[run_all] config={cfg.name} k={cfg.k} T={cfg.T} | "
        f"test_acc={accuracies['test']:.3f} | "
        f"selected {len(X_sel)} flip candidates (target={TARGET_CLASS})"
    )

    methods = build_methods(X_train, use_dice_ml=use_dice_ml, dice_method=dice_method)

    summary, all_rows = [], []
    cf_cache = {}
    for name, method in methods.items():
        print(f"[run_all] generating CFs: {name} …")
        cfs = _generate(method, X_sel, clf, graph, mech)
        cf_cache[name] = cfs
        rec = evaluate_method(clf, X_sel, cfs, X_train, graph, mech, TARGET_CLASS)
        rec["method"] = name
        if name == "DiCE":
            rec["dice_backend"] = methods["DiCE"].backend_used
        summary.append(rec)
        all_rows.extend(per_instance_records(clf, X_sel, cfs, graph, mech, name))

    if is_nonlinear:
        # Keep the Stage-4 oracle structural-CFs as ground-truth positive
        # controls alongside the real methods. With a classifier now present we
        # score them through the standard evaluate_method (so they also get
        # validity/OOD), unlike the classifier-free run_nonlinear path.
        for oname, noiseless in (("OracleCF-Pearl", False), ("OracleCF-Rollout", True)):
            print(f"[run_all] building oracle control: {oname} …")
            ocfs = build_oracle_cfs(X_sel, mech, noiseless=noiseless)
            orec = evaluate_method(clf, X_sel, ocfs, X_train, graph, mech, TARGET_CLASS)
            orec["method"] = oname
            summary.append(orec)
            all_rows.extend(per_instance_records(clf, X_sel, ocfs, graph, mech, oname))

    print("[run_all] integrated-gradients attribution foil …")
    attribution = attribution_block(clf, X_sel)

    print("[run_all] shift-VR-lite …")
    X_shift_test = load_or_make_shift_test(cfg, out_dir)
    shift_sel = select_flip_candidates(clf, X_shift_test, n_cf)
    X_shift_sel = X_shift_test[shift_sel]
    shift_methods = {
        "Wachter": methods["Wachter"],
        "DiCE": _DiCESingle(methods["DiCE"]),
        "CARLA": methods["CARLA"],
    }
    shift = shift_vr(clf, shift_methods, X_sel, X_shift_sel, graph, mech, TARGET_CLASS)

    results = {
        "provenance": {
            "config": cfg.as_dict(),
            "seed": cfg.seed,
            "n_cf": int(len(X_sel)),
            "mechanism_type": cfg.mechanism_type,
            "nl_mode": nl_mode if is_nonlinear else None,
            "classifier_accuracy": accuracies,
            "dice_backend": methods["DiCE"].backend_used,
            "dice_method": dice_method,
            "target_class": TARGET_CLASS,
        },
        "methods": summary,
        "attribution": attribution,
        "shift_vr": shift,
    }

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    results_path = EXP_DIR / "results.json"
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)

    per_instance_path = EXP_DIR / "per_instance.csv"
    _write_csv(per_instance_path, all_rows)

    _print_table(summary, attribution, shift)
    print(f"\n[run_all] wrote {results_path}")
    print(f"[run_all] wrote {per_instance_path}")
    return results


def _write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0].keys())
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r[c]) for c in cols))
    path.write_text("\n".join(lines) + "\n")


def _print_table(summary, attribution, shift):
    print("\n=== Axis-C + CF-faith (batch means) ===")
    hdr = (
        f"{'Method':<10}{'valid':>7}{'prox_l1':>9}{'spars':>7}{'ood':>7}"
        f"{'roll_h':>8}{'roll_s':>8}{'pearl_h':>8}{'pearl_s':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in summary:
        print(
            f"{r['method']:<10}{r['validity']:>7.2f}{r['proximity_l1']:>9.3f}"
            f"{r['sparsity']:>7.2f}{r['ood']:>7.2f}"
            f"{r['cf_faith_rollout_hard']:>8.2f}{r['cf_faith_rollout_soft']:>8.2f}"
            f"{r['cf_faith_pearl_hard']:>8.2f}{r['cf_faith_pearl_soft']:>8.2f}"
        )
    print("\n=== Attribution foil (IG) ===")
    print(
        f"  deletion_auc={attribution['deletion_auc']:.3f}  "
        f"insertion_auc={attribution['insertion_auc']:.3f}  "
        f"gap={attribution['insertion_minus_deletion']:+.3f}"
    )
    print("\n=== Shift-VR-lite (validity retention) ===")
    for name, m in shift.items():
        print(
            f"  {name:<10} base={m['validity_base']:.2f} "
            f"shift={m['validity_shift']:.2f} shift_vr={m['shift_vr']:.3f}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description="CausalTemp-XAI end-to-end harness.")
    parser.add_argument(
        "--config",
        default="smoke",
        choices=["smoke", "full", "full_sparse", "smoke_nl", "full_nl"],
    )
    parser.add_argument(
        "--n-cf",
        type=int,
        default=None,
        help="Number of test instances to explain (default: 20 smoke / 100 full).",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--no-dice-ml",
        action="store_true",
        help="Use the from-scratch DPP DiCE fallback instead of the dice-ml backend.",
    )
    parser.add_argument(
        "--dice-method",
        default="gradient",
        choices=["gradient", "random", "genetic", "kdtree"],
        help="dice-ml explainer method (ignored with --no-dice-ml).",
    )
    parser.add_argument(
        "--nl-mode",
        default="oracle",
        choices=["oracle", "real"],
        help=(
            "Nonlinear-config CF path: 'oracle' (default, classifier-free "
            "oracle-CF positive control) or 'real' (train/load an LSTM and run "
            "Wachter/DiCE/CARLA + oracle controls). Ignored for linear configs."
        ),
    )
    args = parser.parse_args(argv)

    n_cf = args.n_cf
    if n_cf is None:
        n_cf = 20 if args.config.startswith("smoke") else 100

    run(
        args.config,
        n_cf,
        args.out_dir,
        use_dice_ml=not args.no_dice_ml,
        dice_method=args.dice_method,
        nl_mode=args.nl_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
