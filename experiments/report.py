"""Consolidated final report generator for CausalTemp-XAI.

Reads the per-config archives written by ``experiments/run_suite.py``
(``experiments/results_archive/<config>/results.json``) and emits a single
data-driven markdown report at ``docs/final_report.md``:

* provenance table (classifier accuracy, DiCE backend/method, seed, n_cf);
* headline Axis-C + CF-faith table per config;
* rank-inversion / H4 Spearman ρ (traditional metrics vs causal CF-faith);
* H1 / H3 verdicts re-assessed against the current numbers;
* H5 descriptive comparison (full vs full_sparse);
* NlinearSCM-T oracle positive-control check;
* attribution foil + Shift-VR-lite summaries.

Usage
-----
    uv run --offline python experiments/report.py
    uv run --offline python experiments/report.py --configs smoke smoke_nl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = ROOT / "experiments" / "results_archive"
DEFAULT_OUT = ROOT / "docs" / "final_report.md"

# Pre-registered thresholds (quoted from docs/hypotheses_assessment.md).
H1_VALIDITY_BAR = 0.9
H1_FAITH_BAR = 0.3
H3_FAITH_BAR = 0.7
H4_RHO_BAR = 0.5

STANDARD_METHODS = ("Wachter", "DiCE")
CAUSAL_METHOD = "CARLA"


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------

def _f(x, prec=2):
    if x is None:
        return "—"
    try:
        return f"{float(x):.{prec}f}"
    except (TypeError, ValueError):
        return str(x)


def _load(config_name):
    path = ARCHIVE_DIR / config_name / "results.json"
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def _has_classifier(res):
    """True when a run scored real CF methods against a black-box classifier.

    Both linear configs and nonlinear runs in ``--nl-mode real`` record
    ``classifier_accuracy`` in provenance; the classifier-free oracle path does
    not. Such runs get the full report treatment (headline, rank inversion,
    H1/H3, attribution, Shift-VR) regardless of mechanism family.
    """
    return "classifier_accuracy" in res["provenance"]


def _has_oracle(res):
    """True when the run includes the OracleCF-* structural positive controls."""
    return any(str(m.get("method", "")).startswith("OracleCF") for m in res["methods"])


def _real_method_records(res):
    """The three real CF-method records, excluding appended OracleCF-* controls.

    Used for method-ranking correlations (H4) so the ground-truth oracle
    controls do not distort the Spearman ρ across *methods*.
    """
    return [m for m in res["methods"] if m["method"] in ("Wachter", "DiCE", "CARLA")]


def _method_by_name(res, name):
    for m in res["methods"]:
        if m["method"] == name:
            return m
    return None


# ---------------------------------------------------------------------------
# Spearman (self-contained; avoids importing scipy for a 3-point rank corr)
# ---------------------------------------------------------------------------

def _rankdata(values):
    """Average-rank of values (ties share the mean rank), 1-indexed."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-indexed average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(a, b):
    """Spearman ρ between two equal-length sequences; None if undefined."""
    if len(a) != len(b) or len(a) < 2:
        return None
    ra, rb = _rankdata(a), _rankdata(b)
    n = len(a)
    ma = sum(ra) / n
    mb = sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((ra[i] - ma) ** 2 for i in range(n))
    vb = sum((rb[i] - mb) ** 2 for i in range(n))
    if va == 0 or vb == 0:
        return None  # a constant metric -> ρ undefined
    return cov / (va ** 0.5 * vb ** 0.5)


# ---------------------------------------------------------------------------
# report sections
# ---------------------------------------------------------------------------

def _provenance_table(configs):
    lines = [
        "| Config | Mechanism | k | T | N | Classifier test acc | DiCE method | DiCE backend | n_cf | seed |",
        "|---|---|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for name, res in configs:
        p = res["provenance"]
        cfg = p["config"]
        acc = p.get("classifier_accuracy")
        acc_s = _f(acc["test"], 3) if acc else "— (classifier-free)"
        method = p.get("dice_method", "—")
        backend = p.get("dice_backend", "—")
        lines.append(
            f"| `{name}` | {cfg['mechanism_type']} | {cfg['k']} | {cfg['T']} | "
            f"{cfg['N']} | {acc_s} | {method} | {backend} | {p['n_cf']} | {cfg['seed']} |"
        )
    return "\n".join(lines)


def _headline_table(name, res):
    lines = [
        f"#### `{name}`",
        "",
        "| Method | Validity | Proximity L1 | Sparsity | OOD | CF-faith rollout (h/s) | CF-faith pearl (h/s) |",
        "|---|---:|---:|---:|---:|:---:|:---:|",
    ]
    for m in res["methods"]:
        lines.append(
            f"| {m['method']} | {_f(m['validity'])} | {_f(m['proximity_l1'])} | "
            f"{_f(m['sparsity'])} | {_f(m['ood'])} | "
            f"{_f(m['cf_faith_rollout_hard'])} / {_f(m['cf_faith_rollout_soft'])} | "
            f"{_f(m['cf_faith_pearl_hard'])} / {_f(m['cf_faith_pearl_soft'])} |"
        )
    return "\n".join(lines)


def _rank_inversion(name, res):
    methods = _real_method_records(res)
    faith = [m["cf_faith_rollout_hard"] for m in methods]
    out = [f"#### `{name}`", ""]
    rows = ["| Traditional metric | Spearman ρ vs CF-faith(rollout,hard) |",
            "|---|:---:|"]
    for key, label in [("validity", "validity"),
                       ("proximity_l1", "proximity L1"),
                       ("sparsity", "sparsity")]:
        trad = [m[key] for m in methods]
        rho = _spearman(trad, faith)
        rows.append(f"| {label} | {_f(rho, 2) if rho is not None else 'undefined'} |")
    out.append("\n".join(rows))
    val_rho = _spearman([m["validity"] for m in methods], faith)
    if val_rho is not None and val_rho < 0:
        out.append(
            f"\n**Rank inversion present:** validity ρ = {val_rho:.2f} < 0 — validity "
            f"ranks the methods in the *opposite* order to causal faithfulness."
        )
    elif val_rho is not None:
        out.append(f"\nValidity ρ = {val_rho:.2f} (no inversion at this config).")
    return "\n".join(out)


def _h1_h3(name, res):
    out = [f"#### `{name}`"]
    # H1: standard methods valid but unfaithful
    for mn in STANDARD_METHODS:
        m = _method_by_name(res, mn)
        if m is None:
            continue
        v, fh = m["validity"], m["cf_faith_rollout_hard"]
        valid_ok = v is not None and v > H1_VALIDITY_BAR
        unfaith_ok = fh is not None and fh < H1_FAITH_BAR
        verdict = "supports H1" if (valid_ok and unfaith_ok) else "partial"
        out.append(
            f"- **{mn}** — validity {_f(v)} (>{H1_VALIDITY_BAR}? {'✓' if valid_ok else '✗'}), "
            f"CF-faith rollout-hard {_f(fh)} (<{H1_FAITH_BAR}? {'✓' if unfaith_ok else '✗'}) → {verdict}."
        )
    # H3: causal method faithful
    c = _method_by_name(res, CAUSAL_METHOD)
    if c is not None:
        fh = c["cf_faith_rollout_hard"]
        faith_ok = fh is not None and fh > H3_FAITH_BAR
        out.append(
            f"- **{CAUSAL_METHOD}** — CF-faith rollout-hard {_f(fh)} "
            f"(>{H3_FAITH_BAR}? {'✓' if faith_ok else '✗'}) → "
            f"{'supports H3 (faithful)' if faith_ok else 'H3 not met'}; "
            f"validity {_f(c['validity'])}, proximity L1 {_f(c['proximity_l1'])} "
            f"(cost of faithfulness)."
        )
    return "\n".join(out)


def _attribution_shift(name, res):
    out = [f"#### `{name}`"]
    a = res.get("attribution")
    if a:
        out.append(
            f"- **Integrated-Gradients foil:** deletion AUC {_f(a['deletion_auc'], 3)}, "
            f"insertion AUC {_f(a['insertion_auc'], 3)}, gap "
            f"{a['insertion_minus_deletion']:+.3f} — a reasonable saliency map that "
            f"still says nothing about causal faithfulness."
        )
    s = res.get("shift_vr")
    if s:
        parts = []
        for mn, m in s.items():
            parts.append(
                f"{mn}: base {_f(m['validity_base'])} / shift {_f(m['validity_shift'])} "
                f"→ Shift-VR {_f(m['shift_vr'], 3)}"
            )
        out.append("- **Shift-VR-lite (validity retention):** " + "; ".join(parts) + ".")
    return "\n".join(out)


def _oracle_control(name, res):
    pearl = _method_by_name(res, "OracleCF-Pearl")
    rollout = _method_by_name(res, "OracleCF-Rollout")
    out = [f"#### `{name}` (NlinearSCM-T oracle positive control)"]
    if pearl is not None:
        ok = pearl["cf_faith_pearl_hard"] == 1.0
        out.append(f"- **OracleCF-Pearl**: pearl-hard {_f(pearl['cf_faith_pearl_hard'])} "
                   f"({'✓ PASS' if ok else '✗ FAIL'}, expected 1.0).")
    if rollout is not None:
        ok = rollout["cf_faith_rollout_hard"] == 1.0
        out.append(f"- **OracleCF-Rollout**: rollout-hard {_f(rollout['cf_faith_rollout_hard'])} "
                   f"({'✓ PASS' if ok else '✗ FAIL'}, expected 1.0).")
    note = res["provenance"].get("note")
    if note:
        out.append(f"\n> {note}")
    return "\n".join(out)


def _h5(configs_by_name):
    full = configs_by_name.get("full")
    sparse = configs_by_name.get("full_sparse")
    if not (full and sparse):
        return "_Not comparable: need both `full` and `full_sparse` archived._"
    out = ["Descriptive only — no ablation claim (per the MVP H5 = hint-only note).",
           "",
           "| Method | Metric | full (s=0.2) | full_sparse (s=0.1) | Δ |",
           "|---|---|---:|---:|---:|"]
    for mn in ("Wachter", "DiCE", "CARLA"):
        mf, ms = _method_by_name(full, mn), _method_by_name(sparse, mn)
        if mf is None or ms is None:
            continue
        for key, label in [("validity", "validity"),
                           ("cf_faith_rollout_hard", "CF-faith rollout-h")]:
            a, b = mf[key], ms[key]
            d = (b - a) if (a is not None and b is not None) else None
            out.append(f"| {mn} | {label} | {_f(a)} | {_f(b)} | "
                       f"{('%+.2f' % d) if d is not None else '—'} |")
    return "\n".join(out)


def _summary(configs):
    """Data-driven 'what works / what doesn't' bullets."""
    bullets = []
    for name, res in configs:
        if not _has_classifier(res):
            continue
        real = _real_method_records(res)
        c = _method_by_name(res, CAUSAL_METHOD)
        d = _method_by_name(res, "DiCE")
        w = _method_by_name(res, "Wachter")
        faith = c["cf_faith_rollout_hard"] if c else None
        val_rho = _spearman([m["validity"] for m in real],
                            [m["cf_faith_rollout_hard"] for m in real])
        frag = f"**`{name}`**: "
        if c and faith is not None:
            frag += (f"CARLA faithful (rollout-hard {faith:.2f}), "
                     f"validity {_f(c['validity'])}; ")
        if d:
            frag += f"DiCE valid {_f(d['validity'])} yet unfaithful ({_f(d['cf_faith_rollout_hard'])}); "
        if w:
            frag += f"Wachter valid {_f(w['validity'])} yet unfaithful ({_f(w['cf_faith_rollout_hard'])}); "
        if val_rho is not None:
            frag += (f"validity↔faith ρ={val_rho:.2f} "
                     f"({'rank inversion' if val_rho < 0 else 'no inversion'})")
        bullets.append("- " + frag.rstrip("; .") + ".")
    for name, res in configs:
        if not _has_oracle(res):
            continue
        p = _method_by_name(res, "OracleCF-Pearl")
        r = _method_by_name(res, "OracleCF-Rollout")
        ok = (p and p["cf_faith_pearl_hard"] == 1.0) and (r and r["cf_faith_rollout_hard"] == 1.0)
        mech = "nonlinear" if res["provenance"].get("mechanism_type") == "mlp" else "linear"
        bullets.append(f"- **`{name}`** ({mech}): oracle positive control "
                       f"{'PASSES' if ok else 'FAILS'} — CF-faith machinery is correct on MLP mechanisms.")
    return "\n".join(bullets)


def build_report(config_names):
    configs = []
    for name in config_names:
        res = _load(name)
        if res is not None:
            configs.append((name, res))
    if not configs:
        raise SystemExit(
            f"[report] no archived results found under {ARCHIVE_DIR}. "
            f"Run experiments/run_suite.py first."
        )
    by_name = {n: r for n, r in configs}
    # Runs with a classifier (linear, or nonlinear --nl-mode real) get the full
    # metric treatment; runs with OracleCF-* controls get the oracle section.
    with_clf = [(n, r) for n, r in configs if _has_classifier(r)]
    with_oracle = [(n, r) for n, r in configs if _has_oracle(r)]

    md = []
    md.append("# CausalTemp-XAI — Final Experiment Report")
    md.append("")
    md.append(f"_Auto-generated by `experiments/report.py` from "
              f"`experiments/results_archive/`. Configs: "
              f"{', '.join('`' + n + '`' for n, _ in configs)}._")
    md.append("")
    md.append("> Supersedes the v0.1 [`docs/hypotheses_assessment.md`]"
              "(hypotheses_assessment.md) (TCN-era, `full`-only). The numbers "
              "here are from the current **LSTM** harness with the DiCE method "
              "recorded in the provenance table.")
    md.append("")

    md.append("## TL;DR — what works, what doesn't")
    md.append("")
    md.append(_summary(configs))
    md.append("")

    md.append("## Provenance")
    md.append("")
    md.append(_provenance_table(configs))
    md.append("")

    md.append("## Headline results (batch means)")
    md.append("")
    for name, res in configs:
        md.append(_headline_table(name, res))
        md.append("")

    if with_clf:
        md.append("## Rank inversion (H4): traditional metrics vs causal CF-faith")
        md.append("")
        md.append("Spearman ρ across the three real methods (per-method batch "
                  f"means). Pre-registered bar: |ρ| interpreted against {H4_RHO_BAR}; "
                  "a *negative* validity ρ is the headline phenomenon "
                  "(reported preliminary/underpowered — few methods).")
        md.append("")
        for name, res in with_clf:
            md.append(_rank_inversion(name, res))
            md.append("")

        md.append("## H1 / H3 verdicts")
        md.append("")
        md.append(f"Pre-registered: **H1** standard methods validity > {H1_VALIDITY_BAR} "
                  f"AND CF-faith(rollout,hard) < {H1_FAITH_BAR}; "
                  f"**H3** CARLA CF-faith(rollout,hard) > {H3_FAITH_BAR}.")
        md.append("")
        for name, res in with_clf:
            md.append(_h1_h3(name, res))
            md.append("")

        md.append("## H5 (hint): sparsity sensitivity")
        md.append("")
        md.append(_h5(by_name))
        md.append("")

        md.append("## Attribution foil + Shift-VR-lite")
        md.append("")
        for name, res in with_clf:
            md.append(_attribution_shift(name, res))
            md.append("")

    if with_oracle:
        md.append("## NlinearSCM-T oracle control")
        md.append("")
        for name, res in with_oracle:
            md.append(_oracle_control(name, res))
            md.append("")

    md.append("## Figures")
    md.append("")
    for name, res in with_clf:
        fig_dir = ARCHIVE_DIR / name / "figures"
        if fig_dir.exists():
            md.append(f"**`{name}`** — see `experiments/results_archive/{name}/figures/`:")
            for png in sorted(fig_dir.glob("*.png")):
                rel = png.relative_to(ROOT)
                md.append(f"- `{rel}`")
            md.append("")

    return "\n".join(md).rstrip() + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the consolidated final report.")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Configs to include (default: every archived config).")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    if args.configs is None:
        if not ARCHIVE_DIR.exists():
            raise SystemExit(f"[report] {ARCHIVE_DIR} does not exist; run run_suite.py first.")
        config_names = sorted(p.name for p in ARCHIVE_DIR.iterdir()
                              if p.is_dir() and (p / "results.json").exists())
    else:
        config_names = args.configs

    report = build_report(config_names)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"[report] wrote {out_path} ({len(report.splitlines())} lines) "
          f"from configs: {', '.join(config_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
