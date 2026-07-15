"""Full-suite experiment driver for CausalTemp-XAI.

Runs the end-to-end pipeline for a list of configs and **archives** each config's
outputs under ``experiments/results_archive/<config>/`` so that successive runs do
not clobber one another (``run_all.py`` / ``figures.py`` both write to fixed
paths). ``experiments/report.py`` reads that archive to build the consolidated
report.

For each config it shells out to the existing module entrypoints (single source
of truth for the pipeline logic), using the *current* interpreter so it inherits
the active ``uv``/venv environment:

    1. generate the dataset            python -m causaltemp_xai.data_io --config X
    2. (linear only) train + freeze    python -m causaltemp_xai.classifiers.lstm --config X --train
    3. run the harness                 python experiments/run_all.py --config X --n-cf N [--dice-method M]
    4. snapshot results.json + per_instance.csv -> results_archive/X/
    5. (linear only) render figures     python experiments/figures.py ... --out-dir results_archive/X/figures

Nonlinear configs (``*_nl``, ``mechanism_type='mlp'``) auto-route inside
``run_all.py`` to the classifier-free oracle-CF path: no LSTM training, no real
CF methods, ``--dice-method`` is irrelevant, and figures are skipped (their
``per_instance.csv`` has ``validity=None`` which fig1 cannot plot — the oracle
CF-faith numbers reach the report via the results table instead).

Usage
-----
    uv run --offline python experiments/run_suite.py                      # full,full_sparse,full_nl
    uv run --offline python experiments/run_suite.py --configs smoke smoke_nl --n-cf 10
    uv run --offline python experiments/run_suite.py --skip-existing      # resume
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaltemp_xai.config import get_config  # noqa: E402

EXP_DIR = ROOT / "experiments"
ARCHIVE_DIR = EXP_DIR / "results_archive"
RESULTS_JSON = EXP_DIR / "results.json"
PER_INSTANCE_CSV = EXP_DIR / "per_instance.csv"

DEFAULT_CONFIGS = ["full", "full_sparse", "full_nl"]


def _run(cmd: list[str]) -> None:
    """Run a subprocess, streaming output; raise on non-zero exit."""
    printable = " ".join(cmd)
    print(f"\n$ {printable}\n", flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(
            f"[run_suite] command failed (exit {result.returncode}): {printable}"
        )


def run_config(config_name: str, n_cf: int, dice_method: str, patience: int,
               nl_mode: str = "oracle") -> Path:
    """Run the full pipeline for one config and archive its outputs.

    Returns the archive directory for this config.
    """
    cfg = get_config(config_name)
    is_linear = cfg.mechanism_type == "linear"
    # A "real-CF" run needs a classifier + real methods + figures. Linear configs
    # always do; nonlinear configs do only when nl_mode == "real" (otherwise they
    # take run_all's classifier-free oracle path).
    run_real = is_linear or nl_mode == "real"
    dest = ARCHIVE_DIR / config_name
    dest.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    print(f"\n{'=' * 72}\n[run_suite] CONFIG={config_name} "
          f"(mechanism={cfg.mechanism_type}, k={cfg.k}, T={cfg.T}, N={cfg.N}, "
          f"real_cf={run_real})\n{'=' * 72}")

    # 1. dataset (deterministic / seeded — safe to regenerate)
    _run([py, "-m", "causaltemp_xai.data_io", "--config", config_name])

    # 2. classifier (any real-CF run — linear, or nonlinear in real mode)
    if run_real:
        _run([
            py, "-m", "causaltemp_xai.classifiers.lstm",
            "--config", config_name, "--train", "--patience", str(patience),
        ])

    # 3. harness
    harness_cmd = [
        py, str(EXP_DIR / "run_all.py"),
        "--config", config_name, "--n-cf", str(n_cf),
    ]
    if run_real:
        harness_cmd += ["--dice-method", dice_method]
    if not is_linear:
        harness_cmd += ["--nl-mode", nl_mode]
    _run(harness_cmd)

    # 4. snapshot the fixed-path outputs into the per-config archive
    if RESULTS_JSON.exists():
        shutil.copy2(RESULTS_JSON, dest / "results.json")
    else:
        raise SystemExit(f"[run_suite] expected {RESULTS_JSON} after run_all; missing")
    if PER_INSTANCE_CSV.exists():
        shutil.copy2(PER_INSTANCE_CSV, dest / "per_instance.csv")

    # 5. figures (any real-CF run — per_instance has real 0/1 validity, which
    # fig1 needs; the classifier-free oracle path leaves validity=None and is
    # skipped)
    if run_real:
        fig_dir = dest / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        _run([
            py, str(EXP_DIR / "figures.py"),
            "--results", str(dest / "results.json"),
            "--per-instance", str(dest / "per_instance.csv"),
            "--out-dir", str(fig_dir),
        ])

    print(f"[run_suite] archived {config_name} -> {dest}")
    return dest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the CausalTemp-XAI experiment suite.")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                        help=f"Configs to run (default: {DEFAULT_CONFIGS}).")
    parser.add_argument("--n-cf", type=int, default=100,
                        help="Test instances to explain per config (default: 100).")
    parser.add_argument("--dice-method", default="random",
                        choices=["gradient", "random", "genetic", "kdtree"],
                        help="dice-ml explainer method for linear configs (default: random).")
    parser.add_argument("--patience", type=int, default=20,
                        help="LSTM early-stopping patience (any real-CF run).")
    parser.add_argument("--nl-mode", default="oracle", choices=["oracle", "real"],
                        help="Nonlinear-config CF path forwarded to run_all.py: "
                             "'oracle' (default, classifier-free control) or "
                             "'real' (train LSTM + run real CF methods).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip configs that already have an archived results.json.")
    args = parser.parse_args(argv)

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    done = []
    for name in args.configs:
        get_config(name)  # validate name up front
        if args.skip_existing and (ARCHIVE_DIR / name / "results.json").exists():
            print(f"[run_suite] --skip-existing: {name} already archived, skipping")
            done.append(name)
            continue
        run_config(name, n_cf=args.n_cf, dice_method=args.dice_method,
                   patience=args.patience, nl_mode=args.nl_mode)
        done.append(name)

    print(f"\n[run_suite] DONE. Archived configs: {', '.join(done)}")
    print(f"[run_suite] next: uv run --offline python experiments/report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
