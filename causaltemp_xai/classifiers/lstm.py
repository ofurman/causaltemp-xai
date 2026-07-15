"""Long Short-Term Memory (LSTM) classifier for time-series classification.

Stacked bidirectional LSTM with dropout between layers.  The last timestep's
hidden state is fed to a linear classification head.  Accepts variable-length
sequences via ``batch_first=True`` PyTorch LSTM.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Core module
# ---------------------------------------------------------------------------


class LSTM(nn.Module):
    """Stacked (optionally bidirectional) LSTM classifier.

    Parameters
    ----------
    n_inputs:
        Number of input features per time step (``k``).
    n_classes:
        Number of output classes.
    hidden_size:
        Number of features in the LSTM hidden state.
    num_layers:
        Number of stacked LSTM layers.
    dropout:
        Dropout probability applied between LSTM layers (ignored when
        ``num_layers == 1``).
    bidirectional:
        If ``True``, use a bidirectional LSTM; the forward and backward hidden
        states are concatenated before the classification head.
    """

    def __init__(
        self,
        n_inputs: int,
        n_classes: int = 2,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=n_inputs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        head_in = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(head_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape ``(N, T, k)`` — batch-first, time second.

        Returns
        -------
        logits : Tensor of shape ``(N, n_classes)``.
        """
        # out: (N, T, D*hidden_size); hn: (D*num_layers, N, hidden_size)
        _, (hn, _) = self.lstm(x)
        # hn[-1] is the last layer's forward hidden state;
        # for bidirectional, concatenate the last forward and backward states.
        if self.bidirectional:
            # hn shape: (num_layers*2, N, hidden_size) — last forward is [-2], backward is [-1]
            last_hidden = torch.cat([hn[-2], hn[-1]], dim=-1)  # (N, 2*hidden_size)
        else:
            last_hidden = hn[-1]  # (N, hidden_size)
        return self.head(last_hidden)


# ---------------------------------------------------------------------------
# Convenience training function
# ---------------------------------------------------------------------------


def train_lstm(
    dataset: tuple[np.ndarray, np.ndarray],
    n_classes: int = 2,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    bidirectional: bool = False,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 100,
    target_acc: float = 0.90,
    device: Optional[str] = None,
    verbose: bool = True,
) -> LSTM:
    """Train an LSTM classifier until *target_acc* is reached or *max_epochs* elapse.

    Parameters
    ----------
    dataset:
        Tuple ``(X, y)`` where ``X`` has shape ``(N, T, k)`` and ``y`` has
        shape ``(N,)`` with integer class labels.
    n_classes:
        Number of output classes.
    hidden_size:
        LSTM hidden size.
    num_layers:
        Number of stacked LSTM layers.
    dropout:
        Dropout rate between layers.
    bidirectional:
        Use a bidirectional LSTM.
    lr:
        Learning rate for Adam.
    batch_size:
        Mini-batch size.
    max_epochs:
        Hard ceiling on training epochs.
    target_acc:
        Training stops early once training accuracy reaches this threshold.
    device:
        Torch device string.  Auto-detected if ``None``.
    verbose:
        Print epoch summary when ``True``.

    Returns
    -------
    LSTM
        Trained model in eval mode.
    """
    X_np, y_np = dataset
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    k = X_np.shape[2] if X_np.ndim == 3 else X_np.shape[1]
    model = LSTM(
        n_inputs=k,
        n_classes=n_classes,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    ).to(dev)

    X_t = torch.tensor(X_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)  # (N, T, k) — no permute needed
            optimiser.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)

        acc = correct / total
        if verbose:
            print(f"Epoch {epoch:03d}/{max_epochs} | loss={total_loss/total:.4f} | acc={acc:.4f}")
        if acc >= target_acc:
            if verbose:
                print(f"  → target accuracy {target_acc:.0%} reached, stopping early.")
            break

    model.eval()
    return model


# ---------------------------------------------------------------------------
# sklearn-style wrapper
# ---------------------------------------------------------------------------


class LSTMClassifier:
    """sklearn-style wrapper around :class:`LSTM` for the benchmark.

    **Shape contract.** Every public method accepts data in ``(T, k)`` (single
    instance) or ``(N, T, k)`` (batch) layout — the same convention used by the
    generator and :class:`~causaltemp_xai.metrics.cf_faith.CFfaith`.

    Parameters
    ----------
    n_inputs:
        Number of variables ``k`` (input features per time step).
    n_classes, hidden_size, num_layers, dropout, bidirectional:
        Forwarded to :class:`LSTM`.
    lr, batch_size, max_epochs:
        Adam optimiser / training-loop hyperparameters.
    patience:
        Early-stopping patience (epochs without monitor-metric improvement).
        The monitor is val loss when a validation set is supplied to
        :meth:`fit`, otherwise train loss.
    target_acc:
        If train accuracy reaches this, training stops early regardless of the
        early-stopping monitor.
    device:
        Torch device string; auto-detected when ``None``.
    seed:
        Seeds torch for reproducible initialisation/training.
    """

    def __init__(
        self,
        n_inputs: int,
        n_classes: int = 2,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        lr: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 10,
        target_acc: float = 0.90,
        device: Optional[str] = None,
        seed: int = 0,
    ) -> None:
        self.hparams = dict(
            n_inputs=n_inputs,
            n_classes=n_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.target_acc = target_acc
        self.seed = seed
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        torch.manual_seed(seed)
        self.model = LSTM(**self.hparams).to(self.device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _as_batch_tensor(self, X) -> tuple[torch.Tensor, bool]:
        """Return ``(N, T, k)`` float tensor and was_single flag."""
        if isinstance(X, torch.Tensor):
            x = X.to(dtype=torch.float32, device=self.device)
        else:
            x = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        single = x.dim() == 2
        if single:
            x = x.unsqueeze(0)  # (1, T, k)
        return x, single  # (N, T, k)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose: bool = False):
        """Train the underlying LSTM with early stopping; restore best weights.

        ``X_train`` / ``X_val`` are ``(N, T, k)``; labels are ``(N,)``.
        """
        torch.manual_seed(self.seed)
        Xtr = torch.as_tensor(np.asarray(X_train), dtype=torch.float32)
        ytr = torch.as_tensor(np.asarray(y_train), dtype=torch.long)
        loader = DataLoader(
            TensorDataset(Xtr, ytr), batch_size=self.batch_size, shuffle=True
        )
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        has_val = X_val is not None and y_val is not None
        if has_val:
            Xva, _ = self._as_batch_tensor(X_val)
            yva = torch.as_tensor(np.asarray(y_val), dtype=torch.long, device=self.device)

        best_monitor = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        epochs_no_improve = 0

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)  # (N, T, k)
                optimiser.zero_grad()
                logits = self.model(xb)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                optimiser.step()
                total_loss += loss.item() * len(yb)
                correct += (logits.argmax(1) == yb).sum().item()
                total += len(yb)
            train_acc = correct / total
            train_loss = total_loss / total

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(Xva)
                    monitor = F.cross_entropy(val_logits, yva).item()
            else:
                monitor = train_loss

            if verbose:
                tag = "val_loss" if has_val else "train_loss"
                print(
                    f"Epoch {epoch:03d}/{self.max_epochs} | "
                    f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
                    f"{tag}={monitor:.4f}"
                )

            if monitor < best_monitor - 1e-5:
                best_monitor = monitor
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if train_acc >= self.target_acc and not has_val:
                best_state = copy.deepcopy(self.model.state_dict())
                break
            if epochs_no_improve >= self.patience:
                if verbose:
                    print(f"  -> early stopping at epoch {epoch} (patience {self.patience}).")
                break

        self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X) -> np.ndarray:
        """Return class probabilities, shape ``(N, n_classes)`` (softmax)."""
        x, _ = self._as_batch_tensor(X)
        self.model.eval()
        with torch.no_grad():
            probs = F.softmax(self.model(x), dim=-1)
        return probs.cpu().numpy()

    def predict(self, X) -> np.ndarray:
        """Return predicted integer labels.

        ``(N, T, k)`` → ``(N,)``; a single ``(T, k)`` instance → scalar ``int``.
        """
        single = np.asarray(X).ndim == 2 if not isinstance(X, torch.Tensor) else X.dim() == 2
        proba = self.predict_proba(X)
        labels = proba.argmax(axis=-1)
        return int(labels[0]) if single else labels

    def score(self, X, y) -> float:
        """Mean accuracy on ``(X, y)``."""
        preds = self.predict(X)
        return float((np.asarray(preds) == np.asarray(y)).mean())

    def torch_logits(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """Differentiable forward for gradient-based CF methods.

        Parameters
        ----------
        x_tensor:
            A **torch tensor** in ``(T, k)`` or ``(N, T, k)`` layout (the public
            convention). Create with ``requires_grad_()`` and backprop through
            the returned logits for gradient-based counterfactual methods.

        Returns
        -------
        logits : Tensor of shape ``(N, n_classes)`` (always batched).
        """
        x = x_tensor
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        self.model.eval()
        return self.model(x)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Save state_dict + hyperparameters to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "hparams": self.hparams,
                "train_cfg": {
                    "lr": self.lr,
                    "batch_size": self.batch_size,
                    "max_epochs": self.max_epochs,
                    "patience": self.patience,
                    "target_acc": self.target_acc,
                    "seed": self.seed,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str, device: Optional[str] = None) -> "LSTMClassifier":
        """Reconstruct a classifier from a checkpoint written by :meth:`save`."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        train_cfg = ckpt.get("train_cfg", {})
        clf = cls(device=device, **ckpt["hparams"], **train_cfg)
        clf.model.load_state_dict(ckpt["state_dict"])
        clf.model.eval()
        return clf


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _train_cli(argv: list[str] | None = None) -> None:
    import argparse

    from causaltemp_xai.config import get_config
    from causaltemp_xai.data_io import DEFAULT_OUT_DIR, generate_and_save, load_dataset

    parser = argparse.ArgumentParser(description="Train an LSTMClassifier on a locked config.")
    parser.add_argument(
        "--config",
        choices=["smoke", "full", "full_sparse", "smoke_nl", "full_nl"],
        required=True,
    )
    parser.add_argument("--train", action="store_true", help="Run training (required).")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    cfg = get_config(args.config)
    out_dir = Path(args.out_dir)
    if not (out_dir / cfg.name / "meta.json").exists():
        print(f"Dataset for '{cfg.name}' not found — generating (deterministic).")
        generate_and_save(cfg, out_dir=out_dir)
    data = load_dataset(cfg.name, out_dir=out_dir)

    clf = LSTMClassifier(
        n_inputs=cfg.k,
        seed=cfg.seed,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    clf.fit(
        data["X_train"], data["Y_train"], data["X_val"], data["Y_val"], verbose=args.verbose
    )

    train_acc = clf.score(data["X_train"], data["Y_train"])
    val_acc = clf.score(data["X_val"], data["Y_val"])
    test_acc = clf.score(data["X_test"], data["Y_test"])
    print(
        f"[{cfg.name}] train_acc={train_acc:.4f} | "
        f"val_acc={val_acc:.4f} | test_acc={test_acc:.4f}"
    )

    ckpt_path = out_dir / cfg.name / "lstm.pt"
    clf.save(ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
    if test_acc < 0.90:
        print(
            f"WARNING: test accuracy {test_acc:.4f} < 0.90 target "
            f"(>=0.85 is the documented fallback)."
        )


if __name__ == "__main__":
    _train_cli()
