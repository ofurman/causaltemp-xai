"""DiCE: Diverse Counterfactual Explanations (Mothilal et al., 2020) — stub.

Reference
---------
Mothilal, R. K., Sharma, A., & Tan, C. (2020).
*Explaining machine learning classifiers through diverse counterfactual
explanations.*  Proceedings of the 2020 ACM FAccT.

DiCE generates a *set* of diverse counterfactuals by jointly optimising:

    L = proximity_loss + diversity_loss + prediction_loss

* **proximity_loss** – L2 distance from each CF to the original.
* **diversity_loss** – determinantal point process (DPP) term that
  encourages the CFs to be spread out in feature space.
* **prediction_loss** – cross-entropy between the CF prediction and the
  target class.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiCECF:
    """Stub: DiCE diverse counterfactual generator.

    The real implementation should jointly optimise a set of *n_cfs*
    counterfactuals by minimising:

        L = sum_i pred_loss(cf_i) + lam_prox * sum_i ||cf_i - x||^2
              - lam_div * log det K

    where ``K`` is the pairwise-similarity kernel matrix between CFs
    (determinantal point process diversity term).

    Parameters
    ----------
    target_class : int
        Desired output class.
    n_cfs : int
        Number of diverse CFs to generate per instance.
    lam_prox : float
        Proximity regularisation weight.
    lam_div : float
        DPP diversity weight.
    lr : float
        Adam learning rate.
    n_steps : int
        Optimisation iterations.
    """

    def __init__(
        self,
        target_class: int = 1,
        n_cfs: int = 5,
        lam_prox: float = 0.5,
        lam_div: float = 1.0,
        lr: float = 0.05,
        n_steps: int = 500,
        background_data: Optional[np.ndarray] = None,
        use_dice_ml: bool = True,
        method: str = "gradient",
    ) -> None:
        self.target_class = target_class
        self.n_cfs = n_cfs
        self.lam_prox = lam_prox
        self.lam_div = lam_div
        self.lr = lr
        self.n_steps = n_steps
        #: dice-ml explainer method ("gradient" | "random" | "genetic" | "kdtree").
        self.method = method
        # dice-ml needs a background dataset (for feature ranges/MADs); the
        # uniform generate(x, model) interface does not supply one, so it is
        # passed here. When absent, the from-scratch DPP fallback is used.
        self.background_data = (
            np.asarray(background_data, dtype=np.float32)
            if background_data is not None
            else None
        )
        self.use_dice_ml = use_dice_ml
        #: Set to "dice-ml" or "fallback" after the first generate() call.
        self.backend_used: Optional[str] = None

    def generate(self, x: np.ndarray, model) -> np.ndarray:
        """Generate *n_cfs* diverse counterfactuals for a single instance.

        Parameters
        ----------
        x : ndarray of shape ``(T, k)``
            Original time-series instance (public ``(T, k)`` layout).
        model : LSTMClassifier
            Differentiable classifier exposing ``torch_logits``.

        Returns
        -------
        cfs : ndarray of shape ``(n_cfs, T, k)``.

        Notes
        -----
        Uses the official ``dice-ml`` PyTorch gradient backend when a
        ``background_data`` set was supplied and the library path succeeds;
        otherwise falls back to a from-scratch DPP-diversity optimiser (see the
        plan's decision gate / MVP risk table).
        """
        if self.use_dice_ml and self.background_data is not None:
            try:
                cfs = self._generate_dice_ml(x, model)
                self.backend_used = "dice-ml"
                return cfs
            except Exception as exc:  # noqa: BLE001 — one focused attempt, then fall back
                self._last_dice_error = repr(exc)
        self.backend_used = "fallback"
        return self._generate_fallback(x, model)

    def generate_batch(self, X: np.ndarray, model) -> np.ndarray:
        """Generate ``n_cfs`` CFs per instance → shape ``(N, n_cfs, T, k)``."""
        X = np.asarray(X, dtype=np.float32)
        return np.stack([self.generate(x, model) for x in X], axis=0)

    # ------------------------------------------------------------------
    # From-scratch DPP-diversity optimiser (controlled fallback)
    # ------------------------------------------------------------------

    def _generate_fallback(self, x: np.ndarray, model) -> np.ndarray:
        """Joint DPP-diversity CF search over ``n_cfs`` candidates.

        Minimises ``CE(f(cf_i), target) + lam_prox * ||cf_i - x||^2`` summed over
        candidates, minus ``lam_div * logdet(K)`` with
        ``K[i,j] = 1 / (1 + ||cf_i - cf_j||^2)`` (DPP diversity).
        """
        x_arr = np.asarray(x, dtype=np.float32)
        T, k = x_arr.shape
        x_t = torch.as_tensor(x_arr)
        target = torch.full((self.n_cfs,), self.target_class, dtype=torch.long)

        gen = torch.Generator().manual_seed(0)
        jitter = 0.05 * torch.randn(self.n_cfs, T, k, generator=gen)
        cfs = (x_t.unsqueeze(0) + jitter).clone().detach().requires_grad_(True)
        optimiser = torch.optim.Adam([cfs], lr=self.lr)

        for _step in range(self.n_steps):
            optimiser.zero_grad()
            logits = model.torch_logits(cfs)  # (n_cfs, n_classes)
            pred_loss = F.cross_entropy(logits, target)
            prox = ((cfs - x_t.unsqueeze(0)) ** 2).flatten(1).sum(dim=1).mean()

            flat = cfs.flatten(1)  # (n_cfs, T*k)
            sq = torch.cdist(flat, flat) ** 2
            K = 1.0 / (1.0 + sq) + 1e-4 * torch.eye(self.n_cfs)
            div = -torch.logdet(K)

            loss = pred_loss + self.lam_prox * prox + self.lam_div * div
            loss.backward()
            optimiser.step()

        return cfs.detach().cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Official dice-ml gradient backend
    # ------------------------------------------------------------------

    def _generate_dice_ml(self, x: np.ndarray, model) -> np.ndarray:
        """Generate CFs via dice-ml's PyTorch gradient method.

        Builds a flatten/reshape adapter ``(N, T*k) -> P(class=target)`` around
        ``LSTMClassifier.torch_logits`` and a ``dice_ml.Data`` from the flattened
        background set, then runs the configured ``self.method`` (default
        ``"gradient"``; ``"random"`` for fast sampling-based search).
        """
        import dice_ml
        import pandas as pd

        x_arr = np.asarray(x, dtype=np.float32)
        T, k = x_arr.shape
        D = T * k
        cols = [f"f{i}" for i in range(D)]

        # Use float64 columns: dice-ml's random/genetic samplers assign float64
        # sampled values back into the frame, which newer pandas (>=2.1) refuses
        # to coerce into float32 columns (raises TypeError on upcast). float64 is
        # safe for the gradient backend too (values are tensorised downstream).
        bg = self.background_data.reshape(self.background_data.shape[0], D).astype(np.float64)
        with torch.no_grad():
            bg_pred = model.predict(self.background_data)
        df = pd.DataFrame(bg, columns=cols)
        df["label"] = np.asarray(bg_pred).astype(int)

        data = dice_ml.Data(
            dataframe=df, continuous_features=cols, outcome_name="label"
        )

        class _FlatAdapter(nn.Module):
            def __init__(self, clf, T, k, target):
                super().__init__()
                self.clf = clf
                self.T, self.k, self.target = T, k, target

            def forward(self, x2d):
                x3d = x2d.view(-1, self.T, self.k)
                logits = self.clf.torch_logits(x3d)
                probs = F.softmax(logits, dim=-1)
                return probs[:, self.target : self.target + 1]

        adapter = _FlatAdapter(model, T, k, self.target_class)
        m = dice_ml.Model(model=adapter, backend="PYT")
        exp = dice_ml.Dice(data, m, method=self.method)

        query = pd.DataFrame(x_arr.reshape(1, D).astype(np.float64), columns=cols)
        result = exp.generate_counterfactuals(
            query, total_CFs=self.n_cfs, desired_class=self.target_class
        )
        cf_df = result.cf_examples_list[0].final_cfs_df
        cf_vals = cf_df[cols].to_numpy(dtype=np.float32)
        if cf_vals.shape[0] == 0:
            raise RuntimeError("dice-ml returned no counterfactuals")
        return cf_vals.reshape(cf_vals.shape[0], T, k)
