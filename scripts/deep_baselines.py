#!/usr/bin/env python3
"""Deep imbalanced-regression baselines on a shared MLP backbone.

This module provides faithful, tabular-adapted re-implementations of three
published deep imbalanced-regression methods so they can be compared against
the tree-based methods in ``run_radical_smoke.py`` under an identical protocol
(chronological split, standardized inputs, fixed-tail metrics):

* RankSim   -- Gong et al., "RankSim: Ranking Similarity Regularization for
               Deep Imbalanced Regression", ICML 2022.
* ConR      -- Keramati et al., "ConR: Contrastive Regularizer for Deep
               Imbalanced Regression", ICLR 2024.
* VIR       -- Wang & Wang, "Variational Imbalanced Regression", NeurIPS 2023.

All three share a common MLP encoder/regression head so that differences in
the reported metrics reflect the imbalance-handling objective rather than the
backbone. Inputs are assumed pre-scaled (StandardScaler) by the caller; targets
are standardized internally and predictions are returned on the original scale.

The regularizer formulations are batch-based surrogates of the published
objectives, documented inline. They are intended as strong, faithful baselines
for tabular imbalanced regression, not bit-exact reproductions of the original
vision/age-estimation code.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_OK = True
except Exception:  # pragma: no cover - torch always present in our env
    _TORCH_OK = False


def torch_available() -> bool:
    return _TORCH_OK


# --------------------------------------------------------------------------- #
# Shared utilities
# --------------------------------------------------------------------------- #
def _set_seed(seed: int) -> None:
    np.random.seed(seed % (2**31 - 1))
    if _TORCH_OK:
        torch.manual_seed(seed)


def _standardize_targets(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    mu = float(np.mean(y))
    sd = float(np.std(y))
    if sd < 1e-8:
        sd = 1.0
    return (y - mu) / sd, mu, sd


def _epoch_schedule(n: int) -> int:
    """Fewer epochs for large datasets; more for small ones (bounded)."""
    return int(np.clip(300000 // max(n, 1) + 60, 80, 260))


def _density_weights(y: np.ndarray, bins: int = 24, max_weight: float = 8.0) -> np.ndarray:
    """LDS-style label-density weights (used by VIR's reweighted likelihood)."""
    counts, edges = np.histogram(y, bins=bins)
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    smooth = np.convolve(counts.astype(np.float64), kernel, mode="same")
    bin_idx = np.clip(np.digitize(y, edges[1:-1], right=False), 0, len(counts) - 1)
    density = np.maximum(smooth[bin_idx], 1.0)
    weights = np.sqrt(np.median(smooth[smooth > 0]) / density)
    return np.clip(weights, 1.0, max_weight).astype(np.float64)


if _TORCH_OK:

    class _MLPRegressor(nn.Module):
        """Shared encoder + linear regression head used by RankSim and ConR."""

        def __init__(self, in_dim: int, hidden: int = 256, feat_dim: int = 128):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, feat_dim),
                nn.ReLU(),
            )
            self.head = nn.Linear(feat_dim, 1)

        def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            z = self.encoder(x)
            return self.head(z).squeeze(-1), z

    class _VIRNet(nn.Module):
        """Probabilistic encoder/decoder for Variational Imbalanced Regression."""

        def __init__(self, in_dim: int, hidden: int = 256, latent: int = 64):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.fc_mu = nn.Linear(hidden, latent)
            self.fc_logvar = nn.Linear(hidden, latent)
            self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU())
            self.mean_head = nn.Linear(hidden, 1)
            self.logvar_head = nn.Linear(hidden, 1)

        def encode(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            h = self.shared(x)
            return self.fc_mu(h), torch.clamp(self.fc_logvar(h), -8.0, 8.0)

        def decode(self, z: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            h = self.dec(z)
            pred_logvar = torch.clamp(self.logvar_head(h).squeeze(-1), -8.0, 8.0)
            return self.mean_head(h).squeeze(-1), pred_logvar


def _ranksim_regularizer(z: "torch.Tensor", y: "torch.Tensor", temp: float = 2.0) -> "torch.Tensor":
    """Listwise ranking-similarity surrogate of RankSim.

    For each anchor, the distribution over neighbours induced by feature-space
    cosine similarity is aligned (cross-entropy) with the distribution induced
    by negative label distance, so the ordering of feature similarities matches
    the ordering of label similarities (the RankSim objective).
    """
    b = z.shape[0]
    if b < 4:
        return z.new_zeros(())
    yl = y.view(-1, 1)
    label_d = torch.abs(yl - yl.t())
    zn = F.normalize(z, dim=1)
    feat_sim = zn @ zn.t()
    eye = torch.eye(b, device=z.device, dtype=torch.bool)
    neg_inf = torch.finfo(z.dtype).min
    p_label = F.softmax((-label_d / temp).masked_fill(eye, neg_inf), dim=1)
    log_p_feat = F.log_softmax((feat_sim / temp).masked_fill(eye, neg_inf), dim=1)
    return -(p_label * log_p_feat).sum(dim=1).mean()


def _conr_regularizer(
    z: "torch.Tensor", y: "torch.Tensor", omega: float = 0.5, temp: float = 0.2
) -> "torch.Tensor":
    """Contrastive regularizer (ConR).

    Positives are samples whose (standardized) label lies within ``omega`` of the
    anchor; negatives are pushed with a power that grows with label distance, so
    examples close in feature space but far in label space are repelled.
    """
    b = z.shape[0]
    if b < 4:
        return z.new_zeros(())
    zn = F.normalize(z, dim=1)
    sim = (zn @ zn.t()) / temp
    yl = y.view(-1, 1)
    label_d = torch.abs(yl - yl.t())
    eye = torch.eye(b, device=z.device, dtype=torch.bool)
    pos_mask = (label_d <= omega) & (~eye)
    push = 1.0 + label_d  # ConR pushing power scales with label distance
    sim_exp = torch.exp(sim).masked_fill(eye, 0.0) * push
    denom = sim_exp.sum(dim=1, keepdim=True) + 1e-8
    log_prob = sim - torch.log(denom)
    pos_count = pos_mask.sum(dim=1)
    has_pos = pos_count > 0
    if not bool(has_pos.any()):
        return z.new_zeros(())
    loss = -(log_prob * pos_mask).sum(dim=1)[has_pos] / pos_count[has_pos].clamp(min=1)
    return loss.mean()


def _iterate_minibatches(n: int, batch: int, rng: np.random.Generator):
    idx = rng.permutation(n)
    for start in range(0, n, batch):
        yield idx[start : start + batch]


def _fit_regularized_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
    reg_kind: str,
    reg_lambda: float,
    batch: int = 512,
) -> tuple[np.ndarray, dict]:
    """Train the shared MLP with ERM + a RankSim/ConR regularizer."""
    _set_seed(seed)
    rng = np.random.default_rng(seed)
    ys, mu_y, sd_y = _standardize_targets(y_train)
    device = torch.device("cpu")
    model = _MLPRegressor(X_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    xt = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.as_tensor(ys, dtype=torch.float32, device=device)
    epochs = _epoch_schedule(len(y_train))
    model.train()
    for _ in range(epochs):
        for batch_idx in _iterate_minibatches(len(y_train), batch, rng):
            bi = torch.as_tensor(batch_idx, dtype=torch.long, device=device)
            pred, z = model(xt[bi])
            loss = F.mse_loss(pred, yt[bi])
            if reg_lambda > 0.0:
                if reg_kind == "ranksim":
                    loss = loss + reg_lambda * _ranksim_regularizer(z, yt[bi])
                elif reg_kind == "conr":
                    loss = loss + reg_lambda * _conr_regularizer(z, yt[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        xtest = torch.as_tensor(X_test, dtype=torch.float32, device=device)
        pred_std, _ = model(xtest)
        pred = pred_std.cpu().numpy() * sd_y + mu_y
    return pred.astype(np.float64), {"n_fit": int(len(y_train)), "epochs": int(epochs)}


def fit_predict_ranksim(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    thresholds: tuple[float, float],
    seed: int,
) -> tuple[np.ndarray, dict]:
    return _fit_regularized_mlp(X_train, y_train, X_test, seed, "ranksim", reg_lambda=1.0)


def fit_predict_conr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    thresholds: tuple[float, float],
    seed: int,
) -> tuple[np.ndarray, dict]:
    return _fit_regularized_mlp(X_train, y_train, X_test, seed, "conr", reg_lambda=0.5)


def fit_predict_vir(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    thresholds: tuple[float, float],
    seed: int,
    batch: int = 512,
    beta: float = 1e-3,
) -> tuple[np.ndarray, dict]:
    """Variational Imbalanced Regression with label-density reweighted likelihood."""
    _set_seed(seed)
    rng = np.random.default_rng(seed)
    ys, mu_y, sd_y = _standardize_targets(y_train)
    w = _density_weights(y_train)
    device = torch.device("cpu")
    model = _VIRNet(X_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    xt = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.as_tensor(ys, dtype=torch.float32, device=device)
    wt = torch.as_tensor(w, dtype=torch.float32, device=device)
    epochs = _epoch_schedule(len(y_train))
    model.train()
    for _ in range(epochs):
        for batch_idx in _iterate_minibatches(len(y_train), batch, rng):
            bi = torch.as_tensor(batch_idx, dtype=torch.long, device=device)
            mu, logvar = model.encode(xt[bi])
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            pred_mean, pred_logvar = model.decode(z)
            inv_var = torch.exp(-pred_logvar)
            nll = 0.5 * (pred_logvar + (yt[bi] - pred_mean) ** 2 * inv_var)
            nll = (wt[bi] * nll).mean()
            kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()
            loss = nll + beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        xtest = torch.as_tensor(X_test, dtype=torch.float32, device=device)
        samples = []
        for _ in range(8):  # predictive mean over latent samples
            mu, logvar = model.encode(xtest)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            pred_mean, _ = model.decode(z)
            samples.append(pred_mean.cpu().numpy())
        pred = np.mean(np.vstack(samples), axis=0) * sd_y + mu_y
    return pred.astype(np.float64), {"n_fit": int(len(y_train)), "epochs": int(epochs)}
