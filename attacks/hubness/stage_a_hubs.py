"""Stage A — Hub vector generation.

Given a query distribution Q_est (and optionally a reconstructed memory
D_est), compute K hub vectors h_1..h_K ∈ S^(d-1) such that injecting them
into the memory store displaces legitimate retrievals.

Two methods:
  spherical_kmeans(Q, K)              — cluster query vectors; centroids = hubs.
                                         Closed-form, fast, default-fallback hubs.
  facility_location(Q, K, D=None)     — Adam on the unit sphere maximizing
                                         Σ_q max_k cos(h_k, q) − max_d cos(d, q).
                                         Tailored per-conversation hubs.

The displacement objective max_d cos(d, q) is a constant w.r.t. h, so it
shifts the loss but doesn't change the optimum. We compute it for diagnostic
reporting (per-query "are we beating the legitimate top-1?") and to enable
early-stopping when all queries are already displaced.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


@dataclass
class HubDiagnostics:
    """Per-hub-set quality measurements."""
    objective: float                  # Σ_q max_k cos(h_k, q)
    mean_max_sim: float               # mean over q of max_k cos(h_k, q)
    mean_top_d_sim: float | None      # mean over q of max_d cos(d, q); None if D is None
    displacement_rate: float | None   # fraction of queries where max_k cos(h_k, q) > max_d cos(d, q)


def _diagnose(H: np.ndarray, Q: np.ndarray, D: np.ndarray | None) -> HubDiagnostics:
    sims_h = Q @ H.T                        # (n_q, K)
    max_h = sims_h.max(axis=1)              # (n_q,)
    if D is not None:
        sims_d = Q @ D.T
        max_d = sims_d.max(axis=1)
        return HubDiagnostics(
            objective=float(max_h.sum()),
            mean_max_sim=float(max_h.mean()),
            mean_top_d_sim=float(max_d.mean()),
            displacement_rate=float((max_h > max_d).mean()),
        )
    return HubDiagnostics(
        objective=float(max_h.sum()),
        mean_max_sim=float(max_h.mean()),
        mean_top_d_sim=None,
        displacement_rate=None,
    )


def spherical_kmeans(
    Q: np.ndarray,
    K: int,
    *,
    n_init: int = 8,
    max_iter: int = 100,
    tol: float = 1e-5,
    seed: int = 0,
) -> np.ndarray:
    """Spherical k-means on L2-normalized vectors.

    Returns K cluster centroids (K, d), L2-normalized. Picks best of `n_init`
    runs by Σ_q max_k cos(q, h_k). Q is normalized internally if needed.
    """
    Q = _normalize_rows(np.asarray(Q, dtype=np.float64))
    n, d = Q.shape
    if K > n:
        raise ValueError(f"K={K} exceeds number of query vectors n={n}")
    rng = np.random.default_rng(seed)

    best_obj, best_H = -np.inf, None
    for _ in range(n_init):
        idx = rng.choice(n, size=K, replace=False)
        H = Q[idx].copy()
        prev_obj = -np.inf
        for _ in range(max_iter):
            sims = Q @ H.T
            assign = sims.argmax(axis=1)
            new_H = np.zeros_like(H)
            for k in range(K):
                members = Q[assign == k]
                if len(members) == 0:
                    new_H[k] = Q[rng.integers(n)]
                else:
                    new_H[k] = members.sum(axis=0)
            new_H = _normalize_rows(new_H)
            obj = sims.max(axis=1).sum()
            H = new_H
            if obj - prev_obj < tol:
                break
            prev_obj = obj
        obj = (Q @ H.T).max(axis=1).sum()
        if obj > best_obj:
            best_obj, best_H = obj, H

    return best_H.astype(np.float32)


def facility_location(
    Q: np.ndarray,
    K: int,
    D: np.ndarray | None = None,
    *,
    n_iter: int = 500,
    lr: float = 0.05,
    seed: int = 0,
    device: str = "cpu",
) -> np.ndarray:
    """Adam on the unit sphere maximizing the soft-max-coverage of Q.

    Objective (maximized): Σ_q max_k cos(h_k, q).
    With D provided, equivalent objective Σ_q [max_k − max_d] differs only
    by a constant; D is used for diagnostic reporting (see compute_hubs).

    Hubs are parameterized as unconstrained R^d and re-normalized in the
    forward pass; final returned vectors are L2-normalized.
    """
    Q_np = _normalize_rows(np.asarray(Q, dtype=np.float32))
    Q_t = torch.tensor(Q_np, device=device)

    n, d = Q_np.shape
    if K > n:
        raise ValueError(f"K={K} exceeds number of query vectors n={n}")
    rng = np.random.default_rng(seed)
    init = Q_np[rng.choice(n, size=K, replace=False)]
    H = torch.tensor(init, device=device, requires_grad=True)
    opt = torch.optim.Adam([H], lr=lr)

    for _ in range(n_iter):
        H_norm = H / H.norm(dim=1, keepdim=True).clamp_min(1e-12)
        sims = Q_t @ H_norm.T
        loss = -sims.max(dim=1).values.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    H_final = (H / H.norm(dim=1, keepdim=True).clamp_min(1e-12)).detach().cpu().numpy()
    return H_final.astype(np.float32)


def compute_hubs(
    Q: np.ndarray,
    K: int,
    D: np.ndarray | None = None,
    *,
    method: str = "kmeans",
    **kwargs,
) -> tuple[np.ndarray, HubDiagnostics]:
    """Top-level hub generator with diagnostics.

    method: 'kmeans' (fast, default-hubs) or 'facility' (Adam, per-conversation).
    Returns (hubs, diagnostics). When D is provided, diagnostics include
    displacement rate vs the legitimate top-1 doc per query.
    """
    if method == "kmeans":
        H = spherical_kmeans(Q, K, **kwargs)
    elif method == "facility":
        H = facility_location(Q, K, D=D, **kwargs)
    else:
        raise ValueError(f"unknown method: {method!r}")
    Q_norm = _normalize_rows(np.asarray(Q, dtype=np.float32))
    D_norm = _normalize_rows(np.asarray(D, dtype=np.float32)) if D is not None else None
    return H, _diagnose(H, Q_norm, D_norm)
