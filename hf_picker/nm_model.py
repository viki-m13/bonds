"""L2GMOM network-momentum model (PyTorch), adapted to underwater avoidance.

Faithful to Pu et al. (arXiv:2308.12212):
  * L2G layer (their Algorithm 1): unrolls L primal-dual-splitting (PDS) steps of
    Kalofolias graph-from-smooth-signals learning. The regularisation
    hyper-parameters (alpha, beta) and per-layer step sizes (gamma) are LEARNABLE
    — that is the "learning to learn" part.
  * Forward (their Eq. 10): A = L2G(V); A_norm = D^-1/2 A D^-1/2;
    y = A_norm @ (U theta) + b  — each asset's score is a graph-neighbour-weighted
    linear combination of momentum/risk features ("network momentum").

`LinearMom` is the no-graph ablation (y = U theta + b) on identical features, so
any gap is attributable to the learned network, not the features or the label.

The model outputs a per-asset score y; for the underwater objective LOWER y =
safer (we train y to track realised underwater fraction), and selection takes
the lowest-y eligible names.
"""
import numpy as np
import torch
import torch.nn as nn


def _triu_index(n: int):
    """Upper-triangle (i<j) row/col indices for the half-vectorisation."""
    iu = torch.triu_indices(n, n, offset=1)
    return iu[0], iu[1]


class L2G(nn.Module):
    """Unrolled PDS graph learner. Maps a feature matrix V (N x F) to a sparse,
    non-negative, symmetric adjacency A (N x N)."""

    def __init__(self, layers: int = 6):
        super().__init__()
        self.layers = layers
        # learnable, kept positive via softplus
        self._alpha = nn.Parameter(torch.tensor(0.0))     # log-degree barrier wt
        self._beta = nn.Parameter(torch.tensor(-2.0))     # ||w||^2 weight
        self._gamma = nn.Parameter(torch.full((layers,), -2.0))  # step sizes

    def _degree(self, w, ri, ci, n):
        d = w.new_zeros(n)
        d.index_add_(0, ri, w)
        d.index_add_(0, ci, w)
        return d

    def _degree_T(self, v, ri, ci):
        return v[ri] + v[ci]                              # D^T v on edges

    def forward(self, V: torch.Tensor) -> torch.Tensor:
        n = V.shape[0]
        ri, ci = _triu_index(n)
        ri, ci = ri.to(V.device), ci.to(V.device)
        diff = V[ri] - V[ci]
        h = (diff * diff).sum(1)                          # pairwise sq distances
        alpha = torch.nn.functional.softplus(self._alpha)
        beta = torch.nn.functional.softplus(self._beta)
        gammas = torch.nn.functional.softplus(self._gamma)

        w = torch.zeros_like(h)
        v = V.new_zeros(n)
        for l in range(self.layers):
            g = gammas[l]
            y1 = w - g * (2 * beta * w + self._degree_T(v, ri, ci))
            y2 = v + g * self._degree(w, ri, ci, n)
            p1 = torch.relu(y1 - 2 * g * h)
            p2 = (y2 - torch.sqrt(y2 * y2 + 4 * alpha * g)) / 2
            q1 = p1 - g * (2 * beta * p1 + self._degree_T(p2, ri, ci))
            q2 = p2 + g * self._degree(p1, ri, ci, n)
            w = w - y1 + q1
            v = v - y2 + q2
        w = torch.relu(w)
        A = V.new_zeros(n, n)
        A[ri, ci] = w
        A = A + A.t()
        return A


class L2GMOM(nn.Module):
    def __init__(self, n_features: int, layers: int = 6, graph: bool = True):
        super().__init__()
        self.graph = graph
        if graph:
            self.l2g = L2G(layers)
        self.theta = nn.Parameter(torch.zeros(n_features))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        """U: (N x F) features for one date. Returns y: (N,)."""
        proj = U @ self.theta + self.b
        if not self.graph:
            return proj
        A = self.l2g(U)
        d = A.sum(1)
        dinv = torch.where(d > 0, d.pow(-0.5), torch.zeros_like(d))
        An = dinv[:, None] * A * dinv[None, :]            # D^-1/2 A D^-1/2
        return An @ (U @ self.theta) + self.b


class LinearMom(nn.Module):
    """No-graph ablation."""

    def __init__(self, n_features: int):
        super().__init__()
        self.net = L2GMOM(n_features, graph=False)

    def forward(self, U):
        return self.net(U)


if __name__ == "__main__":
    torch.manual_seed(0)
    N, F = 120, 10
    U = torch.randn(N, F)
    m = L2GMOM(F, layers=6, graph=True)
    y = m(U)
    A = m.l2g(U)
    print("y", tuple(y.shape), "A", tuple(A.shape),
          "edges>0:", int((A > 0).sum().item() // 2),
          "mean deg:", float((A > 0).float().sum(1).mean()))
    y.sum().backward()
    print("grad on theta ok:", m.theta.grad is not None,
          "grad on alpha ok:", m.l2g._alpha.grad is not None)
