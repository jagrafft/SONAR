# SONAR paper reference

## Citation and link

**A. Trenta, A. Gravina, D. Bacciu.** *SONAR: Long-Range Graph Propagation Through Information Waves.* NeurIPS 2025.

- [OpenReview PDF](https://openreview.net/pdf?id=Hxfjmc95rl)

## Abstract (short)

SONAR models information flow on graphs as oscillations governed by the wave equation, enabling effective long-range propagation. It uses a weighted graph Laplacian $\mathbf{L}^a$ with adaptive edge resistances $a_{uv}$ (learned or fixed) and optional state-dependent dissipation $D$ and external forcing $F$. In the conservative case, energy is preserved (Theorem 3.1) and sensitivity between nodes does not vanish (Theorem 3.2), so information can propagate over long distances.

## Method summary

All math below uses inline `$ ... $` or display `\[ ... \]` LaTeX.

### Continuous dynamics (Eq. 4–5)

The evolution of node states follows the graph wave equation with optional dissipation and forcing:

\[
\ddot{\mathbf{X}}(t) = -\mathbf{L}^a \mathbf{X}(t) \mathbf{W} - D(\mathbf{X}(t)) \odot \dot{\mathbf{X}}(t) + F(\mathbf{X}(t))
\]

Here $\mathbf{L}^a$ is the **weighted graph Laplacian**; the edge weight $a_{uv}$ acts as an inverse resistance governing how easily information flows between nodes $u$ and $v$. $\mathbf{W}$ is a learnable weight matrix; $D$ and $F$ are state-dependent dissipative and forcing terms (implemented as small nets or set to zero).

### Discretization (Eq. 6–7)

Introduce the auxiliary velocity $\mathbf{V} = \dot{\mathbf{X}}$ and step size $h$. The first-order system is discretized as:

\[
\mathbf{V}^{\ell+1} = \mathbf{V}^{\ell} - h\left( \mathbf{L}^a \mathbf{X}^{\ell} \mathbf{W} + D(\mathbf{X}^{\ell}) \odot \mathbf{V}^{\ell} - F(\mathbf{X}^{\ell}) \right)
\]

\[
\mathbf{X}^{\ell+1} = \mathbf{X}^{\ell} + h \mathbf{V}^{\ell+1}
\]

Initial conditions: $\mathbf{X}^0 = \bar{\mathbf{X}}$ (input features); $\mathbf{V}^0 = \mathbf{X}^0 \mathbf{W}_V$ (learnable initial velocity) or $\mathbf{V}^0 = \mathbf{0}$.

### Deep architecture (Eq. 8)

A **SONAR block** runs $L$ propagation steps (the equations above) then applies an MLP. The output of block $i$ is the initial state for block $i+1$. Multiple blocks are stacked to form the full model.

### Energy and sensitivity (Eq. 9, Theorems 3.1–3.2)

The energy

$$
E(t) = \frac{1}{2} \sum_v \|\nabla \mathbf{x}_v(t)\|^2 + \frac{1}{2} \left\| \frac{\partial \mathbf{X}(t)}{\partial t} \right\|^2
$$

is **conserved** when $D$ and $F$ are zero (Theorem 3.1). The sensitivity $\partial \mathbf{x}_v(t) / \partial \mathbf{x}_u(0)$ between nodes $u$ and $v$ equals $\cos(t\sqrt{\mathbf{L}})_{vu}$ (Theorem 3.2), so it oscillates and **does not vanish**, enabling long-range propagation.

---

## Paper → code mapping

| Paper (symbols in `$ ... $`) | Code (all three variants where applicable) |
| ------------------------------ | ------------------------------------------ |
| $\mathbf{L}^a$, edge weights $a_{uv}$ | `LaplacianAggr`: `get_laplacian(..., edge_weight=edge_resistance)`; resistance from `edge_resistance_net` (or fixed). |
| Step size $h$ | `SONARConv.epsilon` |
| $\mathbf{V}^{\ell+1}$ update | `v = v - self.epsilon * (conv + diss*v - forcing)` |
| $\mathbf{X}^{\ell+1}$ update | `x = x + self.epsilon * v` (GraphPropPred, multi_domain) or `x = x + self.epsilon * self.activation(v)` (graph_transfer) |
| $\mathbf{V}^0 = \mathbf{X}^0 \mathbf{W}_V$ | GraphPropPred / graph_transfer: `v = zeros_like(x)`; multi_domain: `v = self.velocity_net(x)` |
| $D(\mathbf{X})$, $F(\mathbf{X})$ | `dissipative_net`, `external_forcing_net` (or `NullForce`) |
| Learnable $a_{uv}$ | `edge_resistance_net`: input node pair (and optional edge features); output scalar, then `.abs()` |
| Block structure (Eq. 8) | `BlockSONAR` / `BlockSONAR_Model`: `num_blocks` × (SONARConv + MLP) |

---

## Experiments (paper sections → repo)

| Paper section | Repo location |
| ------------- | ------------- |
| §4.1 Graph Transfer | `graph_transfer_task` |
| §4.2 Graph Property Prediction | `GraphPropPred` |
| §4.3 Long-Range Graph Benchmark (LRGB) | `multi_domain_benchmarks` |
| §4.4 Heterophilic tasks | `multi_domain_benchmarks` |
| §4.5 Empirical sensitivity analysis | `graph_transfer_task/sensitivity.py` (`forward_sensitivity`, `jacrev`) |
