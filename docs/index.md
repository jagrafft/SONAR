# SONAR

**Long-Range Graph Propagation Through Information Waves** (NeurIPS 2025).

This repository contains the official code for the paper: training and evaluation for graph transfer experiments, graph property prediction (GraphPropPred), and multi-domain benchmarks, plus the SONAR/BlockSONAR model implementations.

## Documentation

- **[Implementations report](SONAR_implementations_report.md)** — Detailed comparison of the three SONAR variants (GraphPropPred, graph_transfer_task, multi_domain_benchmarks).
- **[Paper reference](SONAR_paper_reference.md)** — Paper summary, equations, and paper→code mapping.
- **[Module index](PYTHON_INDEX.md)** — Manual index of all Python files and public entry points.
- **API** — Auto-generated API docs from docstrings (MkDocs + mkdocstrings):
  - [Root](api/root.md) — `runtimes` benchmark script
  - [graph_transfer_task](api/graph_transfer_task.md) — Synthetic long-range transfer
  - [GraphPropPred](api/GraphPropPred.md) — Graph property prediction
  - [multi_domain_benchmarks](api/multi_domain_benchmarks.md) — Heterophilic, LRGB, etc.

## Building the docs

```bash
pip install -r requirements-docs.txt
mkdocs serve   # local preview at http://127.0.0.1:8000
# or
mkdocs build   # output in site/
```

Build may require the project’s runtime dependencies (e.g. `torch`, `torch_geometric`, `ray`) so that mkdocstrings can import the modules.
