# SONAR Python module index

Reference for all `*.py` files in the SONAR repository (NeurIPS 2025: "SONAR: Long-Range Graph Propagation Through Information Waves").

## Repository layout

| Area | Role |
|------|------|
| **Root** | Single benchmark script comparing GCN, SONAR, GPS on runtime and GPU memory. |
| **graph_transfer_task** | Synthetic long-range transfer experiments (line, ring, crossed-ring) with multiple GNNs and BlockSONAR. |
| **GraphPropPred** | Graph property prediction (diam, sssp, ecc) with BlockSONAR and Ray-based model selection. |
| **multi_domain_benchmarks** | Broader benchmarks (heterophilic, LRGB, graph classification, etc.) with Experiment/ModelSelection and BlockSONAR. |

**Dependencies between packages:** `runtimes.py` (root) imports `SONARConv` from `GraphPropPred.models.sonar`. The three SONAR/BlockSONAR implementations (graph_transfer_task, GraphPropPred, multi_domain_benchmarks) are independent copies tailored to each benchmark.

---

## Root

| File | Purpose | Public entry points |
|------|---------|---------------------|
| `runtimes.py` | Runnable script. Benchmarks GCN, SONAR, GPS, sonar2 on Roman-empire (train/test time, GPU memory) for 2–32 layers; writes CSV per layer count, plots vs layers; uses `runtimes.pkl` cache. | `GNN`, `train`, `test` |

**Data/control flow:** Loads HeterophilousGraphDataset (Roman-empire), builds a small GNN (embedding → conv stack → decoder). For each `num_layers` and each `conv_name`, finds a hidden_dim yielding ~100k params, runs 11 epochs, records train/test time and GPU memory (or OOM), saves to CSV and pickle. Then loads pickle and plots train_time_vs_layers.png and test_time_vs_layers.png.

---

## graph_transfer_task

| File | Purpose | Public entry points |
|------|---------|---------------------|
| `main.py` | CLI entry for graph transfer experiments. Builds train/val/test `GraphTransferDataset`, runs model selection (Ray or debug) per model and distance, writes partial and final CSVs. | `__main__`: argparse, dataset creation, `getconf` → exp_list → `run_exp` / `run_single_exp` → aggregate. |
| `conf.py` | Hyperparameter grids and MODELS registry. | `get_PHDGN_conservative_conf`, `get_PHDGN_conf`, `get_GNN_conf`, `get_ADGN_conf`, `get_SWAN_conf`, `get_SONAR_conf`, `get_BlockSONAR_conf`, `MODELS` |
| `train.py` | Train/test loops; single-experiment runner with checkpoint resume; Ray remote for parallel runs. | `train`, `test`, `run_exp` (Ray remote), `run_single_exp` |
| `graph_transfer_data.py` | Synthetic graph generators and PyG dataset. | `line_graph`, `ring_transfer_graph`, `cliquepath_transfer_graph`, `distributions`, `ring_types`, `line_types`, `GraphTransferDataset` |
| `utils.py` | Shared utilities. | `set_seed`, `update_csv`, `cartesian_product` |
| `sensitivity.py` | Runnable script. Sensitivity analysis (finite-diff / jacrev) for BlockSONAR vs GNN on ring/line; saves .npy; ends with ValueError. | `GNN`, `GNN.forward_sensitivity` |
| `run_all.py` | Runnable script. Loops over models and distances, invokes main.py via shell (CUDA_VISIBLE_DEVICES, batch, root). | Script block: `root`, `gpus`, `models`, `distances`. |
| `plot_graph_transfer.py` | Runnable script. Reads result CSVs and baseline CSVs; plots MSE vs source–target distance (log scale) per graph type. | Script block: `plot`, `plot_std`, mappers, colors. |
| `models/__init__.py` | Re-exports models for graph transfer. | `ADGN_Model`, `GAT_Model`, `GCN_Model`, `GIN_Model`, `GPS_Model`, `SAGE_Model`, `PHDGN_Model`, `SWAN_Model`, `BlockSONAR_Model` |
| `models/sonar.py` | Graph transfer variant of SONAR. | `LaplacianAggr`, `NullForce`, `SONARConv`, `BlockSONAR_Model` |
| `models/gnn_model.py` | Base and standard GNNs. | `BasicModel`, `GCN_Model`, `GAT_Model`, `GIN_Model`, `SAGE_Model`, `GPS_Model` |
| `models/adgn_model.py` | Anti-symmetric DGN. | `ADGN_Model` |
| `models/phdgn_model.py` | PHDGN model. | `PHDGN_Model` |
| `models/swan_model.py` | SWAN model. | `SWAN_Model` |
| `models/ausiliar_modules.py` | Auxiliary layers for other models. | (classes used by adgn/phdgn/swan) |
| `models/phdgn_utils.py` | PHDGN helpers. | (functions/classes for PHDGN) |

**Data/control flow:** `main.py` parses args, builds `GraphTransferDataset` for train/val/test, gets configs from `conf.MODELS[model_name]` (model class + getconf generator). For each config and seed it builds `exp_list` and runs `run_exp.remote(...)` or `run_single_exp(...)`. `train.run_single_exp` loads data, builds model, trains with early stopping, saves checkpoints and returns best losses. Results are aggregated per conf_id and written to partial_results_*.csv and final_results_*.csv.

---

## GraphPropPred

| File | Purpose | Public entry points |
|------|---------|---------------------|
| `main.py` | Entry point for GraphPropPred. Argparse (task, model_name, epochs, early_stopping, save_dir, cpus, gpus); calls model_selection; Ray init. | `__main__`: args, `model_selection(...)` |
| `conf.py` | Config generator and registry. | `cartesian_product`, `config_BlockSONAR_GraphProp`, `CONFIGS` |
| `model_selection.py` | Runs model selection over configs. | `model_selection` |
| `train_GraphProp.py` | Training and evaluation; Ray pipeline. | `optimizer_to`, `train`, `evaluate`, `train_val_pipeline_GraphProp` (Ray remote) |
| `read_results.py` | Runnable script. Scans task dirs for results.csv; builds summary table (best val → test score ± std). | Script block: path layout, final_df. |
| `utils/__init__.py` | Dataset loader and constants. | `get_dataset`, `GPPDataset`, `TASKS`, `DATA` |
| `utils/gpp_dataset.py` | GPP dataset (download/process). | `normalize`, `NODE_LVL_TASKS`, `GRAPH_LVL_TASKS`, `TASKS`, `GPPDataset` |
| `utils/io/__init__.py` | I/O helpers. | `dump`, `load`, `create_if_not_exist`, `join` |
| `utils/io/save.py` | Serialization by extension. | `dump`, `load`, `create_if_not_exist`, `_resolve` |
| `utils/io/_save_helpers.py` | Internal JSON/pickle helpers. | `_dump_json`, `_load_json`, `_dump_bin`, `_load_bin`, `IO_HELPERS` |
| `models/__init__.py` | Model export. | `BlockSONAR` |
| `models/sonar.py` | GraphPropPred variant of SONAR (node/graph-level readout). | `LaplacianAggr`, `NullForce`, `SONARConv`, `BlockSONAR` |

**Data/control flow:** `main.py` parses args, creates save_dir/task/model_name dirs, calls `model_selection(model_name=..., task=..., ...)`. `model_selection` downloads data once, then for each config runs `train_val_pipeline_GraphProp.remote(...)`. Each remote loads train/val/test, runs multiple seeds with checkpoint resume, returns best train/val/test loss and score; results are collected into partial_results.csv and complete_results.json. Best config is chosen by avg best_val_score.

---

## multi_domain_benchmarks

| File | Purpose | Public entry points |
|------|---------|---------------------|
| `main.py` | Entry point. Parses args, sets root/result/ckpt dirs, loads dataset; either single experiment or ModelSelection.run(). | `__main__`: two branches (`Experiment(args).run_()` vs `ModelSelection(args).run()`) |
| `run.py` | Example runner script. Loop of nohup commands for dataset/model (e.g. questions, BlockSONAR). | Script block. |
| `experiments.py` | Single-run experiment with folds. | `Experiment`, `Experiment.run_`, `Experiment.single_fold`, `Experiment.train_and_test`, `Experiment.train`, `Experiment.test` |
| `model_selection.py` | Parallel model selection over configs. | `ModelSelection`, `ModelSelection.collect_results`, `ModelSelection.aggregate_res`, `ModelSelection.run` |
| `helpers/parse_arguments.py` | ArgumentParser for all benchmark options. | `parse_arguments` |
| `helpers/classes.py` | Shared enums and config classes. | `Pool`, `ActivationType`, `GumbelArgs`, etc. |
| `helpers/constants.py` | Constants. | (module-level names) |
| `helpers/encoders.py` | Position encoders and dataset encoders/decoders. | `PosEncoder`, dataset encoder/decoder APIs |
| `helpers/metrics.py` | Metric and loss types. | `MetricType`, `LossesAndMetrics` |
| `helpers/utils.py` | Utilities. | `set_seed`, `optimizer_to`, etc. |
| `helpers/dataset_classes/dataset.py` | Dataset enum and loading. | `DataSet`, `DataSetFamily`, `DatasetBySplit`, `load`, `get_folds`, `select_fold_and_split` |
| `helpers/dataset_classes/classic_datasets.py` | Planetoid (Cora, etc.). | (dataset class and load) |
| `helpers/dataset_classes/cycles_dataset.py` | Cycles dataset. | (class and interface) |
| `helpers/dataset_classes/lrgb.py` | LRGB peptides. | `PeptidesFunctionalDataset`, `PeptidesStructuralDataset` |
| `helpers/dataset_classes/root_neighbours_dataset.py` | Root-neighbours dataset. | (class and interface) |
| `helpers/dataset_classes/graph_prop_pred_dataset.py` | GPP dataset integration. | (class and interface) |
| `models/model_configs.py` | Model type and config API. | `ModelType`, `get_conf`, `get_single_conf`, `get()` |
| `models/sonar.py` | Multi-domain SONAR (edge features). | `EdgeFeatureAggregator`, `LaplacianAggr`, `SONARConv`, `BlockSONAR` |
| `models/layers.py` | Shared NN layers. | (public classes/functions) |
| `lrgb/split_generator.py` | LRGB split generation. | (main functions) |
| `lrgb/cosine_scheduler.py` | Cosine schedule with warmup. | (scheduler function) |
| `lrgb/plateau_scheduler.py` | Plateau scheduler. | (class and step) |
| `lrgb/transforms.py` | LRGB transforms. | `apply_transform`, exported transforms |
| `lrgb/encoders/composition.py` | Composition encoders. | (public API) |
| `lrgb/encoders/laplace.py` | Laplace encoders. | (public API) |
| `lrgb/encoders/compute.py` | Encoder computation. | (public API) |
| `lrgb/encoders/mol_encoder.py` | Molecular encoder. | (public API) |
| `lrgb/encoders/kernel.py` | Kernel encoders. | (public API) |
| `lrgb/encoders/voc_encoder.py` | VOC encoder. | (public API) |

**Data/control flow:** `main.py` calls `parse_arguments()`, sets `root_dir`, `result_dir`, `ckpt_dir`, loads dataset once. If `run_single_experiment`: builds single conf and runs `Experiment(args).run_()`. Else: runs `ModelSelection(args).run()`. `ModelSelection` builds `conf_list` from `args.model.get_conf(...)`, then for each conf and seed submits `Experiment.run.remote(...)`. `Experiment.run_()` loads dataset, iterates folds, runs `single_fold` (train_and_test with checkpoint resume), aggregates metrics and returns (mean, std, matrix, exp_args). `ModelSelection` collects results, aggregates by conf_id, writes partial_results.pkl/csv and model_selection_results.csv.
