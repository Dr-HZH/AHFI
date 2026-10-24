# AHFI: Adaptive Hierarchical Fairness Intervention for Fair Graph Neural Networks

This paper has been accepted to CIKM 2026.

## Quick Start

### Installation

```bash
# After cloning the repository
conda env create -f environment.yml
conda activate fffgnn
```

### Basic Usage

```bash
# Quick test on NBA dataset
python main.py --dataset nba --epochs 200

# Use different GNN backbones on Credit dataset
python main.py --dataset credit --backbone GCN --epochs 200
python main.py --dataset credit --backbone GAT --epochs 200
```

## Dataset Setup

Download NBA and Pokec from [FairGNN](https://github.com/EnyanDai/FairGNN/tree/main/dataset), and Bail, Credit, and German from [FairVGNN](https://github.com/yuwvandy/FairVGNN/tree/main/dataset). Place the files under `dataset/`:

```text
dataset/
├── NBA/       # nba.csv, nba.embedding, nba_relationship.txt
├── bail/      # bail.csv, bail_edges.txt
├── credit/    # credit.csv, credit_edges.txt
├── german/    # german.csv, german_edges.txt
└── pokec/     # region_job{,_2}.csv, .embedding, and _relationship.txt
```

## Supported Datasets

- **nba** - NBA player data (Small, ~400 nodes)
- **credit** - Credit default prediction (Medium, ~30K nodes)
- **bail** - Bail decision (Medium, ~18K nodes)
- **german** - German credit data (Small, ~1K nodes)
- **pokec_n** - Social network data (66,569 nodes)
- **pokec_z** - Social network data (67,796 nodes)

## Supported GNN Backbones

- **GCN** - Graph Convolutional Network
- **GAT** - Graph Attention Network
- **GraphSAGE** - Sample and Aggregate
- **GIN** - Graph Isomorphism Network

## Model Parameters

```bash
python main.py --help
```

Key Parameters:
- `--dataset`: Dataset selection (default: nba)
- `--backbone`: GNN backbone (default: GCN)
- `--epochs`: Training epochs (default: 2000)
- `--gamma`: Relative constraint factor (default: 0.1). Controls how much the risk must decrease relative to history.
- `--lambda_decay`: Quadratic dual regularization coefficient beta (default: 0.01).
- `--ema-momentum`: Fairness risk EMA momentum (default: 0.9).
- `--lr`: Learning rate (default: 0.001)

## Output Example

After training, the script reports:
- Accuracy and F1 Score
- Fairness Metrics (Demographic Parity, Equal Opportunity)
- Training Time

## Project Structure

```
AHFI/
├── src/
│   ├── models/
│   │   ├── Adaptive_Hierarchical.py  # Main Model (Scale-Invariant & Learnable)
│   │   └── gnn_backbone.py           # GNN Backbones
│   ├── data/                         # Data Loaders
│   ├── training/                     # Unified Trainer (Primal-Dual)
│   └── utils/                        # Utilities
├── main.py                           # Main Training Script
├── environment.yml                   # Conda Environment
└── requirements.txt                  # Python Dependencies
```

## System Requirements

- Python 3.9 (provided Conda environment)
- PyTorch
- DGL (Deep Graph Library)
- CUDA (Recommended for GPU acceleration)

## License

MIT License

## Citation

```bibtex
@inproceedings{huang2026ahfi,
  title     = {AHFI: Adaptive Hierarchical Fairness Intervention for Fair Graph Neural Networks},
  author    = {Huang, Zihao and Lu, Ke and Xue, Jian},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3799682.3840594},
  url       = {https://doi.org/10.1145/3799682.3840594}
}
```
