import numpy as np
import argparse
import torch
import warnings
import os
from src.models.Adaptive_Hierarchical import Adaptive_Hierarchical
from src.data.pokec_loader import PokecDataLoader
from src.data.nba_loader import NBADataLoader
from src.data.bail_loader import BailDataLoader
from src.data.credit_loader import CreditDataLoader
from src.data.german_loader import GermanDataLoader
from src.training.unified_trainer import UnifiedTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=.5)

    parser.add_argument('--dataset', type=str, default='nba',
                        choices=['pokec_z','pokec_n','nba','bail','credit','german'])
    parser.add_argument('--num-hidden', type=int, default=128)
    parser.add_argument('--experiment_note', type=str, default='')
    parser.add_argument('--sens-number', type=int, default=None)
    parser.add_argument('--label-number', type=int, default=None)
    parser.add_argument('--nclass', type=int, default=2)
    parser.add_argument('--backbone', type=str, default='GCN',
                       choices=['GCN', 'GAT', 'GraphSAGE', 'GIN'])
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)


    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Relative constraint factor for fairness monitor (default: 0.1)')

    parser.add_argument('--lambda_decay', type=float, default=0.01,
                        help='Quadratic dual regularization coefficient beta (default: 0.01)')
    parser.add_argument('--ema-momentum', type=float, default=0.9,
                        help='Fairness risk EMA momentum (default: 0.9)')
    return parser.parse_args()

def main():
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', message='.*CUDA.*')
    warnings.filterwarnings('ignore', message='.*DGL.*')
    args = parse_args()
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    if args.dataset in ['pokec_z', 'pokec_n']:
        dataloader = PokecDataLoader(seed=args.seed, dataset=args.dataset, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    elif args.dataset == 'nba':
        dataloader = NBADataLoader(seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    elif args.dataset == 'bail':
        dataloader = BailDataLoader(seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    elif args.dataset == 'credit':
        dataloader = CreditDataLoader(seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    elif args.dataset == 'german':
        dataloader = GermanDataLoader(seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    data = dataloader.load()
    for key in data:
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].to(device)
    if 'graph' in data:
        data['graph'] = data['graph'].to(device)
    sens_idx = dataloader.sens_idx if hasattr(dataloader, 'sens_idx') else None


    num_nodes_extracted = data['features'].shape[0]
    print(f"Dataset loaded. Total nodes: {num_nodes_extracted}")


    model = Adaptive_Hierarchical(
            nfeat=data['features'].shape[1],
            nhid=args.num_hidden,
            nclass=args.nclass,
            num_nodes=num_nodes_extracted,
            arch=args.backbone,
            heads=args.heads,
            dropout=args.dropout,
            gamma=args.gamma,
            lambda_decay=args.lambda_decay,
            ema_momentum=args.ema_momentum
        )
    model = model.to(device)
    if sens_idx is not None and 'sens_idx' not in data:
        data['sens_idx'] = sens_idx

    model_name = f"Adaptive_Hierarchical_{args.dataset}_{args.backbone}"
    trainer = UnifiedTrainer(args, model, data, model_name=model_name)
    results = trainer.train()
    print("\nTraining Completed!")
    print(f"Accuracy: {results['test_acc']:.4f}")
    print(f"Delta DP: {results['test_parity']:.4f}")
    print(f"Delta EO: {results['test_equality']:.4f}")
    print(f"F1 Score: {results['test_f1']:.4f}")
    print(f"Training Time: {results['training_time']:.1f}s")

if __name__ == "__main__":
    main()
