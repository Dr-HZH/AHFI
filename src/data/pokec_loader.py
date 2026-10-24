import torch
import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import random
import logging
from typing import Dict, Any, Tuple
from .base_loader import BaseDataLoader
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

class PokecDataLoader(BaseDataLoader):
    """Pokec dataset loader"""

    def __init__(self, sens_number: int = None, label_number: int = None, seed: int = 42, dataset: str = "pokec_n", test_ratio: float = 0.15, val_ratio: float = 0.15):
        """
        Initialize Pokec data loader

        Args:
            sens_number: Number of sensitive attributes (deprecated, kept for compatibility)
            label_number: Number of labels (deprecated, kept for compatibility)
            seed: Random seed
            dataset: Dataset name, either "pokec_n" or "pokec_z"
            test_ratio: Test set ratio, default 0.15
            val_ratio: Validation set ratio, default 0.15
        """
        super().__init__(sens_number, label_number, seed)
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dataset", "pokec")
        self.logger = logging.getLogger(__name__)

        # Dataset mapping
        self.dataset_map = {
            "pokec_n": "region_job_2",
            "pokec_z": "region_job"
        }

        if dataset not in self.dataset_map:
            raise ValueError(f"Unsupported dataset: {dataset}. Please choose 'pokec_n' or 'pokec_z'")

        self.dataset = self.dataset_map[dataset]
        self.sens_attr = "region"  # Sensitive attribute: region
        self.predict_attr = "I_am_working_in_field"  # Prediction attribute: work field
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self._check_data_files()

    def _check_data_files(self):
        """Check if necessary data files exist"""
        required_files = {
            f"{self.dataset}.csv": "User data file",
            f"{self.dataset}_relationship.txt": "User relationship data file",
            f"{self.dataset}.embedding": "User embedding file"
        }

        missing_files = []
        for file_name, description in required_files.items():
            file_path = os.path.join(self.data_dir, file_name)
            if not os.path.exists(file_path):
                missing_files.append(f"{file_name} ({description})")

        if missing_files:
            raise FileNotFoundError(
                f"Missing required data files:\n" +
                "\n".join(f"- {file}" for file in missing_files) +
                f"\nPlease ensure these files exist in directory: {self.data_dir}"
            )

    def _log_split_stats(self, labels, sens, idx_train, idx_val, idx_test):
        """Log dataset split statistics"""
        total = len(labels)
        train_size = len(idx_train)
        val_size = len(idx_val)
        test_size = len(idx_test)

        self.logger.info(f"Dataset split: Total samples={total}, Training={train_size} ({train_size/total:.1%}), "
                        f"Validation={val_size} ({val_size/total:.1%}), Test={test_size} ({test_size/total:.1%})")

        # Calculate sensitive attribute distribution in each subset
        for subset_name, subset_idx in [("Training", idx_train), ("Validation", idx_val), ("Test", idx_test)]:
            s0_count = (sens[subset_idx] == 0).sum().item()
            s1_count = (sens[subset_idx] == 1).sum().item()
            l0_count = (labels[subset_idx] == 0).sum().item()
            l1_count = (labels[subset_idx] == 1).sum().item()

            self.logger.info(f"{subset_name} - Sensitive attr=0: {s0_count} ({s0_count/len(subset_idx):.1%}), "
                           f"Sensitive attr=1: {s1_count} ({s1_count/len(subset_idx):.1%})")
            self.logger.info(f"{subset_name} - Label=0: {l0_count} ({l0_count/len(subset_idx):.1%}), "
                           f"Label=1: {l1_count} ({l1_count/len(subset_idx):.1%})")

            # Check sample count for each sensitive attribute + label combination
            s0_l0 = ((sens[subset_idx] == 0) & (labels[subset_idx] == 0)).sum().item()
            s0_l1 = ((sens[subset_idx] == 0) & (labels[subset_idx] == 1)).sum().item()
            s1_l0 = ((sens[subset_idx] == 1) & (labels[subset_idx] == 0)).sum().item()
            s1_l1 = ((sens[subset_idx] == 1) & (labels[subset_idx] == 1)).sum().item()

            self.logger.info(f"{subset_name} - Sens=0,Label=0: {s0_l0}, Sens=0,Label=1: {s0_l1}, "
                           f"Sens=1,Label=0: {s1_l0}, Sens=1,Label=1: {s1_l1}")

    def split_data(self, labels, sens):
        """Use the common class-stratified paper protocol."""
        return super().split_data(labels, sens)

    def split_sens(self, sens: torch.Tensor) -> torch.LongTensor:
        """Split sensitive attribute training set

        Following the original FairGNN implementation

        Args:
            sens: Sensitive attributes

        Returns:
            Sensitive attribute training set indices
        """
        # Convert to numpy array
        sens_cpu = sens.cpu().numpy()

        # Get indices of valid sensitive attributes
        sens_idx = set(np.where(sens_cpu >= 0)[0])

        # Exclude validation and test set indices from sensitive attribute indices
        idx_val_cpu = self.data.get('idx_val', torch.LongTensor([])).cpu().numpy()
        idx_test_cpu = self.data.get('idx_test', torch.LongTensor([])).cpu().numpy()

        sens_train_idx = list(sens_idx - set(idx_val_cpu) - set(idx_test_cpu))

        # Shuffle and limit sensitive attribute training set size if sens_number is specified
        random.seed(self.seed)
        random.shuffle(sens_train_idx)
        if self.sens_number is not None:
            sens_train_idx = sens_train_idx[:self.sens_number]

        return torch.LongTensor(sens_train_idx)

    def load(self) -> Dict[str, Any]:
        """Load Pokec dataset"""
        # Load raw data
        adj, features, labels, sens = self._load_raw_data()

        # Build graph
        graph = self._build_graph(adj)

        # Split dataset
        idx_train, idx_val, idx_test = self.split_data(labels, sens)

        # Create data dictionary
        self.data = {
            'graph': graph,
            'features': features,
            'labels': labels,
            'sens': sens,
            'idx_train': idx_train,
            'idx_val': idx_val,
            'idx_test': idx_test
        }

        # Get sensitive attribute training indices
        idx_sens_train = self.split_sens(sens)
        self.data['idx_sens_train'] = idx_sens_train

        return self.data

    def _load_raw_data(self) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load raw data

        Returns:
            Tuple containing:
                - adj: Adjacency matrix
                - features: Node features
                - labels: Node labels
                - sens: Sensitive attributes
        """
        # Read CSV file
        data_path = os.path.join(self.data_dir, f"{self.dataset}.csv")
        idx_features_labels = pd.read_csv(data_path)

        # Get column names
        header = list(idx_features_labels.columns)
        header.remove("user_id")  # Remove ID column
        header.remove(self.sens_attr)  # Remove sensitive attribute column
        header.remove(self.predict_attr)  # Remove prediction attribute column

        # Build feature matrix and labels
        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[self.predict_attr].values

        # Build graph
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)}

        edges_path = os.path.join(self.data_dir, f"{self.dataset}_relationship.txt")
        edges_unordered = np.genfromtxt(edges_path, dtype=int)

        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                        dtype=int).reshape(edges_unordered.shape)

        # Build adjacency matrix (symmetrize)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                           shape=(labels.shape[0], labels.shape[0]),
                           dtype=np.float32)

        # Build symmetric adjacency matrix (undirected graph)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])  # Add self-loops

        # Convert to tensors
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        sens = torch.FloatTensor(idx_features_labels[self.sens_attr].values)

        # Binarize labels: following original FairGNN implementation
        labels[labels > 1] = 1

        # Binarize sensitive attributes: following original FairGNN implementation
        sens[sens > 0] = 1

        return adj, features, labels, sens

    def _build_graph(self, adj: np.ndarray) -> dgl.DGLGraph:
        """Build DGL graph"""
        # Convert adjacency matrix to DGL graph
        src, dst = np.nonzero(adj)
        graph = dgl.graph((src, dst))
        return graph

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize feature matrix - special attention for NBA dataset which needs feature normalization

        Args:
            features: Feature matrix

        Returns:
            Normalized feature matrix
        """
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diag(r_inv)
        features = r_mat_inv.dot(features)
        return features

    def _normalize_adj(self, adj: sp.spmatrix) -> sp.spmatrix:
        """Normalize adjacency matrix

        Args:
            adj: Adjacency matrix

        Returns:
            Normalized adjacency matrix
        """
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def _random_split(self, total_size: int) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """Random data splitting (used when stratified sampling fails)"""
        indices = list(range(total_size))
        np.random.seed(self.seed)
        np.random.shuffle(indices)

        test_size = int(total_size * self.test_ratio)
        val_size = int(total_size * self.val_ratio)
        train_size = total_size - test_size - val_size

        idx_train = torch.LongTensor(indices[:train_size])
        idx_val = torch.LongTensor(indices[train_size:train_size + val_size])
        idx_test = torch.LongTensor(indices[train_size + val_size:])

        return idx_train, idx_val, idx_test
