import os
import torch
import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Dict, Any, Tuple, List, Union
from .base_loader import BaseDataLoader
from sklearn.model_selection import train_test_split
import random

class CreditDataLoader(BaseDataLoader):
    """Credit dataset loader"""

    def __init__(self, sens_number: int = 200, label_number: int = 500, seed: int = 42, test_ratio: float = 0.2, val_ratio: float = 0.2):
        """
        Initialize Credit data loader

        Args:
            sens_number: Number of sensitive attribute training samples
            label_number: Number of label training samples
            seed: Random seed
            test_ratio: Test set ratio
            val_ratio: Validation set ratio
        """
        super().__init__(sens_number, label_number, seed)
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dataset", "credit")
        self.sens_attr = "Age"  # Sensitive attribute: age
        self.predict_attr = "NoDefaultNextMonth"  # Prediction attribute: default risk
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self._check_data_files()

    def _check_data_files(self):
        """Check if necessary data files exist"""
        required_files = {
            "credit.csv": "Credit data file",
            "credit_edges.txt": "Credit relationship data file"
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

    def load(self) -> Dict[str, Any]:
        """Load Credit dataset

        Returns:
            Dictionary containing graph, features, labels and indices
        """
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
        # Load credit data
        data_path = os.path.join(self.data_dir, "credit.csv")
        df = pd.read_csv(data_path)

        # Set user ID as index
        if 'user_id' not in df.columns:
            df['user_id'] = np.arange(len(df))

        # Get feature columns - exclude ID and label columns
        feature_cols = df.columns.tolist()
        feature_cols.remove('user_id')
        feature_cols.remove(self.sens_attr)
        feature_cols.remove(self.predict_attr)

        # Convert features to sparse matrix
        features = sp.csr_matrix(df[feature_cols].values, dtype=np.float32)
        features = torch.FloatTensor(np.array(features.todense()))

        # Get labels and sensitive attributes
        sens = torch.FloatTensor(df[self.sens_attr].values)
        labels = torch.LongTensor(df[self.predict_attr].values)

        # Ensure labels are binary, set values > 1 to 1
        labels[labels > 1] = 1

        # Build graph
        idx = np.array(df['user_id'], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)}

        # Load edge data
        edges_path = os.path.join(self.data_dir, "credit_edges.txt")
        edges_unordered = np.loadtxt(edges_path, dtype=int)

        # Map node IDs
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                        dtype=int).reshape(edges_unordered.shape)

        # Build adjacency matrix
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                          shape=(labels.shape[0], labels.shape[0]),
                          dtype=np.float32)

        # Build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])

        return adj, features, labels, sens

    def _build_graph(self, adj: np.ndarray) -> dgl.DGLGraph:
        """Build DGL graph

        Args:
            adj: Adjacency matrix

        Returns:
            DGL graph
        """
        # Convert adjacency matrix to DGL graph
        src, dst = np.nonzero(adj)
        graph = dgl.graph((src, dst))
        return graph

    def split_data(self, labels, sens):
        """Use the common class-stratified paper protocol."""
        return super().split_data(labels, sens)

    def split_sens(self, sens: torch.Tensor) -> torch.LongTensor:
        """Split sensitive attribute training set

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

        # Shuffle and limit sensitive attribute training set size
        random.seed(self.seed)
        random.shuffle(sens_train_idx)
        sens_train_idx = sens_train_idx[:self.sens_number]

        return torch.LongTensor(sens_train_idx)
