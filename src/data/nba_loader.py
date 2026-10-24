import os
import torch
import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Dict, Any, Tuple, List, Union
from .base_loader import BaseDataLoader
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import random

class NBADataLoader(BaseDataLoader):
    """NBA dataset loader"""

    def __init__(self, sens_number: int = 50, label_number: int = 100, seed: int = 20, test_ratio: float = 0.15, val_ratio: float = 0.15):
        """
        Initialize NBA data loader

        Args:
            sens_number: Number of sensitive attribute categories
            label_number: Number of labels
            seed: Random seed
            test_ratio: Test set ratio, default 0.15 (small to increase training set)
            val_ratio: Validation set ratio, default 0.15 (small to increase training set)
        """
        super().__init__(sens_number, label_number, seed)
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dataset", "NBA")
        self.sens_attr = "country"  # Sensitive attribute: country
        self.predict_attr = "SALARY"  # Prediction attribute: salary
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self._check_data_files()

    def _check_data_files(self):
        """Check if necessary data files exist"""
        required_files = {
            "nba.csv": "Player data file",
            "nba_relationship.txt": "Player relationship data file",
            "nba.embedding": "Player embedding file"
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
        """Load NBA dataset

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
        # Load graph embedding
        graph_embedding = np.genfromtxt(
            os.path.join(self.data_dir, "nba.embedding"),
            skip_header=1,
            dtype=float
        )
        embedding_df = pd.DataFrame(graph_embedding)
        embedding_df[0] = embedding_df[0].astype(int)
        embedding_df = embedding_df.rename(index=int, columns={0:"user_id"})

        # Load player data and merge embeddings
        idx_features_labels = pd.read_csv(os.path.join(self.data_dir, "nba.csv"))

        # Ensure ID column name is correct
        if "user_id" in idx_features_labels.columns:
            id_column = "user_id"
        else:
            id_column = idx_features_labels.columns[0]  # Assume first column is ID
            # Rename first column to user_id for merging
            idx_features_labels = idx_features_labels.rename(columns={id_column: "user_id"})

        idx_features_labels = pd.merge(idx_features_labels, embedding_df, how="left", on="user_id")
        idx_features_labels = idx_features_labels.fillna(0)

        # Get feature columns
        header = list(idx_features_labels.columns)
        header.remove("user_id")
        header.remove(self.sens_attr)
        header.remove(self.predict_attr)

        # Build feature matrix
        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[self.predict_attr].values

        # Binarize labels: following original FairGNN implementation
        labels[labels > 1] = 1

        # Build graph
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)}

        # Load relationship data
        edges_unordered = np.genfromtxt(os.path.join(self.data_dir, "nba_relationship.txt"), dtype=int)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                        dtype=int).reshape(edges_unordered.shape)

        # Build adjacency matrix
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                          shape=(labels.shape[0], labels.shape[0]),
                          dtype=np.float32)

        # Build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])

        # Convert to tensors
        features = torch.FloatTensor(np.array(features.todense()))
        features = self._normalize_features(features.numpy())
        features = torch.FloatTensor(features)

        labels = torch.LongTensor(labels)
        sens = torch.FloatTensor(idx_features_labels[self.sens_attr].values)

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

    def split_data(self, labels, sens):
        """Use the common class-stratified paper protocol."""
        return super().split_data(labels, sens)

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

        # Shuffle and limit sensitive attribute training set size
        random.seed(self.seed)
        random.shuffle(sens_train_idx)
        sens_train_idx = sens_train_idx[:self.sens_number]

        return torch.LongTensor(sens_train_idx)
