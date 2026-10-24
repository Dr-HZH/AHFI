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

class GermanDataLoader(BaseDataLoader):
    """German credit dataset loader"""

    def __init__(self, sens_number: int = 200, label_number: int = 500, seed: int = 42, test_ratio: float = 0.2, val_ratio: float = 0.2):
        """
        Initialize German data loader

        Args:
            sens_number: Number of sensitive attribute training samples
            label_number: Number of label training samples
            seed: Random seed
            test_ratio: Test set ratio
            val_ratio: Validation set ratio
        """
        super().__init__(sens_number, label_number, seed)
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dataset", "german")
        self.sens_attr = "Gender"  # Sensitive attribute: gender
        self.predict_attr = "GoodCustomer"  # Prediction attribute: credit risk
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self._check_data_files()

    def _check_data_files(self):
        """Check if necessary data files exist"""
        required_files = {
            "german.csv": "German credit data file",
            "german_edges.txt": "German credit relationship data file"
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
        """Load German credit dataset

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

    def _log_split_stats(self, labels, sens, idx_train, idx_val, idx_test):
        """记录数据集划分统计信息"""
        total = len(labels)
        train_size = len(idx_train)
        val_size = len(idx_val)
        test_size = len(idx_test)

        self.logger.info(f"数据集划分: 总样本={total}, 训练集={train_size} ({train_size/total:.1%}), "
                        f"验证集={val_size} ({val_size/total:.1%}), 测试集={test_size} ({test_size/total:.1%})")

        # 计算各子集中的敏感属性分布
        for subset_name, subset_idx in [("训练集", idx_train), ("验证集", idx_val), ("测试集", idx_test)]:
            # 检查子集是否为空，跳过空集以避免除零错误
            if len(subset_idx) == 0:
                self.logger.warning(f"{subset_name}为空，跳过统计分析")
                continue

            s0_count = (sens[subset_idx] == 0).sum().item()
            s1_count = (sens[subset_idx] == 1).sum().item()
            l0_count = (labels[subset_idx] == 0).sum().item()
            l1_count = (labels[subset_idx] == 1).sum().item()

            self.logger.info(f"{subset_name} - 敏感属性=0: {s0_count} ({s0_count/len(subset_idx):.1%}), "
                           f"敏感属性=1: {s1_count} ({s1_count/len(subset_idx):.1%})")
            self.logger.info(f"{subset_name} - 标签=0: {l0_count} ({l0_count/len(subset_idx):.1%}), "
                           f"标签=1: {l1_count} ({l1_count/len(subset_idx):.1%})")

            # 检查每个敏感属性+标签组合的样本数
            s0_l0 = ((sens[subset_idx] == 0) & (labels[subset_idx] == 0)).sum().item()
            s0_l1 = ((sens[subset_idx] == 0) & (labels[subset_idx] == 1)).sum().item()
            s1_l0 = ((sens[subset_idx] == 1) & (labels[subset_idx] == 0)).sum().item()
            s1_l1 = ((sens[subset_idx] == 1) & (labels[subset_idx] == 1)).sum().item()

            self.logger.info(f"{subset_name} - 敏感属性=0,标签=0: {s0_l0}, 敏感属性=0,标签=1: {s0_l1}, "
                           f"敏感属性=1,标签=0: {s1_l0}, 敏感属性=1,标签=1: {s1_l1}")

    def _load_raw_data(self) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load raw data

        Returns:
            Tuple containing:
                - adj: Adjacency matrix
                - features: Node features
                - labels: Node labels
                - sens: Sensitive attributes
        """
        # Load German credit data
        data_path = os.path.join(self.data_dir, "german.csv")
        df = pd.read_csv(data_path)

        # Set user ID as index
        if 'user_id' not in df.columns:
            df['user_id'] = np.arange(len(df))

        # Get feature columns - exclude ID and label columns
        feature_cols = df.columns.tolist()
        feature_cols.remove('user_id')
        feature_cols.remove(self.sens_attr)
        feature_cols.remove(self.predict_attr)

        # Following fairgnn_utils.py processing, remove specific feature columns
        if 'OtherLoansAtStore' in feature_cols:
            feature_cols.remove('OtherLoansAtStore')
        if 'PurposeOfLoan' in feature_cols:
            feature_cols.remove('PurposeOfLoan')

        # Sensitive attribute processing - ensure binary encoding
        df[self.sens_attr] = df[self.sens_attr].map({'Female': 1, 'Male': 0})

        # Label processing - ensure -1 is replaced with 0 (consistent with fairgnn_utils.py)
        df[self.predict_attr] = df[self.predict_attr].replace(-1, 0)

        # Process numerical and categorical features separately
        df_features = df[feature_cols].copy()

        # Process categorical features (using one-hot encoding)
        categorical_cols = []
        for col in feature_cols:
            if df_features[col].dtype == 'object':  # If object type (string)
                self.logger.info(f"处理分类特征: {col}")
                categorical_cols.append(col)

        # Use pandas get_dummies method for one-hot encoding
        if categorical_cols:
            self.logger.info(f"对以下分类特征进行One-Hot编码: {categorical_cols}")
            df_encoded = pd.get_dummies(df_features, columns=categorical_cols, drop_first=False)
            self.logger.info(f"编码后特征数量: {df_encoded.shape[1]}")
        else:
            df_encoded = df_features

        # Ensure all features are numerical
        for col in df_encoded.columns:
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')

        # Replace NaN values
        df_encoded.fillna(0, inplace=True)

        # Convert features to sparse matrix
        features = sp.csr_matrix(df_encoded.values, dtype=np.float32)
        features = torch.FloatTensor(np.array(features.todense()))

        # Get labels and sensitive attributes
        sens = torch.FloatTensor(df[self.sens_attr].values)
        labels = torch.LongTensor(df[self.predict_attr].values)

        # Build graph
        idx = np.array(df['user_id'], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)}

        # Load edge data
        edges_path = os.path.join(self.data_dir, "german_edges.txt")
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
        """分割敏感属性训练集

        Args:
            sens: 敏感属性

        Returns:
            敏感属性训练集索引
        """
        # 转换为numpy数组
        sens_cpu = sens.cpu().numpy()

        # 获取有效敏感属性的索引
        sens_idx = set(np.where(sens_cpu >= 0)[0])

        # 从敏感属性索引中排除验证集和测试集索引
        idx_val_cpu = self.data.get('idx_val', torch.LongTensor([])).cpu().numpy()
        idx_test_cpu = self.data.get('idx_test', torch.LongTensor([])).cpu().numpy()

        sens_train_idx = list(sens_idx - set(idx_val_cpu) - set(idx_test_cpu))

        # 打乱并限制敏感属性训练集大小
        random.seed(self.seed)
        random.shuffle(sens_train_idx)
        sens_train_idx = sens_train_idx[:self.sens_number]

        return torch.LongTensor(sens_train_idx)
