import torch
import dgl
import numpy as np
from typing import Dict, Any, Tuple
from abc import ABC, abstractmethod

class BaseDataLoader(ABC):
    """Base data loader class"""

    def __init__(self, sens_number: int = None, label_number: int = None, seed: int = 42, test_ratio: float = 0.15, val_ratio: float = 0.15):
        """
        Initialize base data loader

        Args:
            sens_number: Number of sensitive attributes (deprecated, kept for compatibility)
            label_number: Number of labels (deprecated, kept for compatibility)
            seed: Random seed
            test_ratio: Test set ratio, default 0.15
            val_ratio: Validation set ratio, default 0.15
        """
        self.sens_number = sens_number
        self.label_number = label_number
        self.seed = seed
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        np.random.seed(seed)
        torch.manual_seed(seed)

    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """Load dataset"""
        pass

    def to_device(self, data: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        """Move data to specified device"""
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
            elif isinstance(data[key], dgl.DGLGraph):
                data[key] = data[key].to(device)
        return data

    def split_data(self, labels, sens, test_idx=False):
        """Disjoint class-stratified splits; defaults follow the paper's 70/15/15."""
        from sklearn.model_selection import train_test_split
        if test_idx:
            raise ValueError("Overlapping validation/test sets are not supported")
        labels_np = labels.detach().cpu().numpy()
        valid = np.flatnonzero(labels_np >= 0)
        heldout_ratio = self.val_ratio + self.test_ratio
        if not (self.val_ratio > 0 and self.test_ratio > 0 and heldout_ratio < 1):
            raise ValueError("Validation/test ratios must be positive and sum to less than 1")
        train, heldout = train_test_split(
            valid, test_size=heldout_ratio, random_state=self.seed,
            stratify=labels_np[valid])
        val, test = train_test_split(
            heldout, test_size=self.test_ratio / heldout_ratio,
            random_state=self.seed, stratify=labels_np[heldout])
        return tuple(torch.as_tensor(idx, dtype=torch.long) for idx in (train, val, test))

    def split_sens(self, sens: torch.Tensor) -> torch.Tensor:
        """Split sensitive attribute training set"""
        # Get all sample indices
        all_idx = np.arange(len(sens))

        # Shuffle indices, no longer limiting quantity
        np.random.shuffle(all_idx)

        return torch.LongTensor(all_idx)
